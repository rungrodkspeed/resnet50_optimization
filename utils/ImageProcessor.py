import pycuda.autoinit

import os
import cv2
import asyncio
import aiofiles
import numpy as np
import concurrent.futures
import pycuda.driver as cuda

from .Kernel import kernel_code

from PIL import Image
from pycuda.compiler import SourceModule

class ImageProcessor:
    
    def __init__(self, paths, batch_size, dtype=np.float32):
        
        self.mean = np.array( [0.485, 0.456, 0.406] ).reshape((3, 1, 1))
        self.std = np.array( [0.229, 0.224, 0.225] ).reshape((3, 1, 1))
        
        self.dtype = dtype
        self.batch_size = batch_size
        self.num_processes = os.cpu_count()
        
        self.paths = [paths[i:i+self.batch_size] for i in range(0, len(paths), self.batch_size)]
            
    async def _read_async(self, path):
        async with aiofiles.open(path, "rb") as file:
            content = await file.read()
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # -----> this line using RGB formatted.
            return image
           
    async def _process(self, batch_paths):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            tasks = [self._read_async(path) for path in batch_paths]
            images = await asyncio.gather(*tasks)
            processed_images = list(executor.map(self.transforms, images))
            return np.array(processed_images, dtype=self.dtype)
        
    def transforms(self, image):
        resized_image = cv2.resize(image, (256, 256))
        cropped_image = resized_image[16:240, 16:240]
        tensor_image = cropped_image.transpose((2, 0, 1))
        normalized_image = ( ( tensor_image / 255.0 ) -  self.mean) / self.std
        return normalized_image


    def process(self):
        processed_batches = []
        for batch_paths in self.paths:
            processed_batch = asyncio.run(self._process(batch_paths))
            processed_batches.append(processed_batch)
        return processed_batches


class ImageProcessorDevice:
    
    def __init__(self, image : Image, dtype:np.dtype):
        self.dtype = dtype
        self.device = cuda.Device(0).name()
    
        self.host_data : np.ndarray = ( np.ascontiguousarray(image) / 255. ).astype(dtype)
        self.height, self.width, self.channels = self.host_data.shape
        
        self.host_pinned = cuda.pagelocked_empty((self.height, self.width, self.channels), dtype=self.dtype)
        self.host_pinned[:] = self.host_data
        
        self.ptr = cuda.mem_alloc( self.host_pinned.nbytes )
        
        self.module = SourceModule(kernel_code)
        
        self.pipeline = ['resize', 'center_crop', 'transpose', 'normalize']
        self.target_size = [256, 224, 224, 224]
        
        self.blocks = [(32, 32, 1), (32, 32, 1), (32, 32, 1), (1024, 1, 1)]
        self.grids = [
            ((self.target_size[0] - 1) // self.blocks[0][0] + 1, (self.target_size[0] - 1) // self.blocks[0][1] + 1, 1),
            ((self.target_size[0] - 1) // self.blocks[1][0] + 1, (self.target_size[0] - 1) // self.blocks[1][1] + 1, 1),
            ((self.target_size[1] - 1) // self.blocks[2][0] + 1, (self.target_size[1] - 1) // self.blocks[2][1] + 1, self.channels),
            ((self.target_size[2] * self.target_size[2] + self.blocks[2][0] - 1) // self.blocks[3][0], 1, 1)]

        self.args = [[np.int32(self.channels), np.int32(self.width), np.int32(self.height), np.int32(self.target_size[0]), np.int32(self.target_size[0])],
                    [np.int32(self.channels), np.int32(self.target_size[0]), np.int32(self.target_size[0]), np.int32(self.target_size[1])],
                    [np.int32(self.channels), np.int32(self.target_size[1]), np.int32(self.target_size[1])],
                    [np.int32(self.target_size[2] * self.target_size[2])]]

        self.out_pinned = cuda.pagelocked_empty((self.channels, self.target_size[-1], self.target_size[-1]), dtype=self.dtype)
        self.out_ptr = None
        
    def process(self):
        
        cuda.memcpy_htod(self.ptr, self.host_pinned)
        
        for i, f in enumerate(self.pipeline):
            
            cuda_function = self.module.get_function(f)
            
            ds = self.target_size[i] * self.target_size[i] * self.channels * np.dtype(self.dtype).itemsize
            
            self.out_ptr = cuda.mem_alloc( ds )
            cuda_function(self.out_ptr, self.ptr, *self.args[i] , block=self.blocks[i], grid=self.grids[i])
            
            self.ptr = self.out_ptr
                 
        cuda.memcpy_dtoh(self.out_pinned, self.out_ptr)
        
    def process_async(self, num_stream=1):
        
        stream = [cuda.Stream() for _ in range(num_stream)]
        
        cuda.memcpy_htod_async(self.ptr, self.host_pinned, stream[0])
        
        for i, f in enumerate(self.pipeline):
            
            cuda_function = self.module.get_function(f)
            
            ds = self.target_size[i] * self.target_size[i] * self.channels * np.dtype(self.dtype).itemsize
            
            self.out_ptr = cuda.mem_alloc( ds )
            cuda_function(self.out_ptr, self.ptr, *self.args[i] , block=self.blocks[i], grid=self.grids[i], stream=stream[i % num_stream])
            
            self.ptr = self.out_ptr
                 
        cuda.memcpy_dtoh_async(self.out_pinned, self.out_ptr, stream[-1])
        
        for s in stream:
            s.synchronize()
            