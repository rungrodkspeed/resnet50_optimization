import gc
import time
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

import pycuda.autoinit

from utils import get_path, classes, EPSILON, dtype_np
from ImageProcessor.ImageProcessor import ImageProcessor


def load_engine(path):
    TRT_LOGGER = trt.Logger()

    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    return engine

def softmax(x, axis=None):
    if axis is None:
        axis = -1

    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)

    return exp_x / ( np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)

def deallocate(context, buffer):
    
    for mem_allocate in buffer:
        mem_allocate.free()
    
    del context
    
    gc.collect()
    
def _inference_async(engine, context, data, output_shape, stream=None):
    
    buffer = []
    for _, binding in enumerate(engine):
        mode = engine.get_tensor_mode(binding)
        if mode.name == 'INPUT':
            context.set_input_shape(binding, data.shape)
            size = trt.volume(data.shape) * data.dtype.itemsize
            device_input = cuda.mem_alloc(size)
            cuda.memcpy_htod_async(device_input, data.tobytes(), stream = stream)
            buffer.append(device_input)
        else:
            size = trt.volume(output_shape) * data.dtype.itemsize
            device_output = cuda.mem_alloc(size)
            buffer.append(device_output)

    host_output = cuda.pagelocked_empty(output_shape, dtype=data.dtype)
    
    context.execute_async_v2(bindings=buffer, stream_handle = stream.handle)
    
    cuda.memcpy_dtoh_async(host_output, buffer[-1], stream = stream)
    
    stream.synchronize()
    
    return host_output, buffer

def inference_async(engine, data, num_stream=1):
        
    context = engine.create_execution_context()
    
    assert not (context is None), "Context is None."
    
    stream = [cuda.Stream() for _ in range(num_stream)]
    
    start_time = time.time()
    
    res, buffers = [], []
    for i, batch in enumerate(data):
    
        output_shape = list(context.get_tensor_shape('output'))
        output_shape[0] = batch.shape[0]
        output_shape = tuple(output_shape)
        
        host_pinned = cuda.pagelocked_empty(batch.shape, dtype=batch.dtype)
        host_pinned[:] = batch
        
        host_output, buffer = _inference_async(engine, context, host_pinned, output_shape, stream=stream[i % num_stream])
        res.extend(host_output)
        buffers.extend(buffer)
        
    inference_time = time.time() - start_time    
        
    print(f'Inference : {inference_time} sec.')
    n_frame = (len(data) - 1) * data[0].shape[0] + data[-1].shape[0]
    print(f'num frames : {n_frame}')
    print(f'FPS : {1 / inference_time * n_frame}')
    
    
    pred = np.argmax(res, axis=1)
    scores = softmax(res, axis=1)
    
    deallocate(context, buffers)

    return [(classes[int( pred[i] )], scores[i, int(pred[i])]) for i in range(len(pred))]


def _inference(engine, context, data, output_shape):
    
    buffer = []
    for _, binding in enumerate(engine):
        mode = engine.get_tensor_mode(binding)
        if mode.name == 'INPUT':
            context.set_input_shape(binding, data.shape)
            size = trt.volume(data.shape) * data.dtype.itemsize
            device_input = cuda.mem_alloc(size)
            cuda.memcpy_htod_async(device_input, data.tobytes())
            buffer.append(device_input)
        else:
            size = trt.volume(output_shape) * data.dtype.itemsize
            device_output = cuda.mem_alloc(size)
            buffer.append(device_output)

    host_output = cuda.pagelocked_empty(output_shape, dtype=data.dtype)
    
    context.execute_v2(bindings=buffer)
    
    cuda.memcpy_dtoh_async(host_output, buffer[-1])
    
    return host_output, buffer

def inference(engine, data):
        
    context = engine.create_execution_context()
    
    assert not (context is None), "Context is None."
        
    start_time = time.time()
    
    res, buffers = [], []
    for batch in data:
    
        output_shape = list(context.get_tensor_shape('output'))
        output_shape[0] = batch.shape[0]
        output_shape = tuple(output_shape)
        
        host_pinned = cuda.pagelocked_empty(batch.shape, dtype=batch.dtype)
        host_pinned[:] = batch
        
        host_output, buffer = _inference(engine, context, host_pinned, output_shape)
        res.extend(host_output)
        buffers.extend(buffer)
        
    inference_time = time.time() - start_time    
        
    print(f'Inference : {inference_time} sec.')
    n_frame = (len(data) - 1) * data[0].shape[0] + data[-1].shape[0]
    print(f'num frames : {n_frame}')
    print(f'FPS : {1 / inference_time * n_frame}')
    
    
    pred = np.argmax(res, axis=1)
    scores = softmax(res, axis=1)
    
    deallocate(context, buffers)

    return [(classes[int( pred[i] )], scores[i, int(pred[i])]) for i in range(len(pred))]


def main(args):
    
    engine = load_engine(args.model_path)
    
    img_paths = get_path(args.image_path)
    
    preprocessor = ImageProcessor(img_paths, batch_size=args.batch_size, dtype=dtype_np[args.dtype])
    sample = preprocessor.process()
    
    print('TensorRT Inferencing.')
    
    if args.use_async:
        res = inference_async(engine, sample, num_stream=args.num_stream)
    else:
        res = inference(engine, sample)
            
    for i, (cls, confidence) in enumerate(res):
        print(f'({i+1}) specie : {cls}, confidence {confidence * 100} %')
    
    del engine
    gc.collect()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("TensorRT Inference.")
    parser.add_argument('--sample', type=str, default='./sample/daisy.jpg', help='image path')
    parser.add_argument('--image_path', type=str, default='./sample', help='folder of image path')
    parser.add_argument('--model_path', type=str, default='./tensorrt_checkpoints/resnet50_FLOAT.plan', help='model path')
    parser.add_argument('--dtype', type=str, choices=['INT8', 'HALF', 'FLOAT', 'DOUBLE'],default='FLOAT', help='select data type.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    
    parser.add_argument('--use_async', action='store_true', default=False, help='Asynchronous process.')
    parser.add_argument('--num_stream', type=int, default=1, help='number of stream.')
    
    args = parser.parse_args()
        
    main(args)