import gc
import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

import pycuda.autoinit

def generate_data(tensor_size, batches = 100, dtype=np.float32):
    
    if dtype == np.uint8:
        
        data = np.random.randint(0, 256, size=(batches, *tensor_size), dtype=dtype)
    
    else:
    
        data = np.random.random(size=(batches, *tensor_size)).astype(dtype)
    
    return data

def load_torch(checkpoint_path, gpu=False):
    
    from models import resnet50
    
    model = resnet50()
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device), device

def load_session(checkpoint_path, gpu=False):
    
    import onnxruntime as ort
    
    providers=['CPUExecutionProvider']
    
    if gpu:
        providers.extend(['CUDAExecutionProvider'])

    return ort.InferenceSession(checkpoint_path, providers=providers)

def load_engine(path):
        
    TRT_LOGGER = trt.Logger()

    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    return engine

def deallocate(context, buffer):
    
    for mem_allocate in buffer:
        mem_allocate.free()
    
    del context
    
    gc.collect()

def bench_torch(model, device, num_data = 200):
    
    from torchvision import transforms
    
    data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_data = generate_data(tensor_size=(224,224,3), batches=num_data, dtype=np.uint8)
    
    preprocess = torch.stack([data_transforms(image) for image in raw_data]).to(device)
    
    start_time = time.time()
    _ = model(preprocess)
    inference_time = time.time() - start_time
    print(f'Inference with {device}: {inference_time} sec.')
    print(f'FPS : { 1 / inference_time * num_data}')


def bench_onnx(session, num_data=200):
    
    
    sample = generate_data(tensor_size=(3,224,224), batches=num_data)
    
    ort_input = [{session.get_inputs()[0].name: sample}]
        
    start_time = time.time()
    _ = [session.run(None, inpt)[0] for inpt in ort_input]
    inference_time = time.time() - start_time
    print(f'FPS : {1 / inference_time * num_data}')

   
def _inferenceTRT(engine, context, data, output_shape):
    
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
    
    
    start_time = time.time()
    context.execute_v2(bindings=buffer)
    timer = time.time() - start_time
    
    cuda.memcpy_dtoh_async(host_output, buffer[-1])
    
    return host_output, buffer, timer

def bench_trt(engine, num_data=200):
        
    context = engine.create_execution_context()
    
    assert not (context is None), "Context is None."
    
    data = generate_data(tensor_size=(3,224,224), batches=num_data)
    
    buffers = []

    output_shape = list(context.get_tensor_shape('output'))
    output_shape[0] = data.shape[0]
    output_shape = tuple(output_shape)
    
    host_pinned = cuda.pagelocked_empty(data.shape, dtype=data.dtype)
    host_pinned[:] = data
    
    _, buffer, inference_time = _inferenceTRT(engine, context, host_pinned, output_shape)

    buffers.extend(buffer)
    
    deallocate(context, buffers)

    print(f'FPS : {1 / inference_time * num_data}')



def main(mode, num_data, gpu=False):
    
    if mode == 'torch':
        model, device = load_torch('./checkpoints/checkpoint_epoch_500_acc_0.8472727272727273.pth', gpu=gpu)
        bench_torch(model, device, num_data=num_data)
        
    if mode == 'onnx':
        session = load_session('./onnx_checkpoints/resnet50.onnx', gpu=gpu)
        bench_onnx(session, num_data=num_data)
        
        
    if mode == 'tensorrt':
        engine = load_engine('./tensorrt_checkpoints/resnet50_FLOAT.plan')
        bench_trt(engine, num_data=num_data)

if __name__ == '__main__':
    
    main('tensorrt', 8)