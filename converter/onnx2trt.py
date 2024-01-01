import os
import time
import numpy as np
import tensorrt as trt

import pycuda.autoinit

import pycuda.driver as cuda

_dtype_trt={'INT8':trt.BuilderFlag.INT8,
            'HALF':trt.BuilderFlag.FP8,
            'FLOAT':trt.BuilderFlag.FP16,
            'DOUBLE':trt.BuilderFlag.TF32,}

_dtype_np={'INT8':np.int8,
           'HALF':np.half,
           'FLOAT':np.float32,
           'DOUBLE':np.float64,}

def _load_engine(path, trt_compat=False):
    TRT_LOGGER = trt.Logger()

    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        
        if trt_compat : runtime.engine_host_code_allowed = True
            
        engine = runtime.deserialize_cuda_engine(f.read())

    return engine

def convert_to_trt(args):
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
                exit()
        print("Succeeded parsing .onnx file!")


    inpt = network.get_input(0)
    
    if args.dynamic_batch:
        profile.set_shape(inpt.name, \
                        (args.min_batch_size, 3, args.img_size, args.img_size), \
                        (args.opt_batch_size, 3, args.img_size, args.img_size), \
                        (args.max_batch_size, 3, args.img_size, args.img_size))

    config.add_optimization_profile(profile)


    config.set_flag(_dtype_trt[args.dtype])

    if args.trt_compat:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    
    
    name, ext = os.path.basename(args.trt_path).split('.')
    out_path = os.path.join(os.path.dirname(args.trt_path), f"{name}_{args.dtype}.{ext}")
    with open(out_path, "wb") as f:
        f.write(engineString)
    
    dummy_data = np.zeros((args.opt_batch_size, 3, args.img_size, args.img_size), dtype=_dtype_np[args.dtype])
    
    _inference(out_path, dummy_data, args.trt_compat)

def _inference(path, data , trt_compat=False):
        
    engine = _load_engine(path, trt_compat)
    context = engine.create_execution_context()
    
    assert not (context is None), "Context is None."
    
    output_shape = list(context.get_tensor_shape('output'))
    output_shape[0] = data.shape[0]
    output_shape = tuple(output_shape)
    
    buffer = []
    for _, binding in enumerate(engine):
        mode = engine.get_tensor_mode(binding)
        if mode.name == 'INPUT':
            context.set_input_shape(binding, data.shape)
            size = trt.volume(data.shape) * data.dtype.itemsize
            device_input = cuda.mem_alloc(size)
            cuda.memcpy_htod(device_input, data.tobytes())
            buffer.append(device_input)
        else:
            size = trt.volume(output_shape) * data.dtype.itemsize
            device_output = cuda.mem_alloc(size)
            buffer.append(device_output)

    host_output = np.empty(output_shape, dtype=data.dtype)
    
    start_time = time.time()
    
    context.execute_v2(buffer)
    
    cuda.memcpy_dtoh(host_output, buffer[-1])
    
    inference_time = time.time() - start_time
    
    print("TensorRT :")
    print(f'batch size : {data.shape[0]}')
    print(f'time inferences : {inference_time}')
    print(f'FPS : {1 / inference_time * data.shape[0]}')