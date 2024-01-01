import time
import torch
import argparse
import onnxruntime

from models.resnet50 import resnet50

_dtype={'INT8':torch.int8,
        'HALF':torch.float16,
        'FLOAT':torch.float32,
        'DOUBLE':torch.float64,}

def _load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

def _inference(path, dummy_data):
    
    session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider','CUDAExecutionProvider'])

    ort_input = {session.get_inputs()[0].name: dummy_data}
    
    start_time = time.time()
    ort_out = session.run(None, ort_input)
    
    inference_time = time.time() - start_time
    
    print("ONNX :")
    print(f'batch size : {dummy_data.shape[0]}')
    print(f'time inferences : {inference_time}')
    print(f'FPS : {1 / inference_time * dummy_data.shape[0]}')
    print(f'provider : {session.get_providers()}')

def convert_to_onnx(args):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    input_shape = (args.batch_size, 3, 224, 224)
    model = _load_model(resnet50(), args.model_path, device)

    dummy_data = torch.zeros(size=input_shape, dtype=_dtype[args.dtype]).to(device)

    if args.dynamic_batch:

        dynamic_axes_dict = {'input': {0:'batch_size',
                                       2:'img_x',
                                       3:'img_y',},
                            
                            'output': {0:'batch_size'}
                            }

        torch.onnx.export(model,
            dummy_data,
            args.onnx_path,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
            )
        
    else:        
        torch.onnx.export(model,
                        dummy_data,
                        args.onnx_path,
                        export_params=True,
                        input_names=['input'],
                        output_names=['output']
                        )
        
    _inference(args.onnx_path, dummy_data.cpu().numpy() if args.gpu else dummy_data.numpy())