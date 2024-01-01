import argparse

from converter.onnx2trt import convert_to_trt
from converter.torch2onnx import convert_to_onnx
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Exporter.")
    parser.add_argument('--mode', type=str, choices=['torch2onnx', 'onnx2trt'], default='torch2onnx', help='MODE for converting.')
    
    #torch2onnx part
    parser.add_argument('--model_path', type=str, default='./checkpoints/checkpoint_epoch_500_acc_0.8472727272727273.pth', help='torch Path.')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size.')
    parser.add_argument('--onnx_path', type=str, default='./onnx_checkpoints/resnet50.onnx', help='onnx Path.')

    #onnx2tensorrt part
    parser.add_argument('--img_size', type=int, default=224, help='image size.')
    parser.add_argument('--min_batch_size', type=int, default=1, help='min BATCH SIZE.')
    parser.add_argument('--opt_batch_size', type=int, default=8, help='opt BATCH SIZE.')
    parser.add_argument('--max_batch_size', type=int, default=128, help='max BATCH SIZE.')
    parser.add_argument('--trt_path', type=str, default='./tensorrt_checkpoints/resnet50.plan', help='tensorrt Path.')
    parser.add_argument('--trt_compat', action='store_true', default=False, help='set version compatibility.')
    
    
    parser.add_argument('--dynamic_batch', action='store_true', default=False, help='dynamic axes.')
    parser.add_argument('--dtype', type=str, choices=['INT8', 'HALF', 'FLOAT', 'DOUBLE'],default='FLOAT', help='select data type.')
        
    args = parser.parse_args()
    
    if args.mode == 'torch2onnx':
        convert_to_onnx(args)
    elif args.mode == 'onnx2trt':
        convert_to_trt(args)