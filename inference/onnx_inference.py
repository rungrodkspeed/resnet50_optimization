import sys

sys.path.append('../')

import time
import argparse
import numpy as np
import onnxruntime as ort

from utils import ImageProcessor, get_path, classes, EPSILON


def load_model(args):
    
    providers=['CPUExecutionProvider']
    
    if args.gpu:
        providers.extend(['CUDAExecutionProvider'])

    return ort.InferenceSession(args.model_path, providers=providers)

def softmax(x, axis=None):
    if axis is None:
        axis = -1

    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)

    return exp_x / ( np.sum(exp_x, axis=axis, keepdims=True) + EPSILON)

def inference(model, sample):
    
    ort_input = [{model.get_inputs()[0].name: s} for s in sample] 

    start_time = time.time()
    ort_out = [model.run(None, inpt)[0] for inpt in ort_input]
    
    
    inference_time = time.time() - start_time
    print(f'Inference with {model.get_providers()}: {inference_time} sec.')
    n_frame = (len(sample) - 1) * sample[0].shape[0] + sample[-1].shape[0]
    print(f'num frames : {n_frame}')
    print(f'FPS : {1 / inference_time * n_frame}')


    #print(ort_out)
    pred = np.argmax(ort_out, axis=2).flatten()
    scores = softmax(ort_out, axis=2)
    b, _, f = scores.shape
    scores = scores.reshape((b,f))
    
    return [(classes[int( pred[i] )], scores[i, int(pred[i])]) for i in range(len(pred))]


def main(args):
    session = load_model(args)
    
    image_paths = get_path(args.image_path)
        
    preprocessor = ImageProcessor(paths=image_paths , batch_size=args.batch_size, dtype=np.float32)
    sample = preprocessor.process()
        
    print('ONNX Inferencing.')
    res = inference(session, sample)
    
    for i, (cls, confidence) in enumerate(res):
        print(f'({i+1}) specie : {cls}, confidence {confidence * 100} %')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("ONNX Inference.")
    parser.add_argument('--sample', type=str, default='../sample/daisy.jpg', help='image path')
    parser.add_argument('--image_path', type=str, default='../sample', help='folder of image path')
    parser.add_argument('--model_path', type=str, default='../checkpoints/resnet50.onnx', help='model path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU')
    
    args = parser.parse_args()
    
    main(args)