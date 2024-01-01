import time
import torch
import argparse

from PIL import Image
from utils import classes
from torchvision import transforms
from models.resnet50 import resnet50

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _preprocess(image, transform):
    return transform(image)

def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

def inference(model, sample, device):
    
    inpt = torch.unsqueeze(_preprocess(sample, data_transforms), dim=0).to(device)
    
    start_time = time.time()
    out = model(inpt)
    inference_time = time.time() - start_time
    print(f'Inference with {device}: {inference_time} sec.')
    print(f'FPS : { 1 / inference_time * inpt.shape[0]}')
    
    _, pred = torch.max(out, 1)
    scores = torch.softmax(out, dim=1)
    
    return classes[int( pred.cpu().item() )], scores[0, pred].cpu().item()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Torch Inference.")
    parser.add_argument('--sample', type=str, default='./sample/daisy.jpg', help='image path')
    parser.add_argument('--model_path', type=str, default='./checkpoints/checkpoint_epoch_500_acc_0.8472727272727273.pth', help='model path')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU')
    args = parser.parse_args()
    
    
    sample = Image.open(args.sample).convert('RGB')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    model = load_model(resnet50(), args.model_path, device)
    
    print('Torch Inferencing.')
    res, confidence = inference(model, sample, device)
    print(f'specie : {res} with confidence {confidence * 100} %')