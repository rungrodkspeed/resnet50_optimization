import os
import torch
import numpy as np
import pandas as pd

from torchvision import transforms
from torch.utils.data import DataLoader
from test.generator import TestGenerator
from test.utils import confusion_matrix
from models.resnet50 import resnet50


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


root_dir = './flower_dataset/test'
csv_file_path = './flower_dataset/test.csv'
checkpoint_path = './checkpoints/checkpoint_epoch_500_acc_0.8472727272727273.pth'

test_dataset = TestGenerator(root_dir=root_dir, csv_file=csv_file_path, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=16)

checkpoint = torch.load(checkpoint_path)

model = resnet50()
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
model = model.to('cuda')

classes = {class_label:class_name for class_label, class_name in enumerate(os.listdir('./flower_dataset/train'))}

preds = []
for images in test_loader:
    images = images.to('cuda')
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    preds.extend(predicted.cpu().numpy())


gt_df = pd.read_csv(csv_file_path)['mylabels']
gt = [next((key for key, value in classes.items() if value == flower), None) for flower in gt_df]

cm = confusion_matrix(gt, preds)

print(cm)

print(f'accuracy : {np.trace(cm) / np.sum(cm)}')