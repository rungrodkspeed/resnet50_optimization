import os

from torchvision import transforms

from train import Trainer

data_dir = './flower_dataset'
train_dir = os.path.join(data_dir, 'train')

data_transorms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


trainer_model = Trainer(train_dir, transform=data_transorms, checkpoint_path='checkpoints', patience=5)

trainer_model.train(epochs=1000)