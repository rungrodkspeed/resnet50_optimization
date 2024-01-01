import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from models.resnet50 import resnet50
from .generator import Generator

class Trainer:
    def __init__(self, root_dir, transform, train_split=0.8, checkpoint_path='checkpoints', patience=5):
        self.root_dir = root_dir
        self.transform = transform
        self.train_split = train_split
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.dataset = Generator(root_dir=self.root_dir, transform=self.transform, train_split=self.train_split)
        self.train_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader([self.dataset.get_validation_item(idx) for idx in range(len(self.dataset.val_dataset))], batch_size=32)
      
        self.model = resnet50()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def train(self, epochs):
        for epoch in range(epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    data_loader = self.train_loader
                else:
                    self.model.eval()
                    data_loader = self.val_loader
                    
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / len(self.dataset.train_dataset if phase == 'train' else self.dataset.val_dataset)
                epoch_acc = running_corrects.double() / len(self.dataset.train_dataset if phase == 'train' else self.dataset.val_dataset)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if epoch%100 == 0 and epoch > 0:
                self.save_checkpoint(epoch, epoch_acc)

            if phase == 'val':
                if self.best_acc is None or epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.counter = 0
                else: 
                    self.counter = 1
                    if self.counter >= self.patience:
                        print('Early stopping triggered !')
                        self.save_checkpoint(epoch, epoch_acc)
                        return 

        self.save_checkpoint(epochs, epoch_acc)
        print("Training complete!")
        
    
    def save_checkpoint(self, epoch, acc):
        checkpoint_filename = os.path.join(self.checkpoint_path, f'checkpoint_epoch_{epoch}_acc_{acc}.pth')
        torch.save({
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc':acc
        }, checkpoint_filename)