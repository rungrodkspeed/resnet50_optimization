import os

from PIL import Image
from torch.utils.data import Dataset, random_split


class Generator(Dataset):
    
    def __init__(self, root_dir, transform=None, train_split=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.train_split = train_split
        
        self.image_paths = []
        self.labels = []
        self.classes = []
        
        for class_label, class_name in enumerate(os.listdir(self.root_dir)):
            self.classes.append(class_name)
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_label)
                    
        dataset_size = len(self.image_paths)
        train_size = int(dataset_size * self.train_split)
        val_size = dataset_size - train_size
        
        self.train_dataset, self.val_dataset = random_split(list(zip(self.image_paths, self.labels)), [train_size, val_size])
        
    def __len__(self):
        return len(self.train_dataset)
    
    
    def __getitem__(self, idx):
        
        img_path, label = self.train_dataset[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform['train'](image)
        
        return image, label
    
    def get_validation_item(self, idx):
        
        img_path, label = self.val_dataset[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform['val'](image)
        
        return image, label
    