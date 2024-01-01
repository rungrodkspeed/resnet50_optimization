import os 
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class TestGenerator(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.image_file = pd.read_csv(csv_file)['filename']
        self.transform = transform
        
    def __len__(self):
        return len(self.image_file)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_file.iloc[idx])
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image