import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CamDataset(Dataset):
    
    def __init__(self, csvPath, rootDir, transform=None):
        self.imgs = pd.read_csv(csvPath)
        self.rootDir = rootDir
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgName, label = self.imgs.iloc[index]
        imgPath = os.path.join(self.rootDir, imgName)
        img = self.loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
        
    @staticmethod
    def loader(path):
        return Image.open(path).convert('RGB')

