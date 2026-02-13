import random
from torchvision.transforms import v2
import cv2
import torch
import numpy as np
from torchvision.transforms import functional


class Preprocessor:
    def __init__(self, seed=None):
        self.seed = seed
        self.set_seed()


    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                
    def augment_transform1(self):
        """Transform which does both random cropping and rotation along with preprocessing"""
        transform = v2.Compose([
        v2.Resize((512, 512)), 
        v2.Lambda(lambda img: img.convert("RGB")),  
        v2.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)), 
        v2.Lambda(lambda img: cv2.split(img)[1]), # Extracting the green channel
        v2.Lambda(lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)),  
        v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0)/255.0), # Adding a channel dimension 
        v2.Lambda(lambda img: img.expand(3, -1, -1)),  # Expanding to 3 channels
        v2.RandomCrop((256, 256)), 
        v2.RandomRotation((0, 360)),
        v2.GaussianNoise(mean=0.0, sigma=0.05, clip=True),
        v2.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224)),])
        return transform
    
    def augment_transform2(self):
        """Transform which only does preprocessing and random rotation"""
        transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),
        v2.Lambda(lambda img: cv2.split(img)[1]), # Extracting the green channel
        v2.Lambda(lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)),
        v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0)/255.0), # Adding a channel dimension
        v2.Lambda(lambda img: img.expand(3, -1, -1)), # Expanding to 3 channels
        v2.RandomRotation((0, 360)),
        v2.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224)),])
        return transform
    
    def test_transform(self):
        """Transform which only does preprocessing"""
        transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),
        v2.Lambda(lambda img: cv2.split(img)[1]), # Extracting the green channel
        v2.Lambda(lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)),
        v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0)/255.0), # Adding a channel dimension
        v2.Lambda(lambda img: img.expand(3, -1, -1)), # Expanding to 3 channels
        v2.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224)),])
        return transform
