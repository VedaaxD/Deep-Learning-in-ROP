from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from data_preprocess import Preprocessor

# For augmentation of the training dataset
class AugmentedDataset(Dataset):
    
    def __init__(self, root_dir, augment_repeats=8, augment=False, seed=None):
        self.root = root_dir
        preprocessor = Preprocessor(seed=seed)
        self.transform1 = preprocessor.augment_transform1()
        self.transform2 = preprocessor.augment_transform2()
        self.transform3 = preprocessor.test_transform()
        self.augment_repeats = augment_repeats 
        self.augment = augment

        negative_path = os.path.join(self.root, "Negative")
        positive_path = os.path.join(self.root, "Positive")

        self.negative_imgs = [os.path.join(negative_path, x) for x in os.listdir(negative_path) if not x.startswith(".")]
        self.positive_imgs = [os.path.join(positive_path, x) for x in os.listdir(positive_path) if not x.startswith(".")]

        self.data = [(img, 0) for img in self.negative_imgs] + [(img, 1) for img in self.positive_imgs]

    def __len__(self):
        return len(self.data) * self.augment_repeats  

    def __getitem__(self, idx):
        
        img_path, label = self.data[idx % len(self.data)]  
        img = Image.open(img_path)
        if self.augment:
            transform = self.transform1 if torch.rand(1).item() > 0.5 else self.transform2  
        else:
            transform = self.transform3
        img = transform(img)

        return img, label


# Custom dataset loader for testing
class TestDataset(Dataset):
    def __init__(self, root_dir, seed=None):
        self.root = root_dir
        preprocessor = Preprocessor(seed=seed)
        self.transform = preprocessor.test_transform()

        negative_path = os.path.join(self.root, "Negative")
        positive_path = os.path.join(self.root, "Positive")

        self.negative_imgs = [os.path.join(negative_path, x) for x in os.listdir(negative_path) if not x.startswith(".")]
        self.positive_imgs = [os.path.join(positive_path, x) for x in os.listdir(positive_path) if not x.startswith(".")]

        self.data = [(img, 0) for img in self.negative_imgs] + [(img, 1) for img in self.positive_imgs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        transform = self.transform 
        img = transform(img)

        return img, label
