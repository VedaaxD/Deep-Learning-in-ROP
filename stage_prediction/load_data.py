from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from data_preprocess import Preprocessor

class StageDataset(Dataset):
    def __init__(self, root_dir, augment=False, augment_repeats=1, seed=None):
        self.root = root_dir
        self.augment = augment
        self.augment_repeats = augment_repeats

        preprocessor = Preprocessor(seed=seed)
        self.transform1 = preprocessor.augment_transform1()
        self.transform2 = preprocessor.augment_transform2()
        self.transform3 = preprocessor.transform()

        # Get class folders (Normal, Stage 1, Stage 2, Stage 3)
        self.classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        # Map class name → index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all image paths
        self.data = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            for img_name in os.listdir(cls_path):
                if not img_name.startswith("."):
                    img_path = os.path.join(cls_path, img_name)
                    self.data.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.data) * self.augment_repeats

    def __getitem__(self, idx):

        img_path, label = self.data[idx % len(self.data)]
        img = Image.open(img_path).convert("RGB")

        if self.augment:
            transform = self.transform1 if torch.rand(1).item() > 0.5 else self.transform2
        else:
            transform = self.transform3

        img = transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class TestDataset(Dataset):
    #Test dataset -> no augmentation
    def __init__(self, root_dir, seed=None):
        self.root = root_dir
        preprocessor = Preprocessor(seed=seed)
        self.transform = preprocessor.transform()

        self.classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.data = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            for img_name in os.listdir(cls_path):
                if not img_name.startswith("."):
                    img_path = os.path.join(cls_path, img_name)
                    self.data.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
