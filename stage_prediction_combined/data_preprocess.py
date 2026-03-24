from torchvision.transforms import v2
import random
import cv2
import torch
import numpy as np


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

    def _green_clahe(self, img):

        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        b,g,r= cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g= clahe.apply(g)

        img = cv2.merge([b, g, r])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1) / 255.0
        return img

    def augment_transform1(self):
        return v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.9, 1.0)),
            v2.Lambda(self._green_clahe),
            v2.RandomRotation(15),
            v2.GaussianNoise(mean=0.0, sigma=0.03, clip=True),
            v2.Normalize(
                mean=(0.456, 0.456, 0.456),
                std=(0.224, 0.224, 0.224)
            ),
        ])

    def augment_transform2(self):
        return v2.Compose([
            v2.Resize((224, 224)),
            v2.Lambda(self._green_clahe),
            v2.RandomRotation(10),
            v2.Normalize(
                mean=(0.456, 0.456, 0.456),
                std=(0.224, 0.224, 0.224)
            ),
        ])

    def transform(self):
        return v2.Compose([
            v2.Resize((224, 224)),
            v2.Lambda(self._green_clahe),
            v2.Normalize(
                mean=(0.456, 0.456, 0.456),
                std=(0.224, 0.224, 0.224)
            ),
        ])
