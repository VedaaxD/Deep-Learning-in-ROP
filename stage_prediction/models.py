import torch
import torch.nn as nn
import timm

class GCViT_Pretrained(nn.Module):
    """
    GCViT Tiny with ImageNet pretrained weights modified for 4-class ROP stage classification
    """
    def __init__(self, num_classes=4, model_name="gcvit_tiny"):
        super(GCViT_Pretrained, self).__init__()

        #timm handles classifier replacement correctly
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)