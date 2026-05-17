import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

class GCViT_Pretrained(nn.Module):
    """
    GCViT-Tiny – ImageNet pretrained if already cached locally.
    If not cached, falls back to random init.
    """
    def __init__(self, num_classes=3, model_name="gcvit_tiny"):
        super().__init__()

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes
            )
            print("Loaded GCViT pretrained weights.")
        except:
            print("GCViT pretrained weights unavailable. Using random init.")
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes
            )

    def forward(self, x):
        return self.model(x)

class TimmCNN(nn.Module):
    """
    Generic CNN wrapper:
    backbone -> global avg pool -> Dropout -> Linear(3)
    Loads local pretrained weights manually.
    """
    def __init__(self, model_name: str, num_classes: int = 3, drop_rate: float = 0.3):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        ckpt = None

        # DenseNet121
        if model_name == "densenet121":
            ckpt = load_file("/home/veda/weights/densenet121.safetensors")

        # EfficientNet-B4
        elif model_name == "efficientnet_b4":
            ckpt = torch.load(
                "/home/veda/weights/effnet_b4_pytorch_model.bin",
                map_location="cpu"
            )

        # EfficientNetV2-S
        elif model_name == "efficientnetv2_s":
            ckpt = torch.load(
                "/home/veda/weights/effnet_v2s_pytorch_model.bin",
                map_location="cpu"
            )

        if ckpt is not None:
            msg = self.backbone.load_state_dict(ckpt, strict=False)
            print(f"Loaded weights for {model_name}")
            print(msg)

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

class EfficientNetV2_S(TimmCNN):
    def __init__(self, num_classes=3):
        super().__init__(
            "efficientnetv2_s",
            num_classes=num_classes,
            drop_rate=0.30
        )

class EfficientNet_B4(TimmCNN):
    def __init__(self, num_classes=3):
        super().__init__(
            "efficientnet_b4",
            num_classes=num_classes,
            drop_rate=0.35
        )


class DenseNet121(TimmCNN):
    def __init__(self, num_classes=3):
        super().__init__(
            "densenet121",
            num_classes=num_classes,
            drop_rate=0.25
        )

def get_gradcam_layer(model: nn.Module):

    if isinstance(model, GCViT_Pretrained):
        return model.model.stages[-1].blocks[-1].norm1

    elif isinstance(model, EfficientNetV2_S):
        return model.backbone.blocks[-1][-1].conv_pw

    elif isinstance(model, EfficientNet_B4):
        return model.backbone.conv_head

    elif isinstance(model, DenseNet121):
        return model.backbone.features.denseblock4.denselayer16.conv2

    else:
        raise ValueError(
            f"No GradCAM mapping for {type(model).__name__}"
        )

model_registry = {
    "gcvit": GCViT_Pretrained,
    "efficientnetv2_s": EfficientNetV2_S,
    "efficientnet_b4": EfficientNet_B4,
    "densenet121": DenseNet121,
}
