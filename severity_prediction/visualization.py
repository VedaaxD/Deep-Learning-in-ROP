import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

CLASS_NAMES = ["Normal", "Stage1", "Mild", "Severe"]

# def reshape_transform(tensor):
# tensor comes as (B, H, W, C)
# if len(tensor.shape) == 4:
# tensor = tensor.permute(0, 3, 1, 2)
# return tensor
def reshape_transform(tensor):
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    elif len(tensor.shape) == 3:
        B, N, C = tensor.shape
        H = W = int(np.sqrt(N))
        return tensor.permute(0, 2, 1).reshape(B, C, H, W)
    return tensor


class GradCAMVisualizer:

    def __init__(self, model, target_layer):
        self.model = model.eval().to(next(model.parameters()).device)
        self.target_layer = target_layer

        self.cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=reshape_transform
        )

    def generate_cam(self, input_tensor, class_idx):
        targets = [ClassifierOutputTarget(class_idx)]

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True
        )

        return grayscale_cam[0]

    def compute_average_drop(self, input_tensor, class_idx):
        with torch.no_grad():
            original_output = self.model(input_tensor)
            original_prob = torch.softmax(original_output, dim=1)[0, class_idx]

        cam = self.generate_cam(input_tensor, class_idx)
        cam_mask = torch.from_numpy(cam).to(input_tensor.device)
        cam_mask = cam_mask.unsqueeze(0).unsqueeze(0)

        cam_mask = torch.nn.functional.interpolate(
            cam_mask,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze()

        masked_input = input_tensor.clone()
        masked_input *= cam_mask.unsqueeze(0)

        with torch.no_grad():
            masked_output = self.model(masked_input)
            masked_prob = torch.softmax(masked_output, dim=1)[0, class_idx]

        drop = max(0, (original_prob - masked_prob) / original_prob)
        return float(drop)

    def visualize(self, input_tensor, true_label, save_path):
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

        cam = self.generate_cam(input_tensor, pred_class)

        input_image = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
        visualization = show_cam_on_image(
            input_image.astype(np.float32),
            cam,
            use_rgb=True
        )

        avg_drop = self.compute_average_drop(input_tensor, pred_class)

        plt.figure(figsize=(6, 6))
        plt.imshow(visualization)
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_class]

        plt.title(f"True: {true_name} | Pred: {pred_name} | Conf: {confidence:.2f}\n"
                  f"GradCAM++ | Avg Drop: {avg_drop:.4f}")
        plt.axis("off")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return avg_drop
