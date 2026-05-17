import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from captum.attr import GradientShap, Occlusion

CLASS_NAMES = ["Normal", "Mild", "Severe"]

def reshape_transform(tensor):

    if len(tensor.shape) == 4:
        return tensor

    # For transformer(ViT-like)
    elif len(tensor.shape) == 3:
        B, N, C = tensor.shape
        H = W = int(np.sqrt(N))
        assert H * W == N, "Transformer tokens not square"
        return tensor.permute(0, 2, 1).reshape(B, C, H, W)


def signed_attr_to_rgb(attr_2d: np.ndarray) -> np.ndarray:
    """
    Map a signed attribution map to an RGB image.
    Uses the RdYlGn colormap:
      red  -> negative attribution (against prediction)
      yellow -> zero / neutral
      green -> positive attribution (toward prediction)

    Returns float32 (H, W, 3) in [0, 1].
    """
    a = attr_2d.copy().astype(np.float32)
    mx = np.abs(a).max()
    if mx > 1e-8:
        a = a / mx   #normalise to [-1, 1]
    rgb = plt.cm.RdYlGn((a + 1.0) / 2.0)[..., :3]   # (H, W, 3)
    return rgb.astype(np.float32)

class MultiXAIVisualizer:
    """
    Generates a 4-panel XAI figure for a single image:
      Panel 0 – original fundus image
      Panel 1 – GradCAM++
      Panel 2 – GradSHAP  (red=against, green=toward prediction)
      Panel 3 – Occlusion (red=against, green=toward prediction)

    Also reports GradCAM++-based Average Drop for quantitative evaluation.
    """

    def __init__(self, model, target_layer):
        self.model  = model.eval().to(next(model.parameters()).device)
        self.device = next(model.parameters()).device
        self.target_layer = target_layer

        self.cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=reshape_transform,
        )

    def _gradcam(self, input_tensor, class_idx) -> np.ndarray:
        """Returns (H, W) grayscale GradCAM++ map."""
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True,
        )
        return grayscale_cam[0]

    def _gradshap(self, input_tensor, class_idx) -> np.ndarray:
        """Returns (H, W) signed GradSHAP attribution, averaged over channels."""
        gs       = GradientShap(self.model)
        baseline = torch.zeros_like(input_tensor) *  0.001
        attr = gs.attribute(
            input_tensor,
            baselines=baseline,
            target=class_idx,
            n_samples=50,
            stdevs=0.01,
        )                                           # (1, C, H, W)
        attr_np = attr.squeeze().detach().cpu().numpy()
        if attr_np.ndim == 3:                       # (C, H, W) -> (H, W)
            attr_np = attr_np.mean(axis=0)
        return attr_np

    def _occlusion(self, input_tensor, class_idx) -> np.ndarray:
        """Returns (H, W) signed Occlusion attribution, averaged over channels."""
        occ  = Occlusion(self.model)
        attr = occ.attribute(
            input_tensor,
            strides=(1, 8, 8),
            target=class_idx,
            sliding_window_shapes=(3, 11,11),
            baselines=0,
        )                                           # (1, C, H, W)
        attr_np = attr.squeeze().detach().cpu().numpy()
        if attr_np.ndim == 3:
            attr_np = attr_np.mean(axis=0)
        return attr_np


    def compute_average_drop(self, input_tensor, class_idx) -> float:
        """GradCAM++-based Average Drop (lower = better explanations)."""
        with torch.no_grad():
            orig_prob = torch.softmax(
                self.model(input_tensor), dim=1
            )[0, class_idx]

        cam_np   = self._gradcam(input_tensor, class_idx)
        cam_mask = (
            torch.from_numpy(cam_np)
            .to(self.device)
            .unsqueeze(0).unsqueeze(0)
        )
        cam_mask = torch.nn.functional.interpolate(
            cam_mask,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        masked_input = input_tensor.clone() * cam_mask.unsqueeze(0)
        with torch.no_grad():
            masked_prob = torch.softmax(
                self.model(masked_input), dim=1
            )[0, class_idx]

        return float(max(0.0, (orig_prob - masked_prob) / (orig_prob + 1e-8)))


    def visualize(self, input_tensor, true_label, save_path) -> float:
        """
        Generate the 4-panel XAI figure, save to save_path,
        and return the GradCAM++ Average Drop for this sample.
        """

        with torch.no_grad():
            probs = torch.softmax(self.model(input_tensor), dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = img_np.astype(np.float32)

        cam_map      = self._gradcam(input_tensor, pred_class)
        gradcam_vis  = show_cam_on_image(img_np, cam_map, use_rgb=True)

        gs_attr      = self._gradshap(input_tensor, pred_class)
        gs_rgb       = signed_attr_to_rgb(gs_attr)

        occ_attr     = self._occlusion(input_tensor, pred_class)
        occ_rgb      = signed_attr_to_rgb(occ_attr)

        avg_drop     = self.compute_average_drop(input_tensor, pred_class)

        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_class]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(
            f"XAI — True: {true_name} | Pred: {pred_name} | Conf: {confidence:.2f}",
            fontsize=13, fontweight="bold",
        )

        axes[0].imshow(img_np)
        axes[0].set_title(f"True: {true_name} | Pred: {pred_name}\nConf: {confidence:.2f}")
        axes[0].axis("off")

        axes[1].imshow(gradcam_vis)
        axes[1].set_title(f"GradCAM++\nAvg Drop: {avg_drop:.4f}")
        axes[1].axis("off")

        axes[2].imshow(gs_rgb)
        axes[2].set_title("GradSHAP\n(green=toward pred, red=against)")
        axes[2].axis("off")

        axes[3].imshow(occ_rgb)
        axes[3].set_title("Occlusion\n(green=toward pred, red=against)")
        axes[3].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return avg_drop
GradCAMVisualizer = MultiXAIVisualizer
