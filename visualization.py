import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import math
from sklearn.metrics import roc_curve, roc_auc_score


class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = self._find_target_layer()

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _find_target_layer(self):
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output.squeeze()
        target.backward()
        gradients = self.gradients
        activations = self.activations
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom

        weights = (alphas * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode="bilinear")
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

class AttributionVisualizer:
    def __init__(self, net, input_img):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.input = input_img.unsqueeze(0).to(self.device)
        self.net.eval()
        self.cam_engine = GradCAMPlusPlus(self.net)

    def _original_image(self):
        img_tensor = self.input.squeeze().cpu().detach()
        mean = torch.tensor([0.456, 0.456, 0.456]).view(3,1,1)
        std = torch.tensor([0.224, 0.224, 0.224]).view(3,1,1)
        img_denorm = torch.clamp(img_tensor * std + mean, 0, 1)
        return np.transpose(img_denorm.numpy(), (1,2,0))

    def viz_attr(self, index, save_dir):
        cam = self.cam_engine.generate(self.input)
        original_img = self._original_image()
        heatmap = plt.get_cmap('jet')(cam)[..., :3]
        overlay = 0.5 * original_img + 0.5 * heatmap

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        axs[0].imshow(original_img); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(cam, cmap="jet"); axs[1].set_title("Grad-CAM++"); axs[1].axis("off")
        axs[2].imshow(overlay); axs[2].set_title("Overlay"); axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_img_{index+1}.png")
        plt.close()

def create_samples(loader, net, device):
    TP, TN, FP, FN = [], [], [], []
    all_labels, pred_proba = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = torch.sigmoid(net(images))
            pred_proba.extend(outputs.cpu().numpy().ravel())
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().ravel())

            for i, pred in enumerate(predicted):
                if pred == 1 and labels[i] == 1: TP.append(images[i])
                elif pred == 0 and labels[i] == 0: TN.append(images[i])
                elif pred == 1 and labels[i] == 0: FP.append(images[i])
                elif pred == 0 and labels[i] == 1: FN.append(images[i])

    return TP[:10], TN[:10], FP[:10], FN[:10], all_labels, pred_proba


def plot_metrics(y_true, pred_proba_dict, path):
    plt.figure(figsize=(8,8), dpi=300)
    for model_name, model_pred_proba in pred_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, model_pred_proba)
        auc = roc_auc_score(y_true, model_pred_proba)
        plt.plot(fpr, tpr, label=f"{model_name}(AUC={auc:.4f})")

    plt.plot([0,1],[0,1],'--',color='r')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.savefig(f"./{path}/ROC_PR_Plots.png")
    plt.close()


def plot_line_chart(model_list, accuracy_list, f1_score_list, exp_nos=5):
    plt.figure(figsize=(10,6))
    experiments = np.arange(1, exp_nos+1)
    markers = ['o','s','D','^','x']
    colors = ['b','g','r','c','m']

    for i, model in enumerate(model_list):
        plt.plot(experiments, accuracy_list[i], marker=markers[i], color=colors[i], linestyle='-', label=f'{model} Acc')
        plt.plot(experiments, f1_score_list[i], marker=markers[i], color=colors[i], linestyle='--', label=f'{model} F1')

    plt.xlabel("Experiment"); plt.ylabel("Score"); plt.ylim(0,1)
    plt.legend(); plt.grid(True)
    plt.savefig("./Performance_metrics/GroupedPlot.png", dpi=300)
    plt.close()


def grouped_barplot(model_list, sensitivity_list, specificity_list, exp_nos=5):
    x = np.arange(len(model_list))
    width = 0.35
    rows = math.ceil(exp_nos/2)

    fig, axes = plt.subplots(rows, 2, figsize=(12,4*rows), sharey=True)
    axes = axes.flatten()

    for j in range(exp_nos, len(axes)):
        axes[j].axis("off")

    experiments = [f"Experiment {i+1}" for i in range(exp_nos)]

    for i, ax in enumerate(axes[:exp_nos]):
        ax.bar(x - width/2, sensitivity_list[i], width, label='Sensitivity')
        ax.bar(x + width/2, specificity_list[i], width, label='Specificity')
        ax.set_title(experiments[i])
        ax.set_xticks(x); ax.set_xticklabels(model_list)
        ax.set_ylim(0,1); ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.legend(['Sensitivity','Specificity'], loc='upper center', ncol=2)
    plt.tight_layout(rect=[0.05,0.05,1,0.93])
    plt.savefig("./Performance_metrics/GroupedBarPlot.png", dpi=300)
    plt.close()
