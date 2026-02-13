from train_n_eval import TrainEval
from models import CustomCNN, ResNet18_, ResNet18_pretrained, EfficientNet_pretrained, EfficientNet_
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
from visualization import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def counter():
    count = 0
    while True:
        count += 1
        yield count


#chnged-visualization loop always runs
def _run_visualizer(exp_no, model, model_path, image_lst, device, type):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.to(device)
    model.eval()

    save_dir = f"./Visualization_Results/Exp{exp_no}/{model_path.split('/')[-1].split('.')[0]}/{type}"
    os.makedirs(save_dir, exist_ok=True)

    for index, image in enumerate(image_lst):
        image = image.to(device)
        av = AttributionVisualizer(model, image)
        av.viz_attr(index, save_dir)


def run(model_dict, counter, train_dir, test_dir, augment=False):
    exp_no = next(counter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    criterion = nn.BCEWithLogitsLoss()

    pred_proba_dict = {}
    accuracy_list = []
    f1_score_list = []
    sensitivity_list = []
    specificity_list = []

    for model, pth_filename in model_dict.items():
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

        os.makedirs("./Trained_Models/", exist_ok=True)
        model_path = f"./Trained_Models/{pth_filename}"

        train_eval = TrainEval(model, train_dir, test_dir, model_path, pth_filename, num_epochs)
        train_eval.train_model(optimizer, criterion, scheduler, augment)
        test_accuracy, test_f1, test_sensitivity, test_specificity = train_eval.evaluate_model(exp_no)

        accuracy_list.append(test_accuracy)
        f1_score_list.append(test_f1)
        sensitivity_list.append(test_sensitivity)
        specificity_list.append(test_specificity)

        metrics_dir = f"./Performance_metrics/Exp{exp_no}"
        os.makedirs(metrics_dir, exist_ok=True)

        #use evaluated model weights
        tp_lst, tn_lst, fp_lst, fn_lst, y_true, pred_proba = create_samples(train_eval.test_loader, train_eval.model, device)
        pred_proba_dict[pth_filename.split('.')[0]] = pred_proba

        _run_visualizer(exp_no, model, model_path, tp_lst, device, "TP")
        _run_visualizer(exp_no, model, model_path, tn_lst, device, "TN")
        _run_visualizer(exp_no, model, model_path, fp_lst, device, "FP")
        _run_visualizer(exp_no, model, model_path, fn_lst, device, "FN")

    plot_metrics(y_true, pred_proba_dict, metrics_dir)
    return accuracy_list, f1_score_list, sensitivity_list, specificity_list


def main():
    cntr = counter()

    f1_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []

    dir_viio = "/mnt/data/vedavalli/rop_reproduced/VIIO"
    dir_uho = "/mnt/data/vedavalli/rop_reproduced/UHO"
    dir_combined = "/mnt/data/vedavalli/rop_reproduced/VIIO_UHO_combined"

    #EXP 1 — Train VIIO - Test VIIO
    models = {
        CustomCNN(): "V_customCNN.pth",
        ResNet18_(): "V_ResNet18_scratch.pth",
        ResNet18_pretrained(): "V_ResNet18_pretrained.pth",
        EfficientNet_(): "V_EfficientNetB0_scratch.pth",
        EfficientNet_pretrained(): "V_EfficientNetB0_pretrained.pth"
    }
    a1, f1, s1, sp1 = run(models, cntr, dir_viio, dir_viio, augment=True)
    accuracy_list.append(a1); f1_list.append(f1); sensitivity_list.append(s1); specificity_list.append(sp1)

    #EXP 2 — Train UHO- Test UHO
    models = {
        CustomCNN(): "K_customCNN.pth",
        ResNet18_(): "K_ResNet18_scratch.pth",
        ResNet18_pretrained(): "K_ResNet18_pretrained.pth",
        EfficientNet_(): "K_EfficientNetB0_scratch.pth",
        EfficientNet_pretrained(): "K_EfficientNetB0_pretrained.pth"
    }
    a2, f2, s2, sp2 = run(models, cntr, dir_uho, dir_uho, augment=True)
    accuracy_list.append(a2); f1_list.append(f2); sensitivity_list.append(s2); specificity_list.append(sp2)

    #EXP 3 — Train VIIO - Test UHO
    models = {
        CustomCNN(): "V_Augmented_customCNN.pth",
        ResNet18_(): "V_Augmented_ResNet18_scratch.pth",
        ResNet18_pretrained(): "V_Augmented_ResNet18_pretrained.pth",
        EfficientNet_(): "V_Augmented_EfficientNetB0_scratch.pth",
        EfficientNet_pretrained(): "V_Augmented_EfficientNetB0_pretrained.pth"
    }
    a3, f3, s3, sp3 = run(models, cntr, dir_viio, dir_uho, augment=True)
    accuracy_list.append(a3); f1_list.append(f3); sensitivity_list.append(s3); specificity_list.append(sp3)

    #EXP 4 — Train UHO -Test VIIO
    models = {
        CustomCNN(): "K_Augmented_customCNN.pth",
        ResNet18_(): "K_Augmented_ResNet18_scratch.pth",
        ResNet18_pretrained(): "K_Augmented_ResNet18_pretrained.pth",
        EfficientNet_(): "K_Augmented_EfficientNetB0_scratch.pth",
        EfficientNet_pretrained(): "K_Augmented_EfficientNetB0_pretrained.pth"
    }
    a4, f4, s4, sp4 = run(models, cntr, dir_uho, dir_viio, augment=True)
    accuracy_list.append(a4); f1_list.append(f4); sensitivity_list.append(s4); specificity_list.append(sp4)

    #EXP 5 — Train VIIO+UHO -Test VIIO+UHO
    models = {
        CustomCNN(): "C_Augmented_customCNN.pth",
        ResNet18_(): "C_Augmented_ResNet18_scratch.pth",
        ResNet18_pretrained(): "C_Augmented_ResNet18_pretrained.pth",
        EfficientNet_(): "C_Augmented_EfficientNetB0_scratch.pth",
        EfficientNet_pretrained(): "C_Augmented_EfficientNetB0_pretrained.pth"
    }
    a5, f5, s5, sp5 = run(models, cntr, dir_combined, dir_combined, augment=True)
    accuracy_list.append(a5); f1_list.append(f5); sensitivity_list.append(s5); specificity_list.append(sp5)

    accuracy_array = np.array(accuracy_list).T / 100.0
    f1_array = np.array(f1_list).T
    model_list = ["CustomCNN", "ResNet18_scratch", "ResNet18_pretrained", "EfficientNet_scratch", "EfficientNet_pretrained"]

    plot_line_chart(model_list, accuracy_array, f1_array, exp_nos=5)
    grouped_barplot(model_list, sensitivity_list, specificity_list, exp_nos=5)


if __name__ == "__main__":
    main()
