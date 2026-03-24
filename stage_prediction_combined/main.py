#with 5-fold cross validation
import os

import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import GCViT_Pretrained
from train_and_eval import TrainEval
from visualization import GradCAMVisualizer
from load_data import TestDataset


def compute_class_weights(dataset, device):
    labels = [label for _, label in dataset.data]
    labels = np.array(labels)

    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)

    print("Class counts:", class_counts)
    print("Class weights:", weights)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_experiment(exp_no, train_dir, test_dir, k_folds=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Running Experiment {exp_no} ({k_folds}-Fold CV) ==========")

    fold_results = []

    for fold in range(k_folds):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        #Fresh model for each fold
        model = GCViT_Pretrained(num_classes=3).to(device)
        model_path = f"./Trained_Models/Exp{exp_no}_Fold{fold + 1}_GCViT.pth"

        trainer = TrainEval(
            model=model,
            train_dir=train_dir,
            test_dir=test_dir,
            model_path=model_path,
            pth_filename=f"Exp{exp_no}_Fold{fold + 1}_GCViT.pth",
            n_epochs=35,
            batch_size=32
        )

        #Signal trainer to use k-fold loader
        trainer._kfold_params = (fold, k_folds)

        #get train loader to compute class weights for this fold
        train_loader, _, _ = trainer.create_dataloader_kfold(
            fold_idx=fold, k=k_folds, augment=True
        )

        #compute class weights from this fold's training indices
        fold_labels = np.array([
            train_loader.dataset.dataset.data[i][1]
            for i in train_loader.dataset.indices
        ])
        counts = np.bincount(fold_labels)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        print("Class counts:", counts)
        print("Class weights:", weights)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

        trainer.train_model(optimizer, criterion, scheduler, augment=True)

        acc, macro_f1, weighted_f1 = trainer.evaluate_model(exp_no, fold=fold + 1)

        print("\nRunning GradCAM++...")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        target_layer = model.model.stages[-1].blocks[-1].norm1
        cam_visualizer = GradCAMVisualizer(model, target_layer)

        test_dataset = TestDataset(test_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        avg_drops = []
        mean_drop = 0.0
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)

            with torch.no_grad():
                output = model(image)
                pred_class = torch.argmax(output, dim=1).item()

            save_path = f"./GradCAM/Exp{exp_no}_Fold{fold + 1}/img_{i}.png"

            drop = cam_visualizer.visualize(
                input_tensor=image,
                true_label=label.item(),
                save_path=save_path
            )

            avg_drops.append(drop)

            if i >= 29:
                break

        mean_drop = np.mean(avg_drops)
        std_drop = np.std(avg_drops)
        print(f"Fold {fold + 1} | Mean Avg Drop: {mean_drop:.4f} ± {std_drop:.4f}")

        fold_results.append((acc, macro_f1, weighted_f1, mean_drop))
    #aggregate across folds
    accs      = [r[0] for r in fold_results]
    macro_f1s = [r[1] for r in fold_results]
    w_f1s     = [r[2] for r in fold_results]
    drops     = [r[3] for r in fold_results]

    mean_acc = np.mean(accs);   std_acc  = np.std(accs)
    mean_mf1 = np.mean(macro_f1s); std_mf1 = np.std(macro_f1s)
    mean_wf1 = np.mean(w_f1s);  std_wf1  = np.std(w_f1s)
    mean_drop=np.mean(drops); std_drop = np.std(drops)

    print(f"\n[Exp {exp_no}] {k_folds}-Fold CV Summary:")
    print(f"  Acc:         {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Macro-F1:    {mean_mf1:.4f} ± {std_mf1:.4f}")
    print(f"  Weighted-F1: {mean_wf1:.4f} ± {std_wf1:.4f}")
    print(f" Avg Drop: {mean_drop:.4f} ± {std_drop:.4f}")
    return mean_acc, mean_mf1, mean_wf1, mean_drop

def main():

    # Change paths accordingly
    SZH_PATH = "SZH"
    ROPVL_PATH = "ROP-VL"
    UHO_PATH = "UHO"

    results = []

    # Example: Train SZH ->Test ROP-VL
    results.append(run_experiment(1, SZH_PATH, ROPVL_PATH))

    # Train SZH -> Test UHO
    results.append(run_experiment(2, SZH_PATH, UHO_PATH))

    # Train ROP-VL -> test SZH
    results.append(run_experiment(3, ROPVL_PATH, SZH_PATH))

    # Train ROP-VL ->Test UHO
    results.append(run_experiment(4, ROPVL_PATH, UHO_PATH))

    # Train UHO ->Test SZH
    results.append(run_experiment(5, UHO_PATH, SZH_PATH))

    # Train UHO -> Test ROP-VL
    results.append(run_experiment(6, UHO_PATH, ROPVL_PATH))

    print("\n========= FINAL SUMMARY =========")
    for i, res in enumerate(results, 1):
        acc, macro_f1, weighted_f1, avg_drop = res
        print(f"Exp {i} -> Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | "
              f"Weighted-F1: {weighted_f1:.4f} | AvgDrop: {avg_drop:.4f}")


if __name__ == "__main__":
    main()
