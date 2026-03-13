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


def run_experiment(exp_no, train_dir, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Running Experiment {exp_no} ==========")

    model = GCViT_Pretrained(num_classes=4).to(device)
    model_path = f"./Trained_Models/Exp{exp_no}_GCViT.pth"

    trainer = TrainEval(
        model=model,
        train_dir=train_dir,
        test_dir=test_dir,
        model_path=model_path,
        pth_filename=f"Exp{exp_no}_GCViT.pth",
        n_epochs=35,
        batch_size=32
    )

    train_loader, _, _ = trainer.create_dataloader(augment=True)

    #weighted CE
    class_weights = compute_class_weights(
        train_loader.dataset.dataset,
        device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    #train
    trainer.train_model(optimizer, criterion, scheduler, augment=True)

    # evaluate
    acc, macro_f1, weighted_f1 = trainer.evaluate_model(exp_no)

    # GradCAM++
    print("\nRunning GradCAM++...")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #Correct target layer for GCViT
    target_layer = model.model.stages[-1].blocks[-1].norm1
    cam_visualizer = GradCAMVisualizer(model, target_layer)

    test_dataset = TestDataset(test_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    avg_drops = []
    for i, (image, label) in enumerate(test_loader):

        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            pred_class = torch.argmax(output, dim=1).item()

        save_path = f"./GradCAM/Exp{exp_no}/img_{i}.png"
        drop = cam_visualizer.visualize(
            input_tensor=image,
            true_label=label.item(),
            save_path=save_path
        )

        avg_drops.append(drop)
        if i >= 29:  #visualize only first 30 images
            break
    mean_drop = np.mean(avg_drops)
    std_drop = np.std(avg_drops)
    print(f"\nMean Avg Drop (Exp {exp_no}): {mean_drop:.4f} ± {std_drop:.4f}")
    return acc, macro_f1, weighted_f1, mean_drop


def main():
    SZH_PATH = "SZH"
    ROPVL_PATH = "ROP-VL"
    UHO_PATH = "UHO"
    results = []

    #Train SZH  Test ROP-VL
    results.append(run_experiment(1, SZH_PATH, ROPVL_PATH))

    # Train SZH Test UHO
    results.append(run_experiment(2, SZH_PATH, UHO_PATH))

    # Train ROP-VL Test SZH
    results.append(run_experiment(3, ROPVL_PATH, SZH_PATH))

    # Train ROP-VL Test UHO
    results.append(run_experiment(4, ROPVL_PATH, UHO_PATH))

    # Train UHO Test SZH
    results.append(run_experiment(5, UHO_PATH, SZH_PATH))

    # Train UHO Test ROP-VL
    results.append(run_experiment(6, UHO_PATH, ROPVL_PATH))

    print("\n========= FINAL SUMMARY =========")
    for i, res in enumerate(results, 1):
        acc, macro_f1, weighted_f1, avg_drop = res
        print(f"Exp {i} -> Acc: {acc:.2f} | Macro-F1: {macro_f1:.4f} | "
              f"Weighted-F1: {weighted_f1:.4f} | AvgDrop: {avg_drop:.4f}")

if __name__ == "__main__":
    main()
