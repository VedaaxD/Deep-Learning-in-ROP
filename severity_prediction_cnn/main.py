#
# TWO EXPERIMENT TYPES
# TYPE 1 — INTRA-DATASET  (proper 10-fold CV)
#   - Dataset is split into 10 stratified folds
#   - Fold i  ->  test  (held-out, never seen during training)
#   - Folds 0..i-1, i+1..9  -> train+val  (80/20 internal split for early stop)
#   - No external test data used at any point
#   - Run for each dataset: SZH, ROP-VL, UHO
#   - Output: mean ± std over 10 folds per dataset
#   - NOTE: XAI visualisation is SKIPPED for intra-CV (redundant across 10 folds)
#
# TYPE 2 — CROSS-DATASET  (single train to multiple test)
#   - Train on full dataset A (80/20 internal train/val split)
#   - Test on complete external datasets B and C
#   - One trained model per source dataset (no redundant re-training)
#   - Run for all 3 sources: SZH, ROP-VL, UHO
#   - Output: 6 cross-dataset results (A->B, A->C for each A)
#   - XAI: GradCAM++ + GradSHAP + Occlusion saved per image


import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import model_registry, get_gradcam_layer
from train_and_eval import TrainEval
from visualization import MultiXAIVisualizer
from load_data import TestDataset, StageDataset

output_root = "Results"


# ──────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────

def compute_class_weights(labels_array, device):
    counts  = np.bincount(labels_array,minlength=3)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    print(f"  Class counts: {counts}  |  weights: {np.round(weights, 4)}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

def run_xai(model, model_path, test_dataset_path, save_dir, device, n_images=30):
    """
    Runs XAI ensuring ONE image per class (Normal, Mild, Severe)
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layer = get_gradcam_layer(model)
    visualizer   = MultiXAIVisualizer(model, target_layer)

    test_loader = torch.utils.data.DataLoader(
        TestDataset(test_dataset_path), batch_size=1, shuffle=False
    )

    #fix: class-balanced selection
    selected = {0: None, 1: None, 2: None}
    for image, label in test_loader:
        cls = label.item()

        if selected[cls] is None:
            selected[cls] = (image, cls)

        if all(v is not None for v in selected.values()):
            break
    avg_drops = []

    for cls, data in selected.items():
        if data is None:
            print(f"No sample found for class {cls}")
            continue

        image, label = data
        class_name = ["Normal", "Mild", "Severe"][cls]

        save_path = os.path.join(save_dir, f"{class_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        drop = visualizer.visualize(
            input_tensor=image.to(device),
            true_label=label,
            save_path=save_path,
        )
        avg_drops.append(drop)
    return avg_drops


def build_trainer(model, train_dir, test_dir, model_path, pth_filename):
    return TrainEval(
        model        = model,
        train_dir    = train_dir,
        test_dir     = test_dir,
        model_path   = model_path,
        pth_filename = pth_filename,
        n_epochs     = 35,
        batch_size   = 32,
        output_root  = output_root,
    )

#intra dataset - 10 fold cv
def run_intradataset_cv(dataset_name, dataset_dir, model_key, k_folds=10):
    """
    Proper k-fold CV within a single dataset.
    Fold i is the test set; the remaining k-1 folds are used for training
    (with an 80/20 internal split for validation / early stopping).
    No external test data and no XAI visualisation here.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = model_registry[model_key]

    print(f"\n{'═'*65}")
    print(f"  TYPE 1 — INTRA-DATASET {k_folds}-FOLD CV")
    print(f"  Model: {model_key}  |  Dataset: {dataset_name}")
    print(f"{'═'*65}")

    fold_results = []   # (acc, mf1, wf1, test_loss) per fold

    for fold in range(k_folds):
        print(f"\n{'─'*55}")
        print(f"  Fold {fold+1}/{k_folds} — test fold = {fold}")
        print(f"{'─'*55}")

        model      = ModelClass(num_classes=3).to(device)
        model_path = os.path.join(
            output_root, "Intra_Models", model_key,
            f"{dataset_name}_Fold{fold+1}.pth"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        trainer = build_trainer(
            model, dataset_dir, dataset_dir, model_path,
            f"intra_{model_key}_{dataset_name}_Fold{fold+1}.pth"
        )

        train_loader, val_loader, test_loader, train_idx, _, _ = \
            trainer.create_dataloader_intrafold(fold_idx=fold, k=k_folds, augment=True)

        full_dataset = StageDataset(dataset_dir, seed=0, augment=False)
        fold_labels  = np.array([full_dataset.data[i][1] for i in train_idx])
        class_weights = compute_class_weights(fold_labels, device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

        trainer.train_model(optimizer, criterion, scheduler, train_loader, val_loader)

        acc, macro_f1, weighted_f1, test_loss = trainer.evaluate_model(
            exp_no      = 0,
            fold        = fold + 1,
            test_loader = test_loader,
            test_name   = dataset_name,
        )

        fold_results.append((acc, macro_f1, weighted_f1, test_loss))

    accs, mf1s, wf1s, losses = zip(*fold_results)
    summary = {
        "acc":         (np.mean(accs),   np.std(accs)),
        "macro_f1":    (np.mean(mf1s),   np.std(mf1s)),
        "weighted_f1": (np.mean(wf1s),   np.std(wf1s)),
        "test_loss":   (np.mean(losses), np.std(losses)),
    }

    print(f"\n  ── {dataset_name} | {model_key} | {k_folds}-Fold Intra CV ──")
    for k, (m, s) in summary.items():
        print(f"    {k:<14}: {m:.4f} ± {s:.4f}")

    return summary

#cross-dataset generalization
def run_crossdataset(train_name, train_dir, test_targets, model_key):
    """
    Train once on train_dir (80/20 internal split for early stopping).
    Evaluate + run 3-method XAI on each path in test_targets.

    test_targets: list of {"name": str, "path": str, "exp_no": int}
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = model_registry[model_key]

    print(f"\n{'═'*65}")
    print(f"  TYPE 2 — CROSS-DATASET")
    print(f"  Model: {model_key}  |  Train: {train_name}")
    print(f"  Test targets: {[t['name'] for t in test_targets]}")
    print(f"{'═'*65}")

    model      = ModelClass(num_classes=3).to(device)
    model_path = os.path.join(
        output_root, "Cross_Models", model_key,
        f"Train{train_name}.pth"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    trainer = build_trainer(
        model, train_dir, test_targets[0]["path"], model_path,
        f"cross_{model_key}_Train{train_name}.pth"
    )

    train_loader, val_loader, _ = trainer.create_dataloader_cross(augment=True)

    full_dataset  = StageDataset(train_dir, seed=0, augment=False)
    all_labels    = np.array([lbl for _, lbl in full_dataset.data])
    class_weights = compute_class_weights(all_labels, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # Train once
    trainer.train_model(optimizer, criterion, scheduler, train_loader, val_loader)

    # Evaluate + XAI on every test target
    results = {}
    for target in test_targets:
        print(f"\n  ── Evaluating on: {target['name']} ──")

        acc, macro_f1, weighted_f1, test_loss = trainer.evaluate_model(
            exp_no    = target["exp_no"],
            fold      = None,
            test_dir  = target["path"],
            test_name = target["name"],
        )

        xai_dir = os.path.join(
            output_root, "XAI", "Cross", model_key,
            f"Train{train_name}_Test{target['name']}"
        )
        avg_drops = run_xai(
            model, model_path, target["path"], xai_dir, device, n_images=30
        )
        mean_drop = float(np.mean(avg_drops))
        print(f"  XAI Mean Avg Drop (GradCAM++): {mean_drop:.4f} ± {np.std(avg_drops):.4f}")

        results[target["name"]] = {
            "acc":         acc,
            "macro_f1":    macro_f1,
            "weighted_f1": weighted_f1,
            "test_loss":   test_loss,
            "avg_drop":    mean_drop,
            "exp_no":      target["exp_no"],
        }

    return results

def main():
    SZH_PATH   = "/home/veda/stage_prediction_combined/SZH"
    ROPVL_PATH = "/home/veda/stage_prediction_combined/ROP-VL"
    UHO_PATH   = "/home/veda/stage_prediction_combined/UHO"

    models_to_run = [ "efficientnet_b4"]

    # ── TYPE 1: Intra-dataset 10-fold CV ──────────────────────
    intra_datasets = [
        {"name": "SZH",    "path": SZH_PATH},
        {"name": "ROP-VL", "path": ROPVL_PATH},
        {"name": "UHO",    "path": UHO_PATH},
    ]

    # ── TYPE 2: Cross-dataset ─────────────────────────────────
    #cross_runs = [
     #   {
    #        "train_name":  "SZH",
     #       "train_dir":   SZH_PATH,
      #      "test_targets": [
       #         {"name": "ROP-VL", "path": ROPVL_PATH, "exp_no": 1},
        #        {"name": "UHO",    "path": UHO_PATH,   "exp_no": 2},
    #        ],
     #   },
     #   {
      #      "train_name":  "ROP-VL",
      #      "train_dir":   ROPVL_PATH,
       #     "test_targets": [
        #        {"name": "SZH", "path": SZH_PATH, "exp_no": 3},
         #       {"name": "UHO", "path": UHO_PATH, "exp_no": 4},
    #        ],
     #   },
      #  {
       #     "train_name":  "UHO",
        #    "train_dir":   UHO_PATH,
         #   "test_targets": [
          #      {"name": "SZH",    "path": SZH_PATH,   "exp_no": 5},
           #     {"name": "ROP-VL", "path": ROPVL_PATH, "exp_no": 6},
   #         ],
    #    },
    #]
    cross_runs = [
    {
        "train_name":  "ROP-VL",
        "train_dir":   ROPVL_PATH,
        "test_targets": [
            {"name": "SZH", "path": SZH_PATH, "exp_no": 3},
        ],
    }
]
    intra_results = {}   # [model_key][dataset_name] = summary dict
    cross_results = {}   # [model_key][train_name][test_name] = result dict

    for model_key in models_to_run:
        intra_results[model_key] = {}
        cross_results[model_key] = {}

        #for ds in intra_datasets:
         #   summary = run_intradataset_cv(
          #      dataset_name = ds["name"],
           #     dataset_dir  = ds["path"],
            #    model_key    = model_key,
             #   k_folds      = 10,
            #)
            #intra_results[model_key][ds["name"]] = summary

        for run in cross_runs:
            results = run_crossdataset(
                train_name   = run["train_name"],
                train_dir    = run["train_dir"],
                test_targets = run["test_targets"],
                model_key    = model_key,
            )
            cross_results[model_key][run["train_name"]] = results

    sep = "=" * 90

    #Table 1: Intra-dataset (no AvgDrop — XAI skipped)
    print(f"\n{sep}")
    print("TABLE 1 — INTRA-DATASET 10-FOLD CV  (mean ± std)")
    print(sep)
    print(f"{'Dataset':<10} {'Model':<18} {'Acc(%)':>10} {'Macro-F1':>11} "
          f"{'Wt-F1':>9} {'TestLoss':>10}")
    print("-" * 90)
    for ds in intra_datasets:
        for model_key in models_to_run:
            s = intra_results[model_key][ds["name"]]
            print(
                f"{ds['name']:<10} {model_key:<18} "
                f"{s['acc'][0]:>7.2f}±{s['acc'][1]:.2f}  "
                f"{s['macro_f1'][0]:>8.4f}±{s['macro_f1'][1]:.4f}  "
                f"{s['weighted_f1'][0]:>6.4f}±{s['weighted_f1'][1]:.4f}  "
                f"{s['test_loss'][0]:>7.4f}±{s['test_loss'][1]:.4f}"
            )
        print()

    # Table 2: Cross-dataset (includes AvgDrop from GradCAM++)
    cross_order = [
        ("SZH",    "ROP-VL", 1),
        ("SZH",    "UHO",    2),
        ("ROP-VL", "SZH",    3),
        ("ROP-VL", "UHO",    4),
        ("UHO",    "SZH",    5),
        ("UHO",    "ROP-VL", 6),
    ]

    print(f"\n{sep}")
    print("TABLE 2 — CROSS-DATASET  (train -> external test)  [XAI: GradCAM++, GradSHAP, Occlusion]")
    print(sep)
    print(f"{'Exp':<5} {'Train->Test':<18} {'Model':<18} "
          f"{'Acc(%)':>9} {'Macro-F1':>10} {'Wt-F1':>9} "
          f"{'TestLoss':>10} {'AvgDrop↓':>10}")
    print("-" * 90)
    for train_name, test_name, exp_no in cross_order:
        for model_key in models_to_run:
            r   = cross_results[model_key][train_name][test_name]
            tag = f"{train_name}->{test_name}"
            print(
                f"Exp{exp_no:<2}  {tag:<18} {model_key:<18} "
                f"{r['acc']:>8.2f}  "
                f"{r['macro_f1']:>9.4f}  "
                f"{r['weighted_f1']:>8.4f}  "
                f"{r['test_loss']:>9.4f}  "
                f"{r['avg_drop']:>9.4f}"
            )
        print()

    print(f"\nAll outputs -> {output_root}/")
    print(f"  XAI figures -> {output_root}/XAI/Cross/<model>/<TrainX_TestY>/img_N.png")


if __name__ == "__main__":
    main()