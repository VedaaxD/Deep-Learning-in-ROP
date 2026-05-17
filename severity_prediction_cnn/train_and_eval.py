import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from load_data import StageDataset, TestDataset

safe_num_workers = 2

class TrainEval:
    def __init__(self, model, train_dir, test_dir, model_path, pth_filename,
                 n_epochs, batch_size=32, output_root="/home/veda/stage_prediction_combined_cv_cnn/results"):

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = model.to(self.device)
        self.train_dir   = train_dir
        self.test_dir    = test_dir
        self.batch_size  = batch_size
        self.model_path  = model_path
        self.num_epochs  = n_epochs
        self.output_root = output_root

        self.test_loader     = None
        self.checkpoint_path = os.path.join(
            output_root, "checkpoint",
            pth_filename.replace(".pth", "_checkpoint.pth")
        )
        self.train_loss = []
        self.val_loss   = []

    def _make_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=safe_num_workers,
            pin_memory=True,
            shuffle=shuffle,
            persistent_workers=(safe_num_workers > 0),
        )

    def create_dataloader_cross(self, augment=False):
        """80/20 train/val split on train_dir; test_dir is the external test set."""
        train_dataset = StageDataset(self.train_dir, seed=0, augment=augment)
        test_dataset  = TestDataset(self.test_dir,  seed=0)

        train_len = int(0.8 * len(train_dataset))
        val_len   = len(train_dataset) - train_len
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = self._make_loader(train_set,    shuffle=True)
        val_loader   = self._make_loader(val_set,      shuffle=False)
        test_loader  = self._make_loader(test_dataset, shuffle=False)

        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def create_dataloader_intrafold(self, fold_idx, k=10, augment=False):
        """
        Pure k-fold CV: the held-out fold IS the test set.
        No external test data used.

        Train split  = all folds except fold_idx   (k-1 folds)
        Test  split  = fold_idx only               (1 fold)
        Val   split  = 20% of the train split (for early stopping)
        """
        full_dataset = StageDataset(self.train_dir, seed=0, augment=False)
        labels       = [label for _, label in full_dataset.data]

        skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(skf.split(range(len(full_dataset.data)), labels))
        train_val_idx, test_idx = splits[fold_idx]

        # Further split train_val -> 80% train / 20% val (for early stopping)
        n_val     = max(1, int(0.2 * len(train_val_idx)))
        rng       = np.random.default_rng(42)
        shuffled  = rng.permutation(train_val_idx)
        val_idx   = shuffled[:n_val]
        train_idx = shuffled[n_val:]

        aug_dataset  = StageDataset(self.train_dir, seed=0, augment=augment)
        val_dataset  = StageDataset(self.train_dir, seed=0, augment=False)
        test_dataset = StageDataset(self.train_dir, seed=0, augment=False)  # same dataset, held-out fold

        train_loader = self._make_loader(Subset(aug_dataset,  train_idx), shuffle=True)
        val_loader   = self._make_loader(Subset(val_dataset,  val_idx),   shuffle=False)
        test_loader  = self._make_loader(Subset(test_dataset, test_idx),  shuffle=False)

        self.test_loader = test_loader
        return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx

    def create_dataloader_kfold(self, fold_idx, k=5, augment=False):
        """
        k-fold on train_dir for stability; external test_dir used for evaluation.
        Kept for cross-dataset experiments.
        """
        full_dataset = StageDataset(self.train_dir, seed=0, augment=False)
        labels       = [label for _, label in full_dataset.data]

        skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(skf.split(range(len(full_dataset.data)), labels))
        train_idx, val_idx = splits[fold_idx]

        aug_dataset = StageDataset(self.train_dir, seed=0, augment=augment)
        val_dataset = StageDataset(self.train_dir, seed=0, augment=False)

        train_loader = self._make_loader(Subset(aug_dataset, train_idx), shuffle=True)
        val_loader   = self._make_loader(Subset(val_dataset, val_idx),   shuffle=False)

        test_dataset = TestDataset(self.test_dir, seed=0)
        test_loader  = self._make_loader(test_dataset, shuffle=False)

        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def save_checkpoint(self, optimizer, scheduler, scaler, epoch):
        ckpt = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict":    scaler.state_dict(),
            "epoch":                epoch,
            "loss_history":         (self.train_loss, self.val_loss),
        }
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(ckpt, self.checkpoint_path)

    def load_checkpoint(self, optimizer, scheduler, scaler):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch  = checkpoint["epoch"] + 1
        loss_history = checkpoint.get("loss_history", ([], []))
        return optimizer, scheduler, scaler, start_epoch, loss_history

    def train_model(self, optimizer, criterion, scheduler,
                    train_loader, val_loader,
                    early_stopping_patience=7):
        """
        Unified training method.
        Loaders are passed in explicitly so both intra- and cross-dataset
        experiments can supply their own splits without ambiguity.
        """
        if os.path.exists(self.model_path):
            print(f"Model exists at {self.model_path}. Skipping training.")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            return

        scaler = GradScaler(device='cuda')

        try:
            optimizer, scheduler, scaler, start_epoch, loss_history = \
                self.load_checkpoint(optimizer, scheduler, scaler)
            self.train_loss, self.val_loss = loss_history
            print(f"Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            start_epoch = 1
            self.train_loss, self.val_loss = [], []
            print("Starting from scratch")

        best_val_loss     = float('inf')
        epochs_no_improve = 0
        best_model_state  = None

        for epoch in range(start_epoch, self.num_epochs + 1):

            self.model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.train_loss.append(avg_train_loss)

            self.model.eval()
            val_loss = 0.0
            correct  = 0
            total    = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).long()
                    outputs  = self.model(images)
                    loss     = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds    = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            self.val_loss.append(avg_val_loss)

            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {100 * correct / total:.2f}%")

            scheduler.step(avg_val_loss)
            self.save_checkpoint(optimizer, scheduler, scaler, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss     = avg_val_loss
                epochs_no_improve = 0
                best_model_state  = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                print(f"  [EarlyStop] {epochs_no_improve}/{early_stopping_patience}")
                if epochs_no_improve >= early_stopping_patience:
                    print(f"  Triggered at epoch {epoch}. Best val: {best_val_loss:.4f}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Saved -> {self.model_path}")
        self.plot_loss_curve()

    def evaluate_model(self, exp_no, fold=None, test_loader=None,
                       test_dir=None, test_name=None):
        """
        Evaluate the saved model.

        For intra-dataset CV: pass test_loader directly (held-out fold subset).
        For cross-dataset:    pass test_dir (or leave as self.test_dir).

        Returns: acc, macro_f1, weighted_f1, test_loss
        """
        if test_loader is not None:
            loader = test_loader
        elif test_dir is not None:
            loader = self._make_loader(TestDataset(test_dir, seed=0), shuffle=False)
        else:
            loader = self._make_loader(TestDataset(self.test_dir, seed=0), shuffle=False)

        eval_name = test_name or (os.path.basename(test_dir) if test_dir else "test")

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()

        criterion  = nn.CrossEntropyLoss() #unweighted for test loss
        all_labels = []
        all_preds  = []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                outputs     = self.model(images)
                total_loss += criterion(outputs, labels).item()
                preds       = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        all_labels  = np.array(all_labels)
        all_preds   = np.array(all_preds)
        test_loss   = total_loss / len(loader)
        acc         = 100 * np.sum(all_preds == all_labels) / len(all_labels)
        macro_f1    = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        cm          = confusion_matrix(all_labels, all_preds)

        print(f"\n  [Eval: {eval_name}]  Loss={test_loss:.4f}  "
              f"Acc={acc:.2f}%  MacroF1={macro_f1:.4f}  WtF1={weighted_f1:.4f}")
        print(f"  CM:\n{cm}")

        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Mild", "Severe"])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"Exp{exp_no} | {eval_name}" +
                  (f" | Fold {fold}" if fold is not None else ""))

        fold_suffix = f"_fold{fold}" if fold is not None else ""
        cm_dir = os.path.join(self.output_root, "Confusion_Matrix",
                              f"Exp{exp_no}_{eval_name}")
        os.makedirs(cm_dir, exist_ok=True)
        plt.savefig(os.path.join(cm_dir, f"cm{fold_suffix}.png"))
        plt.close()

        return acc, macro_f1, weighted_f1, test_loss

    def plot_loss_curve(self):
        loss_dir = os.path.join(self.output_root, "Loss_Curves")
        os.makedirs(loss_dir, exist_ok=True)
        epochs = list(range(1, len(self.train_loss) + 1))
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(epochs, self.train_loss, label="Train Loss")
        plt.plot(epochs, self.val_loss,   label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(loss_dir,
                    os.path.basename(self.model_path).replace(".pth", "_loss.png")))
        plt.close()
