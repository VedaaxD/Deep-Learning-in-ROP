import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from load_data import StageDataset, TestDataset


class TrainEval:
    def __init__(self, model, train_dir, test_dir, model_path, pth_filename, n_epochs, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_dir = train_dir
        self.test_dir = test_dir

        self.batch_size = batch_size
        self.model_path = model_path
        self.num_epochs = n_epochs

        self.test_loader = None
        self.checkpoint_path = f"./checkpoint/{pth_filename.split('.')[0]}_checkpoint.pth"

        self.train_loss = []
        self.val_loss = []
#DATA
    def split_data(self, dataset):
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
        return train_set, val_set

    def create_dataloader(self, augment=False):

        train_dataset = StageDataset(self.train_dir, seed=0, augment=augment)
        test_dataset = TestDataset(self.test_dir, seed=0)
        train_set, val_set = self.split_data(train_dataset)

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=True
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False
        )

        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def create_dataloader_kfold(self, fold_idx, k=5, augment=False):
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import Subset

        full_dataset = StageDataset(self.train_dir, seed=0, augment=False)  #no aug for split logic
        labels = [label for _, label in full_dataset.data]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(skf.split(range(len(full_dataset.data)), labels))
        train_idx, val_idx = splits[fold_idx]

        # Re-create with augment flag for actual training
        aug_dataset = StageDataset(self.train_dir, seed=0, augment=augment)
        val_dataset  = StageDataset(self.train_dir, seed=0, augment=False)

        train_set = Subset(aug_dataset, train_idx)
        val_set   = Subset(val_dataset,  val_idx)

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                              num_workers=16, pin_memory=True, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=self.batch_size,
                              num_workers=16, pin_memory=True, shuffle=False)

        test_dataset = TestDataset(self.test_dir, seed=0)
        test_loader  = DataLoader(test_dataset, batch_size=self.batch_size,
                              num_workers=16, pin_memory=True, shuffle=False)

        self.test_loader = test_loader
        return train_loader, val_loader, test_loader
    def save_checkpoint(self, optimizer, scheduler, scaler, epoch):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "loss_history": (self.train_loss, self.val_loss)
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

        start_epoch = checkpoint["epoch"] + 1
        loss_history = checkpoint.get("loss_history", ([], []))
        return optimizer, scheduler, scaler, start_epoch, loss_history
    def train_model(self, optimizer, criterion, scheduler, augment=False):

        if os.path.exists(self.model_path):
            print(f"Final model already exists at {self.model_path}. Skipping training.")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            return

        scaler = GradScaler()
         # Use fold-specific loader if in k-fold mode, else original loader
        if hasattr(self, '_kfold_params'):
            fold_idx, k = self._kfold_params
            train_loader, val_loader, _ = self.create_dataloader_kfold(fold_idx, k, augment=augment)
        else:
            train_loader, val_loader, _ = self.create_dataloader(augment=augment)

        try:
            optimizer, scheduler, scaler, start_epoch, loss_history = self.load_checkpoint(
                optimizer, scheduler, scaler
            )
            self.train_loss, self.val_loss = loss_history
            print(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            start_epoch = 1
            self.train_loss, self.val_loss = [], []
            print("Starting training from scratch")

        for epoch in range(start_epoch, self.num_epochs + 1):

            #train
            self.model.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                optimizer.zero_grad()

                with autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.train_loss.append(avg_train_loss)

            # validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).long()

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            self.val_loss.append(avg_val_loss)

            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"| Train Loss: {avg_train_loss:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f} "
                  f"| Val Acc: {100 * correct / total:.2f}%")

            scheduler.step(avg_val_loss)
            self.save_checkpoint(optimizer, scheduler, scaler, epoch)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Training complete. Model saved to {self.model_path}")
        self.plot_loss_curve()
#evaluation

    def evaluate_model(self, exp_no, fold=None):

        _, _, test_loader = self.create_dataloader(augment=False)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        acc = 100 * np.sum(all_preds == all_labels) / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nFinal Test Accuracy: {acc:.2f}%")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"Weighted F1 Score: {weighted_f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Normal", "Mild", "Severe"])
        disp.plot(cmap=plt.cm.Blues, values_format='d')

        suffix=f"_fold{fold}" if fold is not None else ""
        os.makedirs(f"./Confusion_Matrix/Exp{exp_no}", exist_ok=True)
        cm_path = f"./Confusion_Matrix/Exp{exp_no}/confusion_matrix{suffix}.png"
        plt.savefig(cm_path)
        plt.close()

        return acc, macro_f1, weighted_f1

   #Loss curve
    def plot_loss_curve(self):
        os.makedirs("./Loss_Curves", exist_ok=True)

        epochs = list(range(1, len(self.train_loss) + 1))

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(epochs, self.train_loss, label="Train Loss")
        plt.plot(epochs, self.val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./Loss_Curves/loss_curve.png")
        plt.close()
