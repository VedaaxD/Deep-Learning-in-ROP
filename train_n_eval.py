import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from load_data import AugmentedDataset, TestDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay


class TrainEval:

    def __init__(self, model, train_dir, test_dir, model_path, pth_filename, n_epochs, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.model_path = model_path
        self.num_epochs = n_epochs
        self.test_loader = None
        self.checkpoint_path = f"./checkpoint/{pth_filename.split('.')[0]}checkpoint.pth"
        self.train_loss = []
        self.val_loss = []


        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

    def _split_data(self, dataset):
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size])

    def _create_dataloader(self, augment):
        dataset = AugmentedDataset(self.train_dir, seed=0, augment=augment)
        test_data = TestDataset(self.test_dir, seed=0)

        train_data, val_data = self._split_data(dataset)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=8,
                                  pin_memory=True, shuffle=False, prefetch_factor=2)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=8,
                                pin_memory=True, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=8,
                                 pin_memory=True, shuffle=False)

        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

    def save_checkpoint(self, optimizer, scheduler, scaler, epoch):
        ckpt = {
            "model_state_dict": self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
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

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        loss_history = checkpoint.get("loss_history", ([], []))
        return optimizer, scheduler, scaler, start_epoch, loss_history


    def train_model(self, optimizer, criterion, scheduler, augment):
        scaler = GradScaler()

        #Load final model if exists
        if os.path.exists(self.model_path):
            weights = torch.load(self.model_path, map_location=self.device)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)
            print(f"Model loaded from {self.model_path}")
            return

        #Resume checkpoint
        try:
            optimizer, scheduler, scaler, start_epoch, loss_history = self.load_checkpoint(optimizer, scheduler, scaler)
            self.train_loss, self.val_loss = loss_history
            print(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            start_epoch = 1
            print("Started training from scratch")

        train_loader, val_loader, _ = self._create_dataloader(augment)

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

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

            #Validation
            self.model.eval()
            val_loss = 0.0
            correct, total = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).float().unsqueeze(1)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            self.val_loss.append(avg_val_loss)

            print(f"Epoch [{epoch}/{self.num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")

            scheduler.step(avg_val_loss)
            self.save_checkpoint(optimizer, scheduler, scaler, epoch)

        #Save final model
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), self.model_path)
        else:
            torch.save(self.model.state_dict(), self.model_path)

        print(f"Training complete. Model saved to {self.model_path}")

    def evaluate_model(self, exp_no):
        test_loader = self._create_dataloader(augment=False)[-1]

        weights = torch.load(self.model_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights)

        self.model.eval()

        all_labels = []
        all_probs = []

        os.makedirs(f"./ROC_Curves/Exp{exp_no}", exist_ok=True)

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                probs = torch.sigmoid(self.model(images))
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()

        preds = (all_probs > 0.5).astype(float)

        acc = 100 * np.sum(preds == all_labels) / len(all_labels)
        f1 = f1_score(all_labels, preds, zero_division=1)
        cm = confusion_matrix(all_labels, preds)

        print(f"Final Test Accuracy: {acc:.2f}%")
        print(f"Final Test F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n {cm}")

        specificity = cm[0][0] / cm.sum(axis=1)[0]
        sensitivity = cm[1][1] / cm.sum(axis=1)[1]

        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

        return acc, f1, specificity, sensitivity
