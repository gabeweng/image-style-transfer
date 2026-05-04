"""
train_classifier.py

Fine-tunes ResNet-18 with two classification heads:
  - time_of_day  (daytime / sunset / night)
  - weather      (sunny / cloudy / rainy)

Combined loss: L = L_tod + L_weather

Usage (EC2 / Colab):
    python scripts/train_classifier.py \
        --images_dir   /path/to/CIS_5190_group_project \
        --labels_csv   /path/to/manifest.csv \
        --output_dir   checkpoints/ \
        --epochs       20 \
        --batch_size   32

Checkpoint: checkpoints/classifier_best.pt
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm.auto import tqdm

register_heif_opener()

TOD_CLASSES = ["daytime", "sunset", "night"]
WX_CLASSES = ["sunny", "cloudy", "rainy"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_label_frame(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    if {"status", "file_name", "time_of_day", "weather"}.issubset(df.columns):
        df = df[df["status"] == "kept"].copy()
        df["image_file"] = df["file_name"]
        return df

    # Backward-compatible fallback for older pair CSVs.
    if {"warped_path", "target_tod", "target_weather"}.issubset(df.columns):
        df = df.copy()
        df["image_file"] = df["warped_path"]
        df["time_of_day"] = df["target_tod"]
        df["weather"] = df["target_weather"]
        return df

    raise ValueError(
        "Unsupported labels CSV. Expected manifest.csv with status/file_name/time_of_day/weather "
        "or an older pair CSV with warped_path/target_tod/target_weather."
    )


class ConditionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_dir, row["image_file"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        tod_label = TOD_CLASSES.index(row["time_of_day"].lower())
        wx_label = WX_CLASSES.index(row["weather"].lower())
        return img, tod_label, wx_label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualHeadResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.tod_head = nn.Linear(in_features, len(TOD_CLASSES))
        self.wx_head = nn.Linear(in_features, len(WX_CLASSES))

    def forward(self, x):
        feats = self.backbone(x)
        return self.tod_head(feats), self.wx_head(feats)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(images_dir, labels_csv, output_dir, epochs, batch_size, lr, val_split, resume):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_label_frame(labels_csv)
    df["time_of_day"] = df["time_of_day"].str.lower().str.strip()
    df["weather"] = df["weather"].str.lower().str.strip()

    # Drop rows whose label isn't in our class lists
    df = df[df["time_of_day"].isin(TOD_CLASSES) & df["weather"].isin(WX_CLASSES)]
    df = df[df["image_file"].apply(lambda p: os.path.exists(os.path.join(images_dir, p)))]
    print(f"Dataset size after filtering: {len(df)}")
    if df.empty:
        raise ValueError("No classifier training rows found. Check --images_dir and --labels_csv.")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(448, padding=32),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_size = max(1, int(len(df) * val_split))
    train_size = len(df) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(df.reset_index(drop=True), [train_size, val_size], generator=generator)
    train_df = df.iloc[train_subset.indices].reset_index(drop=True)
    val_df = df.iloc[val_subset.indices].reset_index(drop=True)
    train_ds = ConditionDataset(train_df, images_dir, train_tf)
    val_ds = ConditionDataset(val_df, images_dir, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DualHeadResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    start_epoch = 1

    last_ckpt_path = os.path.join(output_dir, "classifier_last.pt")
    if resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from {last_ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for imgs, tod_labels, wx_labels in progress:
            imgs = imgs.to(device)
            tod_labels = tod_labels.to(device)
            wx_labels = wx_labels.to(device)

            tod_logits, wx_logits = model(imgs)
            loss = criterion(tod_logits, tod_labels) + criterion(wx_logits, wx_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # --- Validate ---
        model.eval()
        tod_correct = wx_correct = total = 0
        with torch.no_grad():
            for imgs, tod_labels, wx_labels in val_loader:
                imgs = imgs.to(device)
                tod_labels = tod_labels.to(device)
                wx_labels = wx_labels.to(device)

                tod_logits, wx_logits = model(imgs)
                tod_correct += (tod_logits.argmax(1) == tod_labels).sum().item()
                wx_correct += (wx_logits.argmax(1) == wx_labels).sum().item()
                total += len(imgs)

        tod_acc = tod_correct / total
        wx_acc = wx_correct / total
        avg_acc = (tod_acc + wx_acc) / 2
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={total_loss/len(train_loader):.4f} | "
            f"val tod={tod_acc:.3f} wx={wx_acc:.3f} avg={avg_acc:.3f}"
        )

        is_best = avg_acc > best_val_acc
        if is_best:
            best_val_acc = avg_acc

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tod_classes": TOD_CLASSES,
            "wx_classes": WX_CLASSES,
            "val_tod_acc": tod_acc,
            "val_wx_acc": wx_acc,
            "best_val_acc": best_val_acc,
        }, last_ckpt_path)
        print(f"  Saved latest checkpoint: {last_ckpt_path}")

        if is_best:
            ckpt_path = os.path.join(output_dir, "classifier_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "tod_classes": TOD_CLASSES,
                "wx_classes": WX_CLASSES,
                "val_tod_acc": tod_acc,
                "val_wx_acc": wx_acc,
                "best_val_acc": best_val_acc,
            }, ckpt_path)
            print(f"  ↳ Saved best checkpoint (avg_acc={avg_acc:.3f})")

    print(f"\nTraining complete. Best avg val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {os.path.join(output_dir, 'classifier_best.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Dataset base folder. For manifest.csv, pass the project base directory.")
    parser.add_argument("--labels_csv", required=True, help="manifest.csv from scripts/preprocess.py or the notebook")
    parser.add_argument("--output_dir", default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Resume from classifier_last.pt if present")
    args = parser.parse_args()

    train(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        resume=args.resume,
    )
