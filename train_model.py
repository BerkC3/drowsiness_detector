"""
Train a binary eye-state classifier (open vs closed).

Dataset layout:
    dataset/
        Closed_Eyes/   -> 0
        Open_Eyes/     -> 1

Run:  python train_model.py
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF

from model import EyeStateModel

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsiness_model.pth")

IMG_SIZE = (24, 24)
BATCH_SIZE = 128
EPOCHS = 30
VAL_SPLIT = 0.2
PATIENCE = 5


class CachedEyeDataset(Dataset):
    """All images sit in RAM; augmentation is tensor-only so it's fast."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].clone()
        label = self.labels[idx]

        if self.augment:
            if random.random() > 0.5:
                img = img.flip(-1)

            angle = random.uniform(-10, 10)
            dx = int(random.uniform(-0.1, 0.1) * IMG_SIZE[0])
            dy = int(random.uniform(-0.1, 0.1) * IMG_SIZE[1])
            img = TF.affine(img, angle=angle, translate=[dx, dy], scale=1.0, shear=0)

            brightness = 0.6 + random.random() * 0.8
            img = (img * brightness).clamp(0.0, 1.0)

        return img, label


def preload_dataset():
    """Read every image once, apply CLAHE, normalize, return tensors."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    base = datasets.ImageFolder(DATASET_DIR)
    total = len(base)

    print(f"Classes: {base.class_to_idx}")
    print(f"Total images: {total}")
    print("[INFO] Pre-loading into RAM ...")

    img_list, lbl_list = [], []
    skipped = 0

    for i, (path, label) in enumerate(base.samples):
        if (i + 1) % 10000 == 0 or i == 0:
            print(f"  [{i + 1}/{total}]")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = clahe.apply(img)
        img_list.append(img.astype(np.float32) / 255.0)
        lbl_list.append(label)

    if skipped:
        print(f"[WARN] Skipped {skipped} unreadable images")

    images = np.array(img_list, dtype=np.float32).reshape(-1, 1, *IMG_SIZE)
    labels = np.array(lbl_list, dtype=np.int64)
    print(f"[INFO] Cached {len(labels)} images ({images.nbytes / 1024 / 1024:.1f} MB)")

    return torch.from_numpy(images), torch.from_numpy(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset not found: {DATASET_DIR}")
        return

    all_images, all_labels = preload_dataset()

    n = len(all_labels)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))

    train_set = CachedEyeDataset(all_images[indices[:train_size]],
                                 all_labels[indices[:train_size]], augment=True)
    val_set = CachedEyeDataset(all_images[indices[train_size:]],
                               all_labels[indices[train_size:]], augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = EyeStateModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    print(model)
    print(f"Train: {train_size} | Val: {val_size}\n")

    best_val_loss = float("inf")
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_correct += ((torch.sigmoid(out) > 0.5).float() == lbls).sum().item()
            t_total += imgs.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.float().unsqueeze(1).to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                v_correct += ((torch.sigmoid(out) > 0.5).float() == lbls).sum().item()
                v_total += imgs.size(0)

        tl = t_loss / t_total
        vl = v_loss / v_total
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"train_loss={tl:.4f} acc={t_correct/t_total:.4f}  "
              f"val_loss={vl:.4f} acc={v_correct/v_total:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  -> saved (val_loss={vl:.4f})")
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest model: {MODEL_OUT}  (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
