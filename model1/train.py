"""
Training script for the colorization U-Net.

Usage:
    python train.py --data_dir /path/to/color/images --epochs 30 --batch_size 16

Requires a folder of ordinary color images (jpg/png). A few thousand images
is enough to see the model start learning; tens of thousands+ gives much
better generalization. Good free sources: COCO, Places365, ImageNet subsets,
or a scrape of your own photo library.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import UNetColorizer
from dataset import ColorizationDataset, lab_to_rgb


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Training on: {device}")

    full_dataset = ColorizationDataset(args.data_dir, image_size=args.image_size)
    val_size = max(1, int(0.05 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    model = UNetColorizer(base_channels=args.base_channels).to(device)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    criterion = nn.L1Loss()  # L1 on ab channels tends to give less "muddy" colors than L2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=0.5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, (L, ab_true) in enumerate(train_loader):
            L, ab_true = L.to(device), ab_true.to(device)

            optimizer.zero_grad()
            ab_pred = model(L)
            loss = criterion(ab_pred, ab_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab_true in val_loader:
                L, ab_true = L.to(device), ab_true.to(device)
                ab_pred = model(L)
                val_loss += criterion(ab_pred, ab_true).item()
        val_loss /= len(val_loader)

        scheduler.step()
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | {elapsed:.1f}s")

        # Save latest + best checkpoints
        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        torch.save(model.state_dict(), latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved ({val_loss:.4f})")

    print("Training complete.")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a colorization U-Net")
    p.add_argument("--data_dir", type=str, required=True,
                    help="Folder of color images to train on")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_decay_every", type=int, default=10)
    p.add_argument("--base_channels", type=int, default=64,
                    help="Lower this (e.g. 32) if you run out of GPU memory")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to resume training from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
