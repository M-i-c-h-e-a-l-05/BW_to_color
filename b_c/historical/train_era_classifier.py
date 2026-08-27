"""
train_era_classifier.py
=======================

Training script for the Historical Era Classifier.

Backbone:
    ResNet18

Dataset Structure
-----------------

dataset/

    train/

        1900s/

        1920s/

        WWII/

        1950s/

        1960s/

        1970s/

        Modern/

    val/

        1900s/

        ...

"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from torch.utils.data import DataLoader

import copy

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

TRAIN_DIR = "dataset/train"

VAL_DIR = "dataset/val"

SAVE_DIR = "weights"

MODEL_NAME = "era_classifier_resnet18.pth"

NUM_CLASSES = 7

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-4

NUM_WORKERS = 4

DEVICE = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------
# Image Transforms
# -------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(

        IMAGE_SIZE,

        scale=(0.8, 1.0),

    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(

        brightness=0.20,

        contrast=0.20,

        saturation=0.20,

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    ),

])

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    ),

])

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

print()

print("=" * 60)

print("Loading Dataset")

print("=" * 60)

train_dataset = ImageFolder(

    TRAIN_DIR,

    transform=train_transform,

)

val_dataset = ImageFolder(

    VAL_DIR,

    transform=val_transform,

)

print()

print("Training Images :", len(train_dataset))

print("Validation Images :", len(val_dataset))

print()

print("Classes")

for idx, name in enumerate(train_dataset.classes):

    print(f"{idx:2d} : {name}")

print()

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

)

# -------------------------------------------------------
# Model
# -------------------------------------------------------

print("=" * 60)

print("Creating ResNet18")

print("=" * 60)

model = resnet18(

    weights="DEFAULT",

)

in_features = model.fc.in_features

model.fc = nn.Linear(

    in_features,

    NUM_CLASSES,

)

model = model.to(DEVICE)

print(model)

# -------------------------------------------------------
# Loss
# -------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,

)

# -------------------------------------------------------
# Scheduler
# -------------------------------------------------------

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3,

)

print()

print("=" * 60)

print("Training Configuration")

print("=" * 60)

print(f"Device        : {DEVICE}")

print(f"Epochs        : {EPOCHS}")

print(f"Batch Size    : {BATCH_SIZE}")

print(f"Learning Rate : {LEARNING_RATE}")

print(f"Classes       : {NUM_CLASSES}")

print("=" * 60)


# -------------------------------------------------------
# Early Stopping
# -------------------------------------------------------

EARLY_STOPPING = 8

patience_counter = 0

best_weights = copy.deepcopy(model.state_dict())


# -------------------------------------------------------
# Training Function
# -------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        running_correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total

    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


# -------------------------------------------------------
# Validation Function
# -------------------------------------------------------

@torch.no_grad()

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    running_correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        running_correct += (predicted == labels).sum().item()

    val_loss = running_loss / total

    val_acc = running_correct / total

    return val_loss, val_acc


# -------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------

best_accuracy = 0.0

history = {

    "train_loss": [],

    "train_acc": [],

    "val_loss": [],

    "val_acc": []

}

print()

print("=" * 60)

print("Starting Training")

print("=" * 60)

start_epoch = 0

resume_checkpoint = os.path.join(
    SAVE_DIR,
    "last_checkpoint.pth"
)

if os.path.exists(resume_checkpoint):

    print()

    print("=" * 60)

    print("Resuming Training")

    print("=" * 60)

    checkpoint = torch.load(
        resume_checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    best_accuracy = checkpoint.get(
        "best_accuracy",
        0.0
    )

    print(f"Resuming from Epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):

    print()

    print(f"Epoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(

        model,

        train_loader,

        criterion,

        optimizer,

        DEVICE,

    )

    val_loss, val_acc = validate(

        model,

        val_loader,

        criterion,

        DEVICE,

    )

    scheduler.step(val_acc)

    history["train_loss"].append(train_loss)

    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)

    history["val_acc"].append(val_acc)

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Val Acc    : {val_acc:.4f}")

    # -------------------------------------------------------
    # Save Last Checkpoint (Every Epoch)
    # -------------------------------------------------------

    last_checkpoint = os.path.join(

        SAVE_DIR,

        "last_checkpoint.pth"

    )

    torch.save(

        {

            "epoch": epoch + 1,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "scheduler_state_dict": scheduler.state_dict(),

            "best_accuracy": best_accuracy,

            "train_loss": train_loss,

            "train_accuracy": train_acc,

            "validation_loss": val_loss,

            "validation_accuracy": val_acc,

            "classes": train_dataset.classes,

        },

        last_checkpoint,

    )

    # -------------------------------------------------------
    # Save Best Model
    # -------------------------------------------------------

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_weights = copy.deepcopy(
                    model.state_dict()
                )
        
        patience_counter = 0

        best_model = os.path.join(

            SAVE_DIR,

            "best_model.pth",

        )


        torch.save(

            {

                "epoch": epoch + 1,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "scheduler_state_dict": scheduler.state_dict(),

                "best_accuracy": best_accuracy,

                "train_loss": train_loss,

                "validation_loss": val_loss,

                "classes": train_dataset.classes,

            },

            best_model,

        )

        print()

        print("=" * 50)

        print("✓ New Best Model Saved")

        print(f"Validation Accuracy : {val_acc:.4f}")

        print(f"Epoch               : {epoch+1}")

        print("=" * 50)

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{EARLY_STOPPING})"
        )

        if patience_counter >= EARLY_STOPPING:

            print()

            print("="*60)

            print("Early stopping triggered.")

            print("="*60)

            break

print()

print("=" * 60)

model.load_state_dict(best_weights)

print("Training Complete")

print("=" * 60)

print(f"Best Validation Accuracy : {best_accuracy:.4f}")

print()

print("Saved Files")

print(f"Best Model      : {os.path.join(SAVE_DIR,'best_model.pth')}")

print(f"Last Checkpoint : {os.path.join(SAVE_DIR,'last_checkpoint.pth')}")

print("=" * 60)

print("\nEvaluating Best Model...")

model.eval()

predictions = []

ground_truth = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())

        ground_truth.extend(labels.numpy())

print(

    classification_report(

        ground_truth,

        predictions,

        target_names=train_dataset.classes,

    )

)

cm = confusion_matrix(
    ground_truth,
    predictions
)

print(cm)


plt.figure(figsize=(10,5))

plt.plot(history["train_loss"], label="Train")

plt.plot(history["val_loss"], label="Validation")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        SAVE_DIR,

        "loss_curve.png"

    )

)

plt.close()

plt.figure(figsize=(10,5))

plt.plot(history["train_acc"], label="Train")

plt.plot(history["val_acc"], label="Validation")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        SAVE_DIR,

        "accuracy_curve.png"

    )

)

plt.close()

NUM_CLASSES