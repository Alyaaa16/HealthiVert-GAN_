# %% [markdown]
# # 2D SE-ResNet50 Classifier Training for Vertebral Fracture Detection
# 
# This notebook trains a binary classifier (Normal vs. Fracture) on 2D sagittal slices of straightened vertebrae. 
# The trained model is required for generating Grad-CAM heatmaps used in the HealthiVert-GAN pipeline.

# %%
# Install MONAI if not already installed
!pip install monai

# %%
import os
import json
import glob
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SEresnet50
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    ToTensor,
    RandRotate,
    RandFlip,
    RandZoom
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# %%
# ==========================================
# Configuration
# ==========================================


# Root directory for the project
PROJECT_ROOT = Path("..").resolve()
DATA_DIR = Path("/kaggle/input/straightened-vertebrae-30s/CT")
JSON_PATH = Path("/kaggle/input/verse-19-genant-fracture-grades/vertebra_data.json")
OUTPUT_DIR = Path("./checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Data Directory: {DATA_DIR}")

# --- DEBUGGING: CHECK PATHS ---
if DATA_DIR.exists():
    print(f"Contents of {DATA_DIR}:")
    try:
        files = list(DATA_DIR.glob("*"))
        print(f"Found {len(files)} items. First 5: {[f.name for f in files[:5]]}")
    except Exception as e:
        print(f"Error listing {DATA_DIR}: {e}")
else:
    print(f"WARNING: {DATA_DIR} does not exist!")
    # Check parent to see where we are
    parent = DATA_DIR.parent
    if parent.exists():
        print(f"Contents of parent ({parent}):")
        print([p.name for p in parent.iterdir()])
    else:
        print(f"Parent {parent} also does not exist.")

# %%
# ==========================================
# Dataset Definition
# ==========================================


from torchvision import models

# ... (Previous imports remain, remove monai.networks.nets.SEresnet50 if unused)

class SagittalSliceDataset(Dataset):
    def __init__(self, data_dir, json_path, split='train', transform=None, slice_range=15):
        # ... (init code remains same) ...
        self.split = split # Store split
        # ...

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        label = sample['label']
        
        try:
            img_nii = nib.load(path)
            img_arr = img_nii.get_fdata()
            
            # Extract Sagittal Slice
            z_center = img_arr.shape[2] // 2
            
            # TRAINING: Random slice augmentation
            if self.split == 'train':
                low = max(0, z_center - self.slice_range)
                high = min(img_arr.shape[2], z_center + self.slice_range)
                slice_idx = random.randint(low, high - 1) if high > low else z_center
            # VALIDATION/TEST: Always take the center slice (Deterministic)
            else:
                slice_idx = z_center
            
            slice_img = img_arr[:, :, slice_idx]
            
            # Add Channel Dimension (1, H, W)
            slice_img = slice_img[np.newaxis, ...]
            
            # Replicate to 3 channels for ImageNet pretrained model
            # (1, H, W) -> (3, H, W)
            slice_img = np.repeat(slice_img, 3, axis=0)
            
            if self.transform:
                slice_img = self.transform(slice_img)
            
            return slice_img.float(), torch.tensor(label).long()
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, 256, 256)), torch.tensor(0).long() # Return 3-channel zero tensor

# ... (Transforms and DataLoaders remain, ensure normalization matches ImageNet if possible, but ScaleIntensity is okay for now)

# ==========================================
# Model Setup
# ==========================================

# Use Pretrained ResNet50
# Note: SE-ResNet is better, but training from scratch on 30 images is impossible.
# Transfer learning is required.
print("Loading Pretrained ResNet50...")
model = models.resnet50(weights='IMAGENET1K_V1')

# Modify the first layer is NOT needed if we replicate input channels (easier for transfer learning stability)
# But standard practice is usually modifying first conv. 
# Let's stick to 3-channel input replication in Dataset (done above) to preserve pretrained weights power.

# Modify correct final layer num_classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # Normal vs Fracture

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %%
# ==========================================
# Training Loop
# ==========================================

best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    # --- TRAINING ---
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, targets in tqdm(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        
    if len(train_loader) > 0:
        epoch_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}")
    
    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(targets.cpu().numpy())
            
    if len(val_loader) > 0:
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = OUTPUT_DIR / "best_ckpt.tar"
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': best_acc
            }, save_path)
            print(f"New Best Model Saved to {save_path}!")

print("Training Complete.")

# %%
# ==========================================
# Validation Report
# ==========================================

if len(val_labels) > 0:
    print("Classification Report ON TEST SET:")
    print(classification_report(val_labels, val_preds, target_names=['Normal', 'Fracture']))
else:
    print("No validation data found to report.")


