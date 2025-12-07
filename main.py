# =========================
#  CT Detection: Zero to Hero
#  Task: 3D Lung Nodule Detection using UNETR (MONAI)
#  Dataset: LUNA16 style (you can use subsets or full dataset)
# =========================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, Resized, ToTensord,
    RandCropByPosNegLabeld, RandAffined, RandGaussianNoised,
    EnsureTyped, NormalizeIntensityd
)
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, Dataset as MonaiDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import wandb  # Optional: for experiment tracking

# -----------------------------
# 1. Custom Dataset (for .nii.gz CT and segmentation)
# -----------------------------
class CTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        # Validate paths exist
        for img_path, mask_path in zip(image_paths, mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data = {
            "image": self.image_paths[idx],
            "label": self.mask_paths[idx]
        }
        if self.transforms:
            data = self.transforms(data)
        return data

# -----------------------------
# 2. Data Transforms (MONAI) - Fixed
# -----------------------------
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
    # Fixed: Properly handle batch data from RandCropByPosNegLabeld
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1.0,
        neg=1.0,
        num_samples=2  # Reduced for memory efficiency
    ),
    RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=0.3,
        spatial_size=(96, 96, 96),
        rotate_range=(0, 0, np.pi/12),
        scale_range=(0.1, 0.1, 0.1),
        translate_range=(5, 5, 5)
    ),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),  # Fixed mean parameter
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

# -----------------------------
# 3. Prepare File Lists with Cross-Validation
# -----------------------------
def prepare_data_splits(data_dir="data", k_folds=5, fold=0):
    """Prepare data with k-fold cross-validation"""
    image_files = sorted([
        os.path.join(data_dir, "images", f) 
        for f in os.listdir(os.path.join(data_dir, "images")) 
        if f.endswith(".nii.gz")
    ])
    label_files = sorted([
        os.path.join(data_dir, "labels", f) 
        for f in os.listdir(os.path.join(data_dir, "labels")) 
        if f.endswith(".nii.gz")
    ])
    
    # Verify matching pairs
    assert len(image_files) == len(label_files), "Mismatch between images and labels"
    for img, lbl in zip(image_files, label_files):
        base_img = os.path.basename(img).replace(".nii.gz", "")
        base_lbl = os.path.basename(lbl).replace(".nii.gz", "")
        if base_img != base_lbl:
            print(f"Warning: Mismatched pair - {base_img} vs {base_lbl}")
    
    # K-fold split
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_indices = list(range(len(image_files)))
    
    for i, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        if i == fold:
            train_images = [image_files[idx] for idx in train_idx]
            train_labels = [label_files[idx] for idx in train_idx]
            val_images = [image_files[idx] for idx in val_idx]
            val_labels = [label_files[idx] for idx in val_idx]
            break
    
    return train_images, train_labels, val_images, val_labels

# Prepare datasets
train_images, train_labels, val_images, val_labels = prepare_data_splits()

train_ds = CTDataset(train_images, train_labels, transforms=train_transforms)
val_ds = CTDataset(val_images, val_labels, transforms=val_transforms)

# Use MONAI's DataLoader for better handling of batch transforms
train_loader = DataLoader(
    train_ds, 
    batch_size=2, 
    shuffle=True, 
    num_workers=2,  # Reduced for stability
    pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_ds, 
    batch_size=1, 
    shuffle=False, 
    num_workers=1,
    pin_memory=True if torch.cuda.is_available() else False
)

# -----------------------------
# 4. Enhanced Model Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.1,  # Added dropout for regularization
    conv_block=True
).to(device)

# Print model summary
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Enhanced loss function - combination of Dice and Focal Loss
dice_loss = DiceLoss(sigmoid=True)
focal_loss = nn.BCEWithLogitsLoss()  # For binary classification

def combined_loss(pred, target, alpha=0.7):
    """Combined Dice + Focal loss"""
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return alpha * dice + (1 - alpha) * focal

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-5,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# Metrics
dice_metric = DiceMetric(include_background=False, reduction="mean")
hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

# -----------------------------
# 5. Improved Training Loop with Validation
# -----------------------------
def train_epoch(model, loader, optimizer, device):
    """Training for one epoch"""
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in loader:
        step += 1
        
        # Handle batch data properly
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Handle different batch sizes from augmentation
        if outputs.shape[0] != labels.shape[0]:
            # Use the first sample from each batch
            outputs = outputs[:labels.shape[0]]
        
        loss = combined_loss(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        
        if step % 10 == 0:
            print(f"  Step {step}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return epoch_loss / max(step, 1)

def validate(model, loader, device):
    """Validation with multiple metrics"""
    model.eval()
    dice_values = []
    hausdorff_values = []
    
    with torch.no_grad():
        for val_data in loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            
            # Use sliding window for inference
            val_outputs = sliding_window_inference(
                val_inputs, 
                roi_size=(96, 96, 96), 
                sw_batch_size=1,  # Reduced for memory
                predictor=model,
                overlap=0.5,
                mode="gaussian"
            )
            
            # Threshold predictions
            val_outputs = torch.sigmoid(val_outputs)
            val_preds = (val_outputs > 0.5).float()
            
            # Calculate metrics
            dice_metric(y_pred=val_preds, y=val_labels)
            hausdorff_metric(y_pred=val_preds, y=val_labels)
            
            # Store individual values
            dice_batch = dice_metric.aggregate().item()
            hausdorff_batch = hausdorff_metric.aggregate().item()
            
            dice_values.append(dice_batch)
            hausdorff_values.append(hausdorff_batch)
            
            dice_metric.reset()
            hausdorff_metric.reset()
    
    return np.mean(dice_values), np.mean(hausdorff_values)

# Training loop
max_epochs = 100
val_interval = 1
best_dice = 0
best_metrics = {"dice": 0, "hausdorff": float('inf')}

for epoch in range(max_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{max_epochs}")
    print(f"{'='*50}")
    
    # Training phase
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    
    # Validation phase
    if (epoch + 1) % val_interval == 0:
        dice_score, hausdorff_dist = validate(model, val_loader, device)
        print(f"Validation - Dice: {dice_score:.4f}, Hausdorff: {hausdorff_dist:.2f}")
        
        # Update learning rate based on validation dice
        scheduler.step(dice_score)
        
        # Save best model
        if dice_score > best_dice:
            best_dice = dice_score
            best_metrics = {"dice": dice_score, "hausdorff": hausdorff_dist}
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'metrics': best_metrics,
            }, "best_unetr_ct_nodule.pth")
            
            print(f"✅ New best model saved! Dice: {dice_score:.4f}")
        
        # Early stopping check (optional)
        if epoch > 20 and dice_score < best_dice - 0.05:
            print("Early stopping triggered")
            break

print(f"\n{'='*50}")
print(f"Training completed!")
print(f"Best Dice: {best_metrics['dice']:.4f}")
print(f"Best Hausdorff Distance: {best_metrics['hausdorff']:.2f}")
print(f"{'='*50}")

# -----------------------------
# 6. Enhanced Inference Pipeline
# -----------------------------
def predict_single_scan(model_path, scan_path, output_path="prediction_nodule.nii.gz"):
    """Predict nodules for a single CT scan"""
    
    # Load best model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Inference transforms (no resizing to preserve original dimensions)
    infer_transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    
    # Load and transform the scan
    test_data = infer_transforms({"image": scan_path})
    test_input = test_data["image"].to(device).unsqueeze(0)
    
    # Get original affine for saving
    original_affine = test_data["image_meta_dict"]["affine"]
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        # Use sliding window inference
        pred = sliding_window_inference(
            test_input,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.6,  # Higher overlap for better boundaries
            mode="gaussian"
        )
        
        # Apply sigmoid and threshold
        pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
        pred_binary = (pred_prob > 0.5).astype(np.uint8)
    
    # Save prediction
    nifti_img = nib.Nifti1Image(pred_binary, affine=original_affine)
    nib.save(nifti_img, output_path)
    
    # Also save probability map
    prob_output = output_path.replace(".nii.gz", "_prob.nii.gz")
    nifti_prob = nib.Nifti1Image(pred_prob.astype(np.float32), affine=original_affine)
    nib.save(nifti_prob, prob_output)
    
    print(f"✅ Prediction saved: {output_path}")
    print(f"✅ Probability map saved: {prob_output}")
    
    return pred_binary, pred_prob

# Example usage
if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    
    # For inference on a new scan:
    # scan_path = "data/images/new_patient.nii.gz"
    # prediction, probabilities = predict_single_scan(
    #     "best_unetr_ct_nodule.pth", 
    #     scan_path,
    #     "predictions/new_patient_nodules.nii.gz"
    # )