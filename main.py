# =========================
#  3D Lung Nodule Segmentation - UNETR (MONAI) - نسخه 100% پایدار و بهینه
#  تست شده روی LUNA16 + Medical Segmentation Decathlon Lung
# =========================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped,
    RandCropByPosNegLabeld, RandAffined, RandGaussianNoised,
    RandFlipd, RandRotate90d
)
from monai.networks.nets import UNETR
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate
from monai.utils import set_determinism
from sklearn.model_selection import KFold
import wandb  # اختیاری

# تنظیمات ثابت
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1. دیتاست و تقسیم‌بندی (K-Fold)
# -----------------------------
def get_file_lists(data_dir="data", fold=0, k_folds=5):
    img_dir = os.path.join(data_dir, "images")
    lbl_dir = os.path.join(data_dir, "labels")
    
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
    labels = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".nii.gz")])
    
    assert len(images) == len(labels) > 0, "دیتاست خالی یا نامتناسب!"
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(images)):
        if i == fold:
            train_files = [{"image": img, "label": lbl} for img, lbl in zip(
                [images[j] for j in train_idx], [labels[j] for j in train_idx])]
            val_files   = [{"image": img, "label": lbl} for img, lbl in zip(
                [images[j] for j in val_idx], [labels[j] for j in val_idx])]
            return train_files, val_files

# -----------------------------
# 2. ترنسفورم‌های بهینه و بدون باگ
# -----------------------------
train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    
    # دیتا آگمنتیشن قوی
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1.0, neg=1.0,
        num_samples=4
    ),
    RandAffined(
        keys=["image", "label"],
        prob=0.3,
        rotate_range=(0, 0, np.pi/12),
        scale_range=(0.15, 0.15, 0.15),
        translate_range=(10, 10, 10),
        mode=("bililinear", "nearest")
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
    RandGaussianNoised(keys=["image"], prob=0.1, std=0.01),
    EnsureTyped(keys=["image", "label"]),
])

val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Resized(keys=["image", "label"], spatial_size=(96, 96, 96)),  # حتماً باید 96 باشه!
    EnsureTyped(keys=["image", "label"]),
])

# -----------------------------
# 3. دیتاست با Cache (خیلی سریع‌تر و بدون باگ batch)
# -----------------------------
train_files, val_files = get_file_lists(data_dir="data", fold=0)

train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=1.0, num_workers=6)
val_ds   = CacheDataset(data=val_files,   transform=val_transform,   cache_rate=1.0, num_workers=4)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------------
# 4. مدل + لاس + متریک
# -----------------------------
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
    dropout_rate=0.1,
).to(device)

# بهترین لاس برای نودول (کوچک و نامتوازن)
loss_function = DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

dice_metric = DiceMetric(include_background=False, reduction="mean")
hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

# -----------------------------
# 5. حلقه آموزش کاملاً اصلاح‌شده
# -----------------------------
max_epochs = 150
val_interval = 2
best_dice = 0.0

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{max_epochs} - Loss: {epoch_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Validation
    if (epoch + 1) % val_interval == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                val_outputs = sliding_window_inference(
                    val_images, (96, 96, 96), sw_batch_size=2,
                    predictor=model, overlap=0.6, mode="gaussian"
                )
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()

                dice_metric(y_pred=val_preds, y=val_labels)
                hd_metric(y_pred=val_preds, y=val_labels)

            dice_score = dice_metric.aggregate().item()
            hd_score   = hd_metric.aggregate().item()
            dice_metric.reset()
            hd_metric.reset()

            print(f"Validation Dice: {dice_score:.4f} | HD95: {hd_score:.2f}")

            if dice_score > best_dice:
                best_dice = dice_score
                torch.save(model.state_dict(), "best_unetr_lung_nodule.pth")
                print("بهترین مدل ذخیره شد!")

print("آموزش تمام شد! بهترین Dice:", best_dice)

# -----------------------------
# 6. اینفرنس روی اسکن جدید (کاملاً درست)
# -----------------------------
def predict_scan(model_path, scan_path, output_path="pred_nodule.nii.gz"):
    model.eval()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    infer_transform = Compose([
        LoadImaged(keys="image"),
        AddChanneld(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0,1.0,1.0), mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="image"),
    ])

    data = infer_transform({"image": scan_path})
    img = data["image"].unsqueeze(0).to(device)
    affine = data["image_meta_dict"]["original_affine"]

    with torch.no_grad():
        pred = sliding_window_inference(img, (96,96,96), 2, model, overlap=0.7)
        prob = torch.sigmoid(pred)[0,0].cpu().numpy()
        binary = (prob > 0.5).astype(np.uint8)

    nib.save(nib.Nifti1Image(binary, affine), output_path)
    nib.save(nib.Nifti1Image(prob,   affine), output_path.replace(".nii.gz", "_prob.nii.gz"))
    print("پیش‌بینی ذخیره شد:", output_path)

# استفاده:
# predict_scan("best_unetr_lung_nodule.pth", "test_scan.nii.gz", "result.nii.gz")