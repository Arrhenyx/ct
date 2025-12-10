# =========================
#  3D Lung Nodule Segmentation - UNETR (MONAI) - نسخه پایدار و بهینه
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
    RandFlipd, RandRotate90d, SpatialPadd
)
from monai.networks.nets import UNETR
from monai.losses import DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate
from monai.utils import set_determinism
from sklearn.model_selection import KFold
import logging
from torch.cuda.amp import autocast, GradScaler
import json
from datetime import datetime
import cc3d
from scipy import ndimage

# =========================
#  کلاس Early Stopping
# =========================
class EarlyStopping:
    def __init__(self, patience=15, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

# =========================
#  تنظیمات اولیه
# =========================
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# تنظیمات logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# 1. دیتاست و تقسیم‌بندی (K-Fold)
# =========================
def get_file_lists(data_dir="data", fold=0, k_folds=5):
    img_dir = os.path.join(data_dir, "imagesTr")
    lbl_dir = os.path.join(data_dir, "labelsTr")
    
    if not os.path.exists(img_dir):
        img_dir = os.path.join(data_dir, "images")
    if not os.path.exists(lbl_dir):
        lbl_dir = os.path.join(data_dir, "labels")
    
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
    labels = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".nii.gz")])
    
    assert len(images) == len(labels) > 0, f"دیتاست خالی یا نامتناسب! تعداد images: {len(images)}, labels: {len(labels)}"
    assert all([os.path.exists(img) for img in images]), "برخی فایل‌های image وجود ندارند"
    assert all([os.path.exists(lbl) for lbl in labels]), "برخی فایل‌های label وجود ندارند"
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_indices = list(range(len(images)))
    
    for i, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        if i == fold:
            train_files = [{"image": images[j], "label": labels[j]} for j in train_idx]
            val_files = [{"image": images[j], "label": labels[j]} for j in val_idx]
            
            print(f"Fold {fold}: {len(train_files)} training, {len(val_files)} validation samples")
            return train_files, val_files
    
    return [], []

# =========================
# 2. ترنسفورم‌های بهینه
# =========================
def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-1000, 
            a_max=400, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1.0, 
            neg=1.0,
            num_samples=4
        ),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.15, 0.15, 0.15),
            translate_range=(8, 8, 8),
            mode=("bilinear", "nearest"),
            padding_mode="border"
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        EnsureTyped(keys=["image", "label"], device=device),
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-1000, 
            a_max=400, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device),
    ])

def get_infer_transforms():
    return Compose([
        LoadImaged(keys="image", ensure_channel_first=False),
        AddChanneld(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys="image", 
            a_min=-1000, 
            a_max=400, 
            b_min=0.0, 
            b_max=1.0, 
            clip=True
        ),
        EnsureTyped(keys="image", device=device),
    ])

# =========================
# 3. توابع utility
# =========================
def save_checkpoint(epoch, model, optimizer, scheduler, best_dice, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_dice': best_dice,
    }, filename)
    logger.info(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded: {filename}, epoch: {checkpoint['epoch']}, best_dice: {checkpoint['best_dice']:.4f}")
    return checkpoint['epoch'], checkpoint['best_dice']

def postprocess_prediction(prediction, min_size=27):
    prediction = prediction.astype(np.uint8)
    
    labels_out = cc3d.connected_components(prediction)
    
    for label in range(1, labels_out.max() + 1):
        if np.sum(labels_out == label) < min_size:
            prediction[labels_out == label] = 0
    
    prediction = ndimage.binary_closing(prediction, structure=np.ones((2, 2, 2)))
    prediction = ndimage.binary_opening(prediction, structure=np.ones((1, 1, 1)))
    
    return prediction

# =========================
# 4. تعریف مدل و آموزش
# =========================
def train_model(fold=0, data_dir="data", k_folds=5, resume_from_checkpoint=None):
    train_files, val_files = get_file_lists(data_dir=data_dir, fold=fold, k_folds=k_folds)
    
    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError("No data files found. Check your data directory structure.")
    
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=1.0, 
        num_workers=6
    )
    val_ds = CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=1.0, 
        num_workers=4
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=2, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=list_data_collate
    )
    
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
    
    loss_function = DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=20, delta=0.001, verbose=True)
    
    start_epoch = 0
    best_dice = 0.0
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, best_dice = load_checkpoint(
            resume_from_checkpoint, 
            model, 
            optimizer, 
            scheduler
        )
        start_epoch += 1
    
    max_epochs = 200
    val_interval = 2
    accumulation_steps = 2
    
    history = {
        'train_loss': [],
        'val_dice': [],
        'val_hd95': [],
        'learning_rate': []
    }
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = loss_function(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
        
        scheduler.step()
        epoch_loss /= len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Loss: {epoch_loss:.4f} - LR: {current_lr:.6f}")
        history['train_loss'].append(epoch_loss)
        history['learning_rate'].append(current_lr)
        
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            model.eval()
            dice_scores = []
            hd_scores = []
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    
                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=(96, 96, 96),
                        sw_batch_size=2,
                        predictor=model,
                        overlap=0.6,
                        mode="gaussian",
                        progress=False
                    )
                    
                    val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                    
                    dice_metric(y_pred=val_preds, y=val_labels)
                    hd_metric(y_pred=val_preds, y=val_labels)
                    
                    dice_score_batch = dice_metric.aggregate().item()
                    hd_score_batch = hd_metric.aggregate().item()
                    
                    dice_scores.append(dice_score_batch)
                    hd_scores.append(hd_score_batch)
                    
                    dice_metric.reset()
                    hd_metric.reset()
            
            avg_dice = np.mean(dice_scores)
            avg_hd = np.mean(hd_scores)
            
            logger.info(f"Validation - Dice: {avg_dice:.4f} | HD95: {avg_hd:.2f}")
            history['val_dice'].append(avg_dice)
            history['val_hd95'].append(avg_hd)
            
            if avg_dice > best_dice:
                best_dice = avg_dice
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_dice=best_dice,
                    filename=f"best_unetr_fold{fold}.pth"
                )
                
                with open(f"training_history_fold{fold}.json", 'w') as f:
                    json.dump(history, f, indent=2)
            
            early_stopping(-avg_dice)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered!")
                break
    
    logger.info(f"Training completed for fold {fold}. Best Dice: {best_dice:.4f}")
    
    return model, history, best_dice

# =========================
# 5. اینفرنس و پیش‌بینی
# =========================
def predict_scan(model_path, scan_path, output_dir="predictions", post_process=True):
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    infer_transform = get_infer_transforms()
    
    data = infer_transform({"image": scan_path})
    img = data["image"].unsqueeze(0).to(device)
    
    affine = None
    if "affine" in data:
        affine = data["affine"]
    elif "image_meta_dict" in data and "original_affine" in data["image_meta_dict"]:
        affine = data["image_meta_dict"]["original_affine"]
    else:
        nifti_img = nib.load(scan_path)
        affine = nifti_img.affine
    
    original_shape = data["image"].shape[1:]
    
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=img,
            roi_size=(96, 96, 96),
            sw_batch_size=2,
            predictor=model,
            overlap=0.7,
            mode="gaussian",
            progress=True
        )
        
        prob = torch.sigmoid(pred)[0, 0].cpu().numpy()
        
        from monai.transforms import Resize
        resize_transform = Resize(spatial_size=original_shape, mode="bilinear")
        prob_resized = resize_transform(prob[None, None, ...])[0, 0].numpy()
        
        binary = (prob_resized > 0.5).astype(np.uint8)
        
        if post_process:
            binary = postprocess_prediction(binary, min_size=27)
    
    base_name = os.path.splitext(os.path.basename(scan_path))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    
    binary_output_path = os.path.join(output_dir, f"{base_name}_segmentation.nii.gz")
    prob_output_path = os.path.join(output_dir, f"{base_name}_probability.nii.gz")
    
    nib.save(nib.Nifti1Image(binary, affine), binary_output_path)
    nib.save(nib.Nifti1Image(prob_resized, affine), prob_output_path)
    
    logger.info(f"Predictions saved:")
    logger.info(f"  - Segmentation: {binary_output_path}")
    logger.info(f"  - Probability map: {prob_output_path}")
    
    return binary_output_path, prob_output_path

# =========================
# 6. K-Fold Cross Validation
# =========================
def run_kfold_cross_validation(data_dir="data", k_folds=5):
    all_folds_results = []
    
    for fold in range(k_folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting training for Fold {fold}/{k_folds-1}")
        logger.info(f"{'='*50}")
        
        try:
            model, history, best_dice = train_model(
                fold=fold,
                data_dir=data_dir,
                k_folds=k_folds,
                resume_from_checkpoint=None
            )
            
            all_folds_results.append({
                'fold': fold,
                'best_dice': best_dice,
                'history': history
            })
            
        except Exception as e:
            logger.error(f"Error training fold {fold}: {str(e)}")
            continue
    
    if all_folds_results:
        avg_dice = np.mean([r['best_dice'] for r in all_folds_results])
        std_dice = np.std([r['best_dice'] for r in all_folds_results])
        
        logger.info(f"\n{'='*50}")
        logger.info(f"K-Fold Cross Validation Results ({k_folds} folds)")
        logger.info(f"{'='*50}")
        
        for result in all_folds_results:
            logger.info(f"Fold {result['fold']}: Best Dice = {result['best_dice']:.4f}")
        
        logger.info(f"\nAverage Dice: {avg_dice:.4f} ± {std_dice:.4f}")
        
        with open("kfold_results.json", 'w') as f:
            json.dump(all_folds_results, f, indent=2)
        
        return all_folds_results, avg_dice, std_dice
    
    return None, 0, 0

# =========================
# 7. تابع اصلی
# =========================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Lung Nodule Segmentation with UNETR')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'predict', 'kfold', 'infer'],
                       help='Execution mode')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, default='best_unetr_fold0.pth',
                       help='Path to model checkpoint for inference')
    parser.add_argument('--scan_path', type=str, default=None,
                       help='Path to scan for inference')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory for predictions')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number for k-fold cross validation')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross validation')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, history, best_dice = train_model(
            fold=args.fold,
            data_dir=args.data_dir,
            k_folds=args.k_folds,
            resume_from_checkpoint=args.resume
        )
        
    elif args.mode == 'kfold':
        results, avg_dice, std_dice = run_kfold_cross_validation(
            data_dir=args.data_dir,
            k_folds=args.k_folds
        )
        
    elif args.mode in ['predict', 'infer']:
        if not args.scan_path:
            raise ValueError("Scan path must be provided for inference mode")
        
        if not os.path.exists(args.scan_path):
            raise FileNotFoundError(f"Scan file not found: {args.scan_path}")
        
        binary_path, prob_path = predict_scan(
            model_path=args.model_path,
            scan_path=args.scan_path,
            output_dir=args.output_dir,
            post_process=True
        )
        
        print(f"Segmentation saved to: {binary_path}")
        print(f"Probability map saved to: {prob_path}")
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
