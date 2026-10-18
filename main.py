//lets add some changes here 
import os
import json
import random
import logging
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
import cc3d
from scipy import ndimage
from sklearn.model_selection import KFold

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandAffined, RandGaussianNoised, RandFlipd, RandRotate90d,
    EnsureTyped, Resize,
)
from monai.networks.nets import UNETR
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate
from monai.utils import set_determinism


# =============================================================================
# Config
# =============================================================================
CFG = {
    "data_dir": "data",
    "k_folds": 5,
    "fold": 0,
    "seed": 42,

    # Preprocessing
    "pixdim": (1.0, 1.0, 1.0),
    "hu_min": -1000.0,
    "hu_max": 400.0,

    # Patch / ROI
    "roi_size": (96, 96, 96),
    "num_samples": 4,          # crops per training volume
    "pos_ratio": 1.0,
    "neg_ratio": 1.0,

    # Model
    "feature_size": 16,
    "hidden_size": 768,
    "mlp_dim": 3072,
    "num_heads": 12,
    "dropout": 0.1,

    # Optimization
    "batch_size": 2,
    "val_batch_size": 1,
    "accum_steps": 2,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "max_epochs": 200,
    "val_interval": 2,
    "patience": 20,
    "early_stop_delta": 0.001,
    "grad_clip": 1.0,
    "amp": True,

    # Inference
    "sw_overlap": 0.6,
    "sw_batch_size": 2,
    "sw_mode": "gaussian",
    "min_cc_size": 27,        # minimum nodule size in voxels after CC
    "tta": False,             # enable flip test-time augmentation

    # I/O
    "output_dir": "predictions",
    "ckpt_dir": "checkpoints",
}


# =============================================================================
# Logging
# =============================================================================
def setup_logger(tag: str = "run") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    fname = f"logs/{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(fname), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger("unetr")


logger = setup_logger()


# =============================================================================
# Reproducibility
# =============================================================================
def seed_everything(seed: int) -> None:
    set_determinism(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Early stopping (maximizes monitored metric)
# =============================================================================
class EarlyStopping:
    """
    Monitors a score that should INCREASE (e.g. Dice).
    Use `mode='min'` to monitor a loss instead.
    """
    def __init__(self, patience: int = 20, delta: float = 1e-3,
                 mode: str = "max", verbose: bool = True):
        assert mode in ("max", "min")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.early_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return score > self.best + self.delta
        return score < self.best - self.delta

    def __call__(self, score: float) -> None:
        if self._is_improvement(score):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"  EarlyStopping {self.counter}/{self.patience} "
                            f"(best={self.best:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True


# =============================================================================
# Data splitting
# =============================================================================
def resolve_dirs(data_dir: str):
    img_dir = os.path.join(data_dir, "imagesTr")
    lbl_dir = os.path.join(data_dir, "labelsTr")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(lbl_dir):
        lbl_dir = os.path.join(data_dir, "labels")
    return img_dir, lbl_dir


def get_file_lists(data_dir: str, fold: int, k_folds: int):
    img_dir, lbl_dir = resolve_dirs(data_dir)
    images = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.endswith((".nii", ".nii.gz")))
    labels = sorted(os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
                    if f.endswith((".nii", ".nii.gz")))
    assert len(images) == len(labels) > 0, \
        f"Empty/mismatched dataset: {len(images)} images vs {len(labels)} labels"

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=CFG["seed"])
    for i, (train_idx, val_idx) in enumerate(kf.split(images)):
        if i == fold:
            train_files = [{"image": images[j], "label": labels[j]} for j in train_idx]
            val_files = [{"image": images[j], "label": labels[j]} for j in val_idx]
            logger.info(f"Fold {fold}: {len(train_files)} train / "
                        f"{len(val_files)} val")
            return train_files, val_files
    raise ValueError(f"Fold {fold} not found for k_folds={k_folds}")


# =============================================================================
# Transforms — consistent preprocessing shared by train/val/infer
# =============================================================================
def _preprocessing():
    """Deterministic, shared preprocessing."""
    return [
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"],
                 pixdim=CFG["pixdim"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=CFG["hu_min"], a_max=CFG["hu_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]


def get_train_transforms():
    return Compose(_preprocessing() + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=CFG["roi_size"],
            pos=CFG["pos_ratio"], neg=CFG["neg_ratio"],
            num_samples=CFG["num_samples"],
        ),
        RandAffined(
            keys=["image", "label"], prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.15, 0.15, 0.15),
            translate_range=(8, 8, 8),
            mode=("bilinear", "nearest"), padding_mode="border",
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3,
                      spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        EnsureTyped(keys=["image", "label"]),  # keep on CPU; .to(device) in loop
    ])


def get_val_transforms():
    """Deterministic — full-volume; used with sliding window in val loop."""
    return Compose(_preprocessing() + [
        EnsureTyped(keys=["image", "label"]),
    ])


def get_infer_transforms():
    """Image-only version, identical to val preprocessing for the image branch."""
    return Compose([
        LoadImaged(keys=["image"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=CFG["pixdim"], mode="bilinear"),
        ScaleIntensityRanged(keys=["image"],
                             a_min=CFG["hu_min"], a_max=CFG["hu_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])


# =============================================================================
# Model / loss / optim helpers
# =============================================================================
def build_model():
    return UNETR(
        in_channels=1,
        out_channels=1,
        img_size=CFG["roi_size"],
        feature_size=CFG["feature_size"],
        hidden_size=CFG["hidden_size"],
        mlp_dim=CFG["mlp_dim"],
        num_heads=CFG["num_heads"],
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=CFG["dropout"],
    )


def build_optimizer_scheduler(model):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=CFG["lr"], weight_decay=CFG["weight_decay"],
                            betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=2, eta_min=1e-6)
    return opt, sch


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_dice):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice": best_dice,
        "config": CFG,
    }, path)
    logger.info(f"  Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info(f"Loaded checkpoint {path} "
                f"(epoch {ckpt.get('epoch', '?')}, dice={ckpt.get('best_dice', 0):.4f})")
    return ckpt.get("epoch", 0), ckpt.get("best_dice", 0.0)


# =============================================================================
# Postprocessing
# =============================================================================
def postprocess_prediction(pred: np.ndarray, min_size: int = 27) -> np.ndarray:
    pred = pred.astype(np.uint8)
    cc = cc3d.connected_components(pred, connectivity=26)
    if cc.max() > 0:
        sizes = np.bincount(cc.ravel())
        small = np.where(sizes < min_size)[0]
        small = small[small != 0]
        if small.size:
            mask = np.isin(cc, small)
            pred[mask] = 0
    pred = ndimage.binary_closing(pred, structure=np.ones((2, 2, 2))).astype(np.uint8)
    pred = ndimage.binary_opening(pred, structure=np.ones((1, 1, 1))).astype(np.uint8)
    return pred


# =============================================================================
# Sliding-window inference (shared by val + predict)
# =============================================================================
@torch.no_grad()
def sliding_window_predict(model, image, use_tta: bool = False):
    """
    image: (1, 1, D, H, W) float tensor on device.
    returns: probabilities (1, 1, D, H, W) on device.
    """
    def _fwd(x):
        return model(x)

    logits = sliding_window_inference(
        inputs=image, roi_size=CFG["roi_size"],
        sw_batch_size=CFG["sw_batch_size"],
        predictor=_fwd, overlap=CFG["sw_overlap"],
        mode=CFG["sw_mode"], progress=False,
    )
    probs = torch.sigmoid(logits)

    if use_tta:
        # 3-axis flip TTA; average probabilities
        for dims in ([2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]):
            inv = torch.flip(image, dims=dims)
            l = sliding_window_inference(
                inputs=inv, roi_size=CFG["roi_size"],
                sw_batch_size=CFG["sw_batch_size"],
                predictor=_fwd, overlap=CFG["sw_overlap"],
                mode=CFG["sw_mode"], progress=False,
            )
            p = torch.sigmoid(l)
            probs = probs + torch.flip(p, dims=dims)
        probs = probs / 8.0

    return probs


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def validate(model, val_loader, dice_metric, hd_metric):
    model.eval()
    dice_scores, hd_scores = [], []
    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        probs = sliding_window_predict(model, imgs, use_tta=False)
        preds = (probs > 0.5).float()

        dice_metric(y_pred=preds, y=y=lbls)
        d = dice_metric.aggregate().item()
        dice_metric.reset()
        if not (np.isnan(d) or np.isinf(d)):
            dice_scores.append(d)

        # HD95 only meaningful if both pred and GT are non-empty
        if preds.sum() > 0 and lbls.sum() > 0:
            hd_metric(y_pred=preds, y=lbls)
            h = hd_metric.aggregate().item()
            hd_metric.reset()
            if not (np.isnan(h) or np.isinf(h)):
                hd_scores.append(h)

    model.train()
    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_hd = float(np.mean(hd_scores)) if hd_scores else float("nan")
    return avg_dice, avg_hd


# =============================================================================
# Training
# =============================================================================
def train_model(fold: int = 0, resume_from_checkpoint: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_files, val_files = get_file_lists(
        CFG["data_dir"], fold=fold, k_folds=CFG["k_folds"])

    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_rate=1.0, num_workers=min(6, os.cpu_count() or 4),
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=1.0, num_workers=min(4, os.cpu_count() or 2),
    )

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=list_data_collate, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["val_batch_size"], shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=list_data_collate, persistent_workers=True,
    )

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.2f}M")

    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3)
    optimizer, scheduler = build_optimizer_scheduler(model)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="mean")

    use_amp = CFG["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (lambda: torch.amp.autocast("cuda", enabled=use_amp)) if use_amp \
               else nullcontext

    early_stop = EarlyStopping(
        patience=CFG["patience"], delta=CFG["early_stop_delta"], mode="max")

    start_epoch, best_dice = 0, 0.0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, best_dice = load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler)
        start_epoch += 1

    history = {"train_loss": [], "val_dice": [], "val_hd95": [], "lr": []}
    ckpt_dir = os.path.join(CFG["ckpt_dir"], f"fold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)

    accum = max(1, CFG["accum_steps"])

    for epoch in range(start_epoch, CFG["max_epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True)

            with autocast():
                logits = model(imgs)
                loss = loss_fn(logits, lbls)

            # Correct accumulation: scale by (1/accum), but the final partial
            # window is scaled by (steps_remaining / accum) so total grads match
            n_steps = len(train_loader)
            is_last = (step + 1) == n_steps
            if not is_last:
                scale = 1.0 / accum
            else:
                # remaining steps in this accumulation window
                rem = (n_steps - step) % accum
                scale = 1.0 / (rem if rem > 0 else accum)

            loss_scaled = loss * scale
            scaler.scale(loss_scaled).backward()

            if ((step + 1) % accum == 0) or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        scheduler.step()  # per-epoch
        epoch_loss = running_loss / max(1, len(train_loader))
        lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch+1}/{CFG['max_epochs']} | "
                    f"loss={epoch_loss:.4f} | lr={lr:.2e}")
        history["train_loss"].append(epoch_loss)
        history["lr"].append(lr)

        if (epoch + 1) % CFG["val_interval"] == 0 or epoch == 0:
            avg_dice, avg_hd = validate(model, val_loader, dice_metric, hd_metric)
            hd_str = f"{avg_hd:.2f}" if not np.isnan(avg_hd) else "n/a"
            logger.info(f"  → val Dice={avg_dice:.4f} | HD95={hd_str}")
            history["val_dice"].append(avg_dice)
            history["val_hd95"].append(avg_hd if not np.isnan(avg_hd) else None)

            # Save "last"
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"),
                            epoch, model, optimizer, scheduler, best_dice)

            if avg_dice > best_dice:
                best_dice = avg_dice
                save_checkpoint(os.path.join(ckpt_dir, "best.pth"),
                                epoch, model, optimizer, scheduler, best_dice)

            with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

            early_stop(avg_dice)
            if early_stop.early_stop:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Fold {fold} done. Best Dice: {best_dice:.4f}")
    return best_dice, history


# =============================================================================
# Inference on a single scan (full-volume, sliding window, optional TTA)
# =============================================================================
def predict_scan(model_path: str, scan_path: str,
                 output_dir: str = None, post_process: bool = True,
                 tta: bool = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = output_dir or CFG["output_dir"]
    tta = CFG["tta"] if tta is None else tta
    os.makedirs(output_dir, exist_ok=True)

    model = build_model().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Reference affine & shape from the original scan
    nii = nib.load(scan_path)
    original_shape = nii.shape
    affine = nii.affine

    data = get_infer_transforms()({"image": scan_path})
    img = data["image"].unsqueeze(0).to(device)

    probs = sliding_window_predict(model, img, use_tta=tta)  # (1,1,D,H,W)
    prob_np = probs[0, 0].cpu().numpy()

    # Resize probability map back to the *original* volume geometry
    resizer = Resize(spatial_size=original_shape, mode="trilinear")
    prob_np = resizer(prob_np[None, None, ...])[0, 0].numpy()

    binary = (prob_np > 0.5).astype(np.uint8)
    if post_process:
        binary = postprocess_prediction(binary, min_size=CFG["min_cc_size"])

    base = os.path.basename(scan_path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    bin_path = os.path.join(output_dir, f"{base}_seg.nii.gz")
    prob_path = os.path.join(output_dir, f"{base}_prob.nii.gz")
    nib.save(nib.Nifti1Image(binary, affine), bin_path)
    nib.save(nib.Nifti1Image(prob_np.astype(np.float32), affine), prob_path)

    logger.info(f"Saved: {bin_path}")
    logger.info(f"Saved: {prob_path}")
    return bin_path, prob_path


# =============================================================================
# K-fold CV
# =============================================================================
def run_kfold():
    results = []
    for fold in range(CFG["k_folds"]):
        logger.info(f"\n{'=' * 60}\nFold {fold}/{CFG['k_folds'] - 1}\n{'=' * 60}")
        try:
            best_dice, hist = train_model(fold=fold)
            results.append({"fold": fold, "best_dice": best_dice})
        except Exception as e:
            logger.exception(f"Fold {fold} failed: {e}")

    if results:
        dices = [r["best_dice"] for r in results]
        logger.info(f"\nK-Fold: mean Dice = {np.mean(dices):.4f} "
                    f"± {np.std(dices):.4f}")
        with open("kfold_results.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser("UNETR lung-nodule segmentation")
    parser.add_argument("--mode", choices=["train", "predict", "kfold"],
                        default="train")
    parser.add_argument("--data_dir", default=CFG["data_dir"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--scan_path", default=None)
    parser.add_argument("--output_dir", default=CFG["output_dir"])
    parser.add_argument("--fold", type=int, default=CFG["fold"])
    parser.add_argument("--k_folds", type=int, default=CFG["k_folds"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tta", action="store_true",
                        help="enable flip TTA at inference")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    CFG["data_dir"] = args.data_dir
    CFG["fold"] = args.fold
    CFG["k_folds"] = args.k_folds
    CFG["output_dir"] = args.output_dir
    CFG["amp"] = not args.no_amp

    seed_everything(CFG["seed"])

    if args.mode == "train":
        train_model(fold=args.fold, resume_from_checkpoint=args.resume)
    elif args.mode == "kfold":
        run_kfold()
    elif args.mode == "predict":
        if not args.scan_path or not args.model_path:
            raise SystemExit("--scan_path and --model_path required for predict")
        predict_scan(args.model_path, args.scan_path,
                     output_dir=args.output_dir, tta=args.tta)


if __name__ == "__main__":
    main()
