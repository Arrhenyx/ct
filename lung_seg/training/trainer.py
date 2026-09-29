"""Training loop with AMP, gradient accumulation, early stopping."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from ..config import Config
from ..data.dataloaders import build_train_loader, build_val_loader
from ..data.splits import get_file_lists
from ..models import build_model
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logging import get_logger
from .early_stopping import EarlyStopping
from .validation import validate


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_optim_sched(model, cfg: Config):
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=2, eta_min=1e-6,
    )
    return opt, sch


def train_model(cfg: Config, fold: Optional[int] = None,
                resume_from: Optional[str] = None) -> Tuple[float, dict]:
    device = _device()
    logger = get_logger()
    logger.info(f"Device: {device}")

    fold = cfg.fold if fold is None else fold
    train_files, val_files = get_file_lists(cfg, fold=fold)

    train_loader = build_train_loader(cfg, train_files)
    val_loader = build_val_loader(cfg, val_files)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.2f}M")

    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3)
    optimizer, scheduler = _make_optim_sched(model, cfg)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="mean",
    )

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (lambda: torch.amp.autocast("cuda", enabled=use_amp)) \
        if use_amp else nullcontext

    early_stop = EarlyStopping(
        patience=cfg.patience, delta=cfg.early_stop_delta, mode="max",
    )

    start_epoch, best_dice = 0, 0.0
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_dice = load_checkpoint(
            resume_from, model, optimizer, scheduler, map_location=device,
        )
        start_epoch += 1

    history = {"train_loss": [], "val_dice": [], "val_hd95": [], "lr": []}
    ckpt_dir = Path(cfg.ckpt_dir) / f"fold{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accum = max(1, cfg.accum_steps)

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        n_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True)

            with autocast():
                logits = model(imgs)
                loss = loss_fn(logits, lbls)

            is_last = (step + 1) == n_steps
            if not is_last:
                scale = 1.0 / accum
            else:
                rem = (n_steps - step) % accum
                scale = 1.0 / (rem if rem > 0 else accum)

            scaler.scale(loss * scale).backward()

            if ((step + 1) % accum == 0) or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()

        scheduler.step()
        epoch_loss = running / max(1, n_steps)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch + 1}/{cfg.max_epochs} | "
            f"loss={epoch_loss:.4f} | lr={lr:.2e}"
        )
        history["train_loss"].append(epoch_loss)
        history["lr"].append(lr)

        if (epoch + 1) % cfg.val_interval == 0 or epoch == 0:
            avg_dice, avg_hd = validate(
                model, val_loader, device, cfg, dice_metric, hd_metric,
            )
            hd_str = f"{avg_hd:.2f}" if np.isfinite(avg_hd) else "n/a"
            logger.info(f"  → val Dice={avg_dice:.4f} | HD95={hd_str}")
            history["val_dice"].append(avg_dice)
            history["val_hd95"].append(
                avg_hd if np.isfinite(avg_hd) else None
            )

            save_checkpoint(
                ckpt_dir / "last.pth", epoch, model, optimizer,
                scheduler, best_dice, cfg.to_dict(),
            )

            if avg_dice > best_dice:
                best_dice = avg_dice
                save_checkpoint(
                    ckpt_dir / "best.pth", epoch, model, optimizer,
                    scheduler, best_dice, cfg.to_dict(),
                )

            with open(ckpt_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            early_stop(avg_dice)
            if early_stop.early_stop:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Fold {fold} done. Best Dice: {best_dice:.4f}")
    return best_dice, history
