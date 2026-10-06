"""Sliding-window validation on full volumes."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from ..config import Config
from ..inference.sliding_window import sliding_window_predict


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: Config,
    dice_metric: DiceMetric,
    hd_metric: HausdorffDistanceMetric,
) -> Tuple[float, float]:
    model.eval()
    dice_scores, hd_scores = [], []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        probs = sliding_window_predict(model, imgs, cfg, use_tta=False)
        preds = (probs > 0.5).float()

        dice_metric(y_pred=preds, y=lbls)
        d = dice_metric.aggregate().item()
        dice_metric.reset()
        if np.isfinite(d):
            dice_scores.append(d)

        if preds.sum() > 0 and lbls.sum() > 0:
            hd_metric(y_pred=preds, y=lbls)
            h = hd_metric.aggregate().item()
            hd_metric.reset()
            if np.isfinite(h):
                hd_scores.append(h)

    model.train()
    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_hd = float(np.mean(hd_scores)) if hd_scores else float("nan")
    return avg_dice, avg_hd