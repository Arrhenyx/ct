"""Full-volume sliding-window validation."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

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
    case_names: Optional[Sequence[str]] = None,
    case_csv: Optional[str | Path] = None,
) -> Tuple[float, float]:
    model.eval()
    dice_scores: list[float] = []
    hd_scores: list[float] = []
    case_rows: list[dict] = []

    for idx, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        probs = sliding_window_predict(model, imgs, cfg, use_tta=False)
        preds = (probs > 0.5).float()

        dice_metric(y_pred=preds, y=lbls)
        d = dice_metric.aggregate().item()
        dice_metric.reset()

        hd = math.nan
        if preds.sum() > 0 and lbls.sum() > 0:
            hd_metric(y_pred=preds, y=lbls)
            hd = hd_metric.aggregate().item()
            hd_metric.reset()

        if np.isfinite(d):
            dice_scores.append(d)
        if np.isfinite(hd):
            hd_scores.append(hd)

        case = case_names[idx] if case_names else f"case_{idx}"
        case_rows.append({"case": case, "dice": d, "hd95": hd})

    model.train()
    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_hd = float(np.mean(hd_scores)) if hd_scores else math.nan

    if case_csv is not None and case_rows:
        case_csv = Path(case_csv)
        case_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(case_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "dice", "hd95"])
            writer.writeheader()
            writer.writerows(case_rows)

    return avg_dice, avg_hd