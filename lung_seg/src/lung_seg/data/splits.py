"""Dataset discovery and K-fold splitting."""
from __future__ import annotations

import os
from typing import List, Tuple

from sklearn.model_selection import KFold

from ..config import Config
from ..utils.logging import get_logger

_NIFTI_EXTS = (".nii", ".nii.gz")


def _resolve_dirs(data_dir: str) -> Tuple[str, str]:
    img_dir = os.path.join(data_dir, "imagesTr")
    lbl_dir = os.path.join(data_dir, "labelsTr")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(lbl_dir):
        lbl_dir = os.path.join(data_dir, "labels")
    return img_dir, lbl_dir


def _list_niftis(folder: str) -> List[str]:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(_NIFTI_EXTS)
    )


def get_file_lists(cfg: Config, fold: int | None = None) -> Tuple[list, list]:
    """Return (train_files, val_files) for the given fold."""
    fold = cfg.fold if fold is None else fold
    img_dir, lbl_dir = _resolve_dirs(cfg.data_dir)
    images = _list_niftis(img_dir)
    labels = _list_niftis(lbl_dir)

    assert len(images) == len(labels) > 0, (
        f"Empty/mismatched dataset: {len(images)} images vs {len(labels)} labels"
    )

    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    for i, (train_idx, val_idx) in enumerate(kf.split(images)):
        if i == fold:
            train_files = [{"image": images[j], "label": labels[j]} for j in train_idx]
            val_files = [{"image": images[j], "label": labels[j]} for j in val_idx]
            get_logger().info(
                f"Fold {fold}: {len(train_files)} train / {len(val_files)} val"
            )
            return train_files, val_files

    raise ValueError(f"Fold {fold} not found (k_folds={cfg.k_folds})")