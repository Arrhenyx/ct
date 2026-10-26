import os

from monai.data import CacheDataset, DataLoader, list_data_collate

from ..config import Config
from .transforms import get_train_transforms, get_val_transforms


def _n_workers(default: int = 4) -> int:
    return min(default, os.cpu_count() or 2)


def build_train_loader(cfg: Config, train_files: list) -> DataLoader:
    ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(cfg),
        cache_rate=1.0,
        num_workers=_n_workers(6),
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=_n_workers(4),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=list_data_collate,
    )


def build_val_loader(cfg: Config, val_files: list) -> DataLoader:
    ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(cfg),
        cache_rate=1.0,
        num_workers=_n_workers(4),
    )
    return DataLoader(
        ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=_n_workers(2),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=list_data_collate,
    )
