"""Preprocessing + augmentation pipelines.

Deterministic preprocessing is shared across train/val/infer so the model
sees the same intensity/orientation/spacing distribution at all stages.
"""
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
)

from ..config import Config


def _preprocessing(cfg: Config):
    return [
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=cfg.pixdim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg.hu_min,
            a_max=cfg.hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]


def get_train_transforms(cfg: Config) -> Compose:
    return Compose(
        _preprocessing(cfg)
        + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.roi_size,
                pos=cfg.pos_ratio,
                neg=cfg.neg_ratio,
                num_samples=cfg.num_samples,
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.15, 0.15, 0.15),
                translate_range=(8, 8, 8),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
            RandRotate90d(
                keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(cfg: Config) -> Compose:
    return Compose(_preprocessing(cfg) + [EnsureTyped(keys=["image", "label"])])


def get_infer_transforms(cfg: Config) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=cfg.pixdim, mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.hu_min,
                a_max=cfg.hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"]),
        ]
    )