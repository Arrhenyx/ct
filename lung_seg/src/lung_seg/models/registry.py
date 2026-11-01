"""Model factory — pick implementation by name in config."""
from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import SwinUNETR, UNETR, UNet

from ..config import Config


def build_model(cfg: Config) -> nn.Module:
    name = cfg.model_name.lower()

    if name == "unetr":
        return UNETR(
            in_channels=1,
            out_channels=1,
            img_size=cfg.roi_size,
            feature_size=cfg.feature_size,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_heads=cfg.num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=cfg.dropout,
        )

    if name == "swinunetr":
        return SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=cfg.swin_feature_size,
            use_checkpoint=False,
            spatial_dims=3,
        )

    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )

    raise ValueError(
        f"Unknown model_name={cfg.model_name!r}. "
        f"Choose from: unetr, swinunetr, unet."
    )