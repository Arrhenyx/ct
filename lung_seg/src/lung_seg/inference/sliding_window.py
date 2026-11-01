"""Sliding-window inference with optional flip TTA."""
from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference

from ..config import Config

_TTA_DIMS = ([2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4])


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    cfg: Config,
    use_tta: bool = False,
) -> torch.Tensor:
    """
    image: (N, 1, D, H, W) float tensor on device.
    returns: probabilities (N, 1, D, H, W) on the same device.
    """

    def _fwd(x):
        return model(x)

    probs = torch.sigmoid(
        sliding_window_inference(
            inputs=image,
            roi_size=cfg.roi_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=_fwd,
            overlap=cfg.sw_overlap,
            mode=cfg.sw_mode,
            progress=False,
        )
    )

    if use_tta:
        for dims in _TTA_DIMS:
            inv = torch.flip(image, dims=dims)
            l = sliding_window_inference(
                inputs=inv,
                roi_size=cfg.roi_size,
                sw_batch_size=cfg.sw_batch_size,
                predictor=_fwd,
                overlap=cfg.sw_overlap,
                mode=cfg.sw_mode,
                progress=False,
            )
            probs = probs + torch.flip(torch.sigmoid(l), dims=dims)
        probs = probs / (1 + len(_TTA_DIMS))

    return probs