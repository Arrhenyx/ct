"""Full-volume prediction on a single NIfTI scan."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Resize

from ..config import Config
from ..data.transforms import get_infer_transforms
from ..models import build_model
from ..utils.checkpoint import load_checkpoint
from ..utils.logging import get_logger
from ..utils.paths import strip_nifti_ext
from .postprocess import postprocess_prediction
from .sliding_window import sliding_window_predict


def predict_scan(
    cfg: Config,
    model_path: str,
    scan_path: str,
    output_dir: Optional[str] = None,
    post_process: bool = True,
    tta: Optional[bool] = None,
) -> Tuple[str, str]:
    logger = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = output_dir or cfg.output_dir
    tta = cfg.tta if tta is None else tta
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    load_checkpoint(model_path, model, map_location=device)
    model.eval()

    # Preserve original geometry for output files.
    nii = nib.load(scan_path)
    original_shape = nii.shape
    affine = nii.affine

    data = get_infer_transforms(cfg)({"image": scan_path})
    img = data["image"].unsqueeze(0).to(device)

    probs = sliding_window_predict(model, img, cfg, use_tta=tta)
    prob_np = probs[0, 0].cpu().numpy()

    resizer = Resize(spatial_size=original_shape, mode="trilinear")
    prob_np = resizer(prob_np[None, None, ...])[0, 0].numpy().astype(np.float32)

    binary = (prob_np > 0.5).astype(np.uint8)
    if post_process:
        binary = postprocess_prediction(binary, min_size=cfg.min_cc_size)

    base = strip_nifti_ext(scan_path)
    bin_path = os.path.join(output_dir, f"{base}_seg.nii.gz")
    prob_path = os.path.join(output_dir, f"{base}_prob.nii.gz")

    nib.save(nib.Nifti1Image(binary, affine), bin_path)
    nib.save(nib.Nifti1Image(prob_np, affine), prob_path)

    logger.info(f"Saved segmentation: {bin_path}")
    logger.info(f"Saved probability:  {prob_path}")
    return bin_path, prob_path