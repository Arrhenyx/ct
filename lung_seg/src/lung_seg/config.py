"""Typed configuration loaded from YAML."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Tuple

import yaml

Tuple3 = Tuple[float, float, float]
IntTuple3 = Tuple[int, int, int]


@dataclass
class Config:
    # Data
    data_dir: str = "data"
    k_folds: int = 5
    fold: int = 0
    seed: int = 42

    # Preprocessing
    pixdim: Tuple3 = (1.0, 1.0, 1.0)
    hu_min: float = -1000.0
    hu_max: float = 400.0

    # Patch
    roi_size: IntTuple3 = (96, 96, 96)
    num_samples: int = 4
    pos_ratio: float = 1.0
    neg_ratio: float = 1.0

    # Model
    model_name: str = "unetr"          # unetr | swinunetr | unet
    feature_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    dropout: float = 0.1
    swin_feature_size: int = 48

    # Optimization
    batch_size: int = 2
    val_batch_size: int = 1
    accum_steps: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 200
    val_interval: int = 2
    patience: int = 20
    early_stop_delta: float = 0.001
    grad_clip: float = 1.0
    amp: bool = True

    # Inference
    sw_overlap: float = 0.6
    sw_batch_size: int = 2
    sw_mode: str = "gaussian"
    min_cc_size: int = 27
    tta: bool = False

    # I/O
    output_dir: str = "predictions"
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    def replace(self, **kwargs: Any) -> "Config":
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_TUPLE_FIELDS = {"pixdim", "roi_size"}


def _coerce(key: str, value: Any) -> Any:
    if key in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load YAML config; overrides (non-None values) win over file values."""
    cfg_dict: dict[str, Any] = {}
    if path is not None:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(Config)}
        for k, v in raw.items():
            if k not in valid:
                continue
            cfg_dict[k] = _coerce(k, v)
    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**cfg_dict)
