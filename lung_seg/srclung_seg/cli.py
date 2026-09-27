"""Top-level CLI entrypoint (`lung-seg` script)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .inference.predictor import predict_scan
from .training.trainer import train_model
from .utils.logging import setup_logger
from .utils.seed import seed_everything


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("lung_seg")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--mode", choices=["train", "predict", "kfold"], default="train")

    # common overrides
    p.add_argument("--data_dir", default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--k_folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--no-amp", action="store_true")

    # predict-only
    p.add_argument("--model_path", default=None)
    p.add_argument("--scan", default=None)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--no-postprocess", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    overrides = dict(
        data_dir=args.data_dir,
        fold=args.fold,
        k_folds=args.k_folds,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if args.no_amp:
        overrides["amp"] = False

    cfg = load_config(args.config, **overrides)
    setup_logger(log_dir=cfg.log_dir, tag=args.mode)
    seed_everything(cfg.seed)

    if args.mode == "train":
        train_model(cfg, fold=cfg.fold, resume_from=args.resume)

    elif args.mode == "kfold":
        results = []
        for fold in range(cfg.k_folds):
            try:
                best, _ = train_model(cfg, fold=fold)
                results.append({"fold": fold, "best_dice": best})
            except Exception as e:  # noqa: BLE001
                print(f"[fold {fold}] failed: {e}")
        if results:
            dices = [r["best_dice"] for r in results]
            print(f"mean Dice = {np.mean(dices):.4f} ± {np.std(dices):.4f}")
            Path("kfold_results.json").write_text(json.dumps(results, indent=2))

    elif args.mode == "predict":
        if not args.scan or not args.model_path:
            print("--scan and --model_path are required for predict")
            return 2
        predict_scan(
            cfg,
            model_path=args.model_path,
            scan_path=args.scan,
            output_dir=cfg.output_dir,
            post_process=not args.no_postprocess,
            tta=args.tta or None,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())