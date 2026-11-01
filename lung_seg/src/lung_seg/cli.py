"""Single entry point: `lung-seg {train,kfold,predict} ...`."""
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


def _common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--k_folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--no-amp", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("lung_seg")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="train a single fold")
    _common_flags(p_train)
    p_train.add_argument("--fold", type=int, default=None)
    p_train.add_argument("--resume", default=None)
    p_train.add_argument("--ckpt_dir", default=None)

    # kfold
    p_kfold = sub.add_parser("kfold", help="run k-fold cross validation")
    _common_flags(p_kfold)
    p_kfold.add_argument("--ckpt_dir", default=None)
    p_kfold.add_argument("--results_path", default="kfold_results.json")

    # predict
    p_pred = sub.add_parser("predict", help="inference on a single scan")
    _common_flags(p_pred)
    p_pred.add_argument("--model_path", required=True)
    p_pred.add_argument("--scan", required=True)
    p_pred.add_argument("--output_dir", default=None)
    p_pred.add_argument("--tta", action="store_true")
    p_pred.add_argument("--no-postprocess", action="store_true")

    return parser


def _build_config(args) -> "Config":
    overrides = dict(
        data_dir=args.data_dir,
        k_folds=args.k_folds,
        seed=args.seed,
        log_dir=args.log_dir,
    )
    if args.no_amp:
        overrides["amp"] = False
    if getattr(args, "fold", None) is not None:
        overrides["fold"] = args.fold
    if getattr(args, "ckpt_dir", None) is not None:
        overrides["ckpt_dir"] = args.ckpt_dir
    if getattr(args, "output_dir", None) is not None:
        overrides["output_dir"] = args.output_dir
    return load_config(args.config, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _build_config(args)
    setup_logger(log_dir=cfg.log_dir, tag=args.command)
    seed_everything(cfg.seed)

    if args.command == "train":
        train_model(cfg, fold=cfg.fold, resume_from=args.resume)
        return 0

    if args.command == "kfold":
        results = []
        for fold in range(cfg.k_folds):
            try:
                best, _ = train_model(cfg, fold=fold)
                results.append({"fold": fold, "best_dice": best})
            except Exception as e:  # noqa: BLE001
                print(f"[fold {fold}] failed: {e}", file=sys.stderr)
        if results:
            dices = [r["best_dice"] for r in results]
            print(f"mean Dice = {np.mean(dices):.4f} ± {np.std(dices):.4f}")
            Path(args.results_path).write_text(json.dumps(results, indent=2))
        return 0

    if args.command == "predict":
        predict_scan(
            cfg,
            model_path=args.model_path,
            scan_path=args.scan,
            output_dir=cfg.output_dir,
            post_process=not args.no_postprocess,
            tta=args.tta or None,
        )
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())