#!/usr/bin/env python
"""Thin wrapper: `python scripts/train.py ...` == `lung-seg train ...`."""
import sys

from lung_seg.cli import main

if __name__ == "__main__":
    sys.exit(main(["train"] + sys.argv[1:]))