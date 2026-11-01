#!/usr/bin/env python
import sys

from lung_seg.cli import main

if __name__ == "__main__":
    sys.exit(main(["kfold"] + sys.argv[1:]))