#!/usr/bin/env python
import sys

from lung_seg.cli import main

if __name__ == "__main__":
    sys.exit(main(["predict"] + sys.argv[1:]))