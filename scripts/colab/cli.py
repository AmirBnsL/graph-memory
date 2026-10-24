#!/usr/bin/env python3
"""Unified Colab Manager script."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
src_dir = ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from graph_memory.colab_mgr import main

if __name__ == "__main__":
    main()
