#!/usr/bin/env python
"""CLI wrapper for the frozen ALT-Lift-30 training run."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.train import main

if __name__ == "__main__":
    main()
