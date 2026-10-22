#!/usr/bin/env python
"""Create a tiny local state-dict fixture for Torch/TorchVision cross-version checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.encoder import VisualProprioEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite compatibility fixture: %s" % args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = VisualProprioEncoder("image_proprio", pretrained_path=None)
    torch.save({
        "model_state_dict": model.state_dict(),
        "proprio_mean": torch.zeros(9),
        "proprio_std": torch.ones(9),
        "variant": "image_proprio",
        "embedding_dim": 128,
        "run_id": "compatibility_fixture",
    }, str(args.output))
    print(str(args.output))


if __name__ == "__main__":
    main()
