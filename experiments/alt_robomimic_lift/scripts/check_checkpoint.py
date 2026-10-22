#!/usr/bin/env python
"""Load a trained or compatibility checkpoint in the active Torch/TorchVision runtime."""

from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.encoder import VisualProprioEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    model = VisualProprioEncoder(checkpoint["variant"], pretrained_path=None)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = {
        "status": "passed",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "variant": checkpoint["variant"],
        "embedding_dim": checkpoint["embedding_dim"],
        "checkpoint_run_id": checkpoint.get("run_id", "compatibility_fixture"),
        "missing_or_unexpected_keys": [],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
