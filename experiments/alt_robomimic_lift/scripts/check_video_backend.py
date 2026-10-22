#!/usr/bin/env python
"""Verify the legacy runtime can encode the predeclared two-camera MP4 outputs."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio as imageio_package
import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite the video backend smoke output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=20, codec="libx264", quality=7, macro_block_size=1)
    writer.append_data(np.zeros((84, 168, 3), dtype=np.uint8))
    writer.append_data(np.full((84, 168, 3), 127, dtype=np.uint8))
    writer.close()
    reader = imageio.get_reader(str(args.output))
    frame_size = reader.get_meta_data().get("size")
    reader.close()
    if tuple(frame_size or ()) != (168, 84):
        raise RuntimeError("Video backend changed the expected side-by-side frame size: %r" % (frame_size,))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"status": "passed", "frame_size": frame_size, "imageio": imageio_package.__version__, "imageio_ffmpeg": imageio_ffmpeg.__version__, "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(), "output_bytes": args.output.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
