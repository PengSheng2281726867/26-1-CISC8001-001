"""Small reproducibility and path-resolution helpers for the isolated Lift run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config_path(value: str, base: Path = HERE) -> Path:
    if len(value) > 2 and value[1] == ":" and value[2] in ("\\", "/"):
        if os.name == "posix":
            return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(HERE), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def json_write(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
