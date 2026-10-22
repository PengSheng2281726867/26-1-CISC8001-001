#!/usr/bin/env bash
# Run one command in the existing, read-only Lift simulator environment.
# All Python caches created by this experiment are redirected into this folder.
set -euo pipefail
LIFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd -- "$LIFT_DIR/../.." && pwd)"
ENV_ROOT="$PROJECT_ROOT/artifacts/environments/robomimic_lift_legacy"
MUJOCO_ROOT="$ENV_ROOT/mujoco200_linux"

export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_ROOT"
export MUJOCO_PY_MJKEY_PATH="$MUJOCO_ROOT/bin/mjkey.txt"
export MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH="$MUJOCO_ROOT/bin:${LD_LIBRARY_PATH:-}"
export CC=gcc
export CXX=g++
export LDSHARED="bash \"$PROJECT_ROOT/environments/robomimic_lift/link_gcc.sh\" -pthread -shared"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -include stdio.h"
export TORCH_HOME="$LIFT_DIR/artifacts/cache/torch"
export XDG_CACHE_HOME="$LIFT_DIR/artifacts/cache/xdg"
export NUMBA_CACHE_DIR="$LIFT_DIR/artifacts/cache/numba"
export MPLCONFIGDIR="$LIFT_DIR/artifacts/cache/matplotlib"
export TMPDIR="$LIFT_DIR/artifacts/tmp"
export PYTHONPATH="$LIFT_DIR/src:${PYTHONPATH:-}"

exec "$ENV_ROOT/venv/bin/python" -B "$@"
