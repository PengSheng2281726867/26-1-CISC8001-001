#!/usr/bin/env bash
# Source this file or call run.ps1; all Python caches remain under the project.
LIFT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIFT_PROJECT_ROOT="$(cd -- "$LIFT_SCRIPT_DIR/../.." && pwd)"
LIFT_ENV_ROOT="$LIFT_PROJECT_ROOT/artifacts/environments/robomimic_lift_legacy"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_PY_MUJOCO_PATH="$LIFT_ENV_ROOT/mujoco200_linux"
export MUJOCO_PY_MJKEY_PATH="$LIFT_ENV_ROOT/mujoco200_linux/bin/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_FORCE_CPU=1
export CC=gcc
export CXX=g++
export LDSHARED="bash \"$LIFT_SCRIPT_DIR/link_gcc.sh\" -pthread -shared"
# GCC 14+ makes this legacy generated-C warning fatal; retain old compiler behavior.
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -include stdio.h"
export NUMBA_CACHE_DIR="$LIFT_ENV_ROOT/cache/numba"
export XDG_CACHE_HOME="$LIFT_ENV_ROOT/cache/xdg"
export MPLCONFIGDIR="$LIFT_ENV_ROOT/cache/matplotlib"
export TORCH_HOME="$LIFT_ENV_ROOT/cache/torch"
export PIP_CACHE_DIR="$LIFT_ENV_ROOT/cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export TMPDIR="$LIFT_ENV_ROOT/tmp"
export UV_CACHE_DIR="$LIFT_ENV_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$LIFT_ENV_ROOT/python"
export UV_PYTHON_BIN_DIR="$LIFT_ENV_ROOT/bin"
export UV_PYTHON_PREFERENCE=only-managed
export PATH="$LIFT_ENV_ROOT/venv/bin:$LIFT_ENV_ROOT/bin:$PATH"
mkdir -p "$TMPDIR" "$LIFT_ENV_ROOT/cache" "$LIFT_ENV_ROOT/results"
