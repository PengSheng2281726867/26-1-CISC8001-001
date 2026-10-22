#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
exec > >(tee -a "$LIFT_ENV_ROOT/setup.log") 2>&1
LIFT_PYTHON="$LIFT_ENV_ROOT/venv/bin/python"
# The old robomimic wrapper imports EGL probing even for CPU/offscreen use.
LIFT_CMAKE_WHEEL="$LIFT_ENV_ROOT/downloads/cmake-3.27.9-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
if [ -f "$LIFT_CMAKE_WHEEL" ]; then
    echo '434714990d82e3c3936a726c1706c6a1d5a34964a7415d1433af0904a994e414  '"$LIFT_CMAKE_WHEEL" | sha256sum -c -
    "$LIFT_PYTHON" -m pip install "$LIFT_CMAKE_WHEEL"
else
    "$LIFT_PYTHON" -m pip install 'cmake==3.27.9'
fi
"$LIFT_PYTHON" -m pip install --no-build-isolation 'egl-probe==1.0.1' 'tensorboard==2.11.2' 'tensorboardX==2.6.2.2' 'protobuf==3.20.3'
"$LIFT_PYTHON" -m pip check
"$LIFT_PYTHON" -m pip freeze > "$LIFT_ENV_ROOT/installed_packages.txt"
