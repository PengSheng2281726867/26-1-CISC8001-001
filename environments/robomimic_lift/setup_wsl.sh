#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
exec > >(tee -a "$LIFT_ENV_ROOT/setup.log") 2>&1
trap 'echo "SETUP FAILED at line $LINENO (exit $?) on $(date -u +%FT%TZ)"' ERR
echo "Starting legacy Lift setup on $(date -u +%FT%TZ)"
mkdir -p "$LIFT_ENV_ROOT/downloads" "$LIFT_ENV_ROOT/bin" "$LIFT_ENV_ROOT/vendor"

download() {
    if [ ! -s "$2" ]; then
        curl -L --fail --retry 3 --retry-all-errors --continue-at - --connect-timeout 20 --max-time 600 "$1" -o "$2.partial"
        mv -- "$2.partial" "$2"
    fi
}

download 'https://github.com/astral-sh/uv/releases/download/0.9.4/uv-x86_64-unknown-linux-gnu.tar.gz' "$LIFT_ENV_ROOT/downloads/uv.tar.gz"
if [ ! -x "$LIFT_ENV_ROOT/bin/uv" ]; then
    tar -xzf "$LIFT_ENV_ROOT/downloads/uv.tar.gz" -C "$LIFT_ENV_ROOT/bin" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
fi
"$LIFT_ENV_ROOT/bin/uv" python install 3.8.20
if [ ! -x "$LIFT_ENV_ROOT/venv/bin/python" ]; then
    "$LIFT_ENV_ROOT/bin/uv" venv --python 3.8.20 --seed "$LIFT_ENV_ROOT/venv"
fi
LIFT_PYTHON="$LIFT_ENV_ROOT/venv/bin/python"
"$LIFT_PYTHON" -m pip install 'pip==23.2.1' 'setuptools==59.5.0' 'wheel==0.41.3'

download 'https://www.roboti.us/download/mujoco200_linux.zip' "$LIFT_ENV_ROOT/downloads/mujoco200_linux.zip"
if [ ! -f "$MUJOCO_PY_MUJOCO_PATH/bin/libmujoco200.so" ]; then
    "$LIFT_PYTHON" -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$LIFT_ENV_ROOT/downloads/mujoco200_linux.zip" "$LIFT_ENV_ROOT"
fi
download 'https://www.roboti.us/file/mjkey.txt' "$MUJOCO_PY_MJKEY_PATH"

download 'https://codeload.github.com/ARISE-Initiative/robosuite/zip/d9baa17d5328d7b0ec15ecda54c76c61bb983124' "$LIFT_ENV_ROOT/downloads/robosuite_offline_study.zip"
if [ ! -d "$LIFT_ENV_ROOT/vendor/robosuite-d9baa17d5328d7b0ec15ecda54c76c61bb983124" ]; then
    "$LIFT_PYTHON" -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$LIFT_ENV_ROOT/downloads/robosuite_offline_study.zip" "$LIFT_ENV_ROOT/vendor"
fi
# mujoco-py's unbounded build requirements otherwise select incompatible Cython 3.
grep -v '^mujoco-py==' "$LIFT_SCRIPT_DIR/requirements.txt" > "$LIFT_ENV_ROOT/requirements_build_dependencies.txt"
"$LIFT_PYTHON" -m pip install -r "$LIFT_ENV_ROOT/requirements_build_dependencies.txt"
"$LIFT_PYTHON" -m pip install --no-build-isolation 'mujoco-py==2.0.2.9'
"$LIFT_PYTHON" -m pip install --no-deps "$LIFT_ENV_ROOT/vendor/robosuite-d9baa17d5328d7b0ec15ecda54c76c61bb983124" 'robomimic==0.2.0'
# CPU Torch allows using the original robomimic environment wrapper, with no training.
"$LIFT_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==1.10.2+cpu' 'torchvision==0.11.3+cpu'
bash "$LIFT_SCRIPT_DIR/fix_torch_stack.sh"
bash "$LIFT_SCRIPT_DIR/install_runtime_extras.sh"
"$LIFT_PYTHON" -m pip freeze > "$LIFT_ENV_ROOT/installed_packages.txt"
"$LIFT_PYTHON" -c 'import mujoco_py,robosuite; print("MuJoCo binding:",mujoco_py.__version__); print("robosuite:",robosuite.__version__)'
echo "Package setup completed on $(date -u +%FT%TZ)"
