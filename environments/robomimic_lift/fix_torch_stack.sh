#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
exec > >(tee -a "$LIFT_ENV_ROOT/setup.log") 2>&1
"$LIFT_ENV_ROOT/venv/bin/python" -B "$LIFT_SCRIPT_DIR/fix_torch_stack.py"
