#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
exec > >(tee -a "$LIFT_ENV_ROOT/validation.log") 2>&1
"$LIFT_ENV_ROOT/venv/bin/python" -u -B "$LIFT_SCRIPT_DIR/validate.py"
