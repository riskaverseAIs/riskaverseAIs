#!/usr/bin/env bash
# Lambda / "works everywhere" setup: fresh venv from requirements.txt
#
# How to use:
#   1. cd /path/to/this/repo
#   2. Run:  bash setup_venv_lambda.sh
#   3. Start Jupyter, kernel → Python (venv-lambda)
#
# Python: needs 3.10+. If `python3.10` is missing but pyenv has 3.10.x:
#   pyenv shell 3.10.14 && bash setup_venv_lambda.sh
# Or force a binary:
#   PYTHON_BIN="$HOME/.pyenv/versions/3.10.14/bin/python" bash setup_venv_lambda.sh
#
# Jupyter needs sqlite3. If pyenv Python was built without libsqlite, use conda/module Python,
# or: SKIP_IPYKERNEL=1 bash setup_venv_lambda.sh  (training/eval CLI only, no notebook kernel)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Force fp16 so training doesn't hit bf16 GradScaler error (set before any Python/training runs)
export ACCELERATE_MIXED_PRECISION=fp16

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '%s' "$PYTHON_BIN"
    return 0
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    printf '%s' "python3.10"
    return 0
  fi
  for c in python3.11 python3.12 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "No Python >= 3.10 found on PATH."
  echo ""
  echo "If you use pyenv, select a version first, then re-run from this repo:"
  echo "  cd \"$SCRIPT_DIR\""
  echo "  pyenv shell 3.10.14"
  echo "  bash setup_venv_lambda.sh"
  echo ""
  echo "Or point at a specific interpreter:"
  echo "  PYTHON_BIN=\"\$HOME/.pyenv/versions/3.10.14/bin/python\" bash setup_venv_lambda.sh"
  exit 1
fi

echo "Using Python: $PY — $($PY -c 'import sys; print(sys.version.split()[0])')"

if [[ "${SKIP_IPYKERNEL:-}" != "1" ]]; then
  if ! "$PY" -c 'import sqlite3' 2>/dev/null; then
    echo ""
    echo "ERROR: This Python has no working sqlite3 (missing _sqlite3). IPython/Jupyter need it."
    echo "Common cause: pyenv-built Python compiled without SQLite dev libraries."
    echo ""
    echo "Fix (pick one):"
    echo "  • ORCD/cluster: use conda instead —  conda create -n risk-eval python=3.11 -y"
    echo "  • Load a full system module:  module avail python  &&  module load <python-module>"
    echo "  • Rebuild pyenv after sqlite headers exist, then: pyenv uninstall 3.10.14 && pyenv install 3.10.14"
    echo "  • Training/eval only (no Jupyter kernel):  SKIP_IPYKERNEL=1 bash setup_venv_lambda.sh"
    echo ""
    exit 1
  fi
fi

"$PY" -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

# Register Jupyter kernel (requires sqlite3 in the base interpreter — see check above)
if [[ "${SKIP_IPYKERNEL:-}" != "1" ]]; then
  python -m pip install ipykernel
  python -m ipykernel install --user --name venv-lambda --display-name "Python (venv-lambda)"
else
  echo "Skipped ipykernel (SKIP_IPYKERNEL=1). Use conda/module Python if you need notebooks later."
fi

# So every time you activate this venv, fp16 is set (avoids bf16 GradScaler error in notebooks/sweep.py)
if ! grep -q 'ACCELERATE_MIXED_PRECISION=fp16' .venv/bin/activate 2>/dev/null; then
  echo 'export ACCELERATE_MIXED_PRECISION=fp16' >> .venv/bin/activate
fi

if [[ "${SKIP_IPYKERNEL:-}" != "1" ]]; then
  echo "Done. Restart Jupyter and switch kernel to Python (venv-lambda)."
fi
echo "Activate in a new shell:  source \"$SCRIPT_DIR/.venv/bin/activate\""
