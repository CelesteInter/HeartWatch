#!/usr/bin/env bash
# Launch the HeartWatch host app from its venv.
# Usage:  ./heartwatchapp/run.sh   (from anywhere)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/heartwatchapp/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "venv not found — creating it..."
    python3 -m venv "$REPO_ROOT/heartwatchapp/.venv"
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r "$REPO_ROOT/heartwatchapp/requirements.txt"
fi

cd "$REPO_ROOT"
exec "$VENV_PY" -m heartwatchapp.main
