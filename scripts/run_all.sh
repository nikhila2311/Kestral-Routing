#!/usr/bin/env bash
# Task runner (Git Bash on Windows, bash elsewhere).
# Usage: bash scripts/run_all.sh <target> [<target> ...]
# Targets: clean audit baselines experiments router train evaluate predict dashboard serve test
#   clean = data cleaning; it does not delete files.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONHASHSEED=42

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  echo "No .venv found. See Setup in README.md." >&2
  exit 1
fi

usage() {
  echo "Usage: bash scripts/run_all.sh <target> [<target> ...]"
  echo "Targets: clean audit baselines experiments router train evaluate predict dashboard serve test"
}

run_target() {
  case "$1" in
    clean) "$PY" -m src.clean ;;
    audit) "$PY" -m src.audit ;;
    baselines) "$PY" -m src.baselines ;;
    experiments) "$PY" -m src.experiments ;;
    router) "$PY" -m src.router ;;
    evaluate) "$PY" -m src.evaluation ;;
    train) "$PY" -m src.predictions train ;;
    predict) "$PY" -m src.predictions predict ;;
    dashboard) "$PY" -m src.dashboard ;;
    serve) "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 ;;
    test) "$PY" -m pytest tests ;;
    *)
      echo "Unknown target: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

if [ $# -eq 0 ]; then
  usage
  exit 1
fi

for target in "$@"; do
  run_target "$target"
done
