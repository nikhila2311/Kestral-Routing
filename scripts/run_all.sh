#!/usr/bin/env bash
# Task runner (Git Bash on Windows, bash elsewhere).
# Usage: bash scripts/run_all.sh <target> [<target> ...]
# Targets: clean audit baselines experiments train evaluate predict serve test
#   clean = data cleaning; it does not delete files.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONHASHSEED=42

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  echo "No .venv found. See Setup in CLAUDE.md." >&2
  exit 1
fi

# Placeholders exit non-zero so an unbuilt step cannot look like a success.
not_implemented() {
  echo "Target not implemented yet: $1" >&2
  exit 2
}

usage() {
  echo "Usage: bash scripts/run_all.sh <target> [<target> ...]"
  echo "Targets: clean audit baselines experiments train evaluate predict serve test"
}

run_target() {
  case "$1" in
    clean) "$PY" -m src.clean ;;
    audit) "$PY" -m src.audit ;;
    baselines) "$PY" -m src.baselines ;;
    experiments) "$PY" -m src.experiments ;;
    train | evaluate | predict | serve) not_implemented "$1" ;;
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
