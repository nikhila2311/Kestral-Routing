#!/usr/bin/env bash
# Clean-machine integration test (macOS/Linux; also runs in Git Bash). Windows: scripts/integration_test.ps1
#
# Usage, from anywhere: bash scripts/integration_test.sh
#   environment: RAW_DIR=<folder with the data pack>  PORT=<free port>  PYTHON=<python 3.12+>
#                KEEP=1 (leave the temp folder; it holds customer data)
#
# Tests the COMMITTED state (HEAD) in a fresh clone with a fresh venv. Nothing in this working copy is changed.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${RAW_DIR:-$REPO/data/raw}"
PORT="${PORT:-8765}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/kestrel_it_XXXXXX")"
CLONE="$WORK/repo"
LOGS="$WORK/logs"
URL="http://127.0.0.1:$PORT"
SERVER=""
FAILED=0
SUMMARY=""
STEP_NO=0
export PYTHONHASHSEED=42
mkdir -p "$LOGS"

stop_server() {
  if [ -n "$SERVER" ] && kill -0 "$SERVER" 2>/dev/null; then
    kill "$SERVER" 2>/dev/null
    wait "$SERVER" 2>/dev/null
  fi
  SERVER=""
}

cleanup() {
  stop_server
  if [ "${KEEP:-}" = "1" ]; then
    echo "Kept $WORK (contains customer data; delete it when done)"
  elif ! rm -rf "$WORK"; then
    echo "WARNING could not delete $WORK (contains customer data; delete it by hand)"
  fi
}
trap cleanup EXIT

# step <name> <function> [args]: after the first failure the remaining steps are reported as SKIP.
step() {
  local name="$1" start status rc log
  shift
  if [ "$FAILED" -ne 0 ]; then
    echo "SKIP  $name"
    SUMMARY="${SUMMARY}SKIP  $name"$'\n'
    return
  fi
  STEP_NO=$((STEP_NO + 1))
  log="$LOGS/step_$STEP_NO.log"
  start=$SECONDS
  "$@" >"$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    status=PASS
  else
    status=FAIL
    FAILED=1
    tail -n 30 "$log" | sed 's/^/    /'
  fi
  echo "$status  $name ($((SECONDS - start))s)"
  SUMMARY="${SUMMARY}${status}  $name"$'\n'
}

find_python() {
  local candidate
  for candidate in ${PYTHON:+"$PYTHON"} python3.13 python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 12))' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  # Git Bash on Windows
  if command -v py >/dev/null 2>&1; then
    py -3.13 -c 'import sys; print(sys.executable)' 2>/dev/null | tr -d '\r' && return 0
  fi
  return 1
}

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

venv_python() {
  if [ -x "$CLONE/.venv/Scripts/python.exe" ]; then
    echo "$CLONE/.venv/Scripts/python.exe"
  else
    echo "$CLONE/.venv/bin/python"
  fi
}

do_clone() { git clone --quiet --no-hardlinks "$REPO" "$CLONE"; }

do_tracking() {
  local tracked
  tracked="$(git -C "$CLONE" ls-files data models)" || return 1
  [ -z "$tracked" ] || { echo "a file under data/ or models/ is tracked"; return 1; }
  [ -n "$(git -C "$CLONE" ls-files predictions.csv)" ] || { echo "predictions.csv is not tracked"; return 1; }
}

do_copy() {
  [ -f "$RAW_DIR/train.csv" ] || { echo "data pack not found in $RAW_DIR"; return 1; }
  mkdir -p "$CLONE/data/raw" && cp -R "$RAW_DIR/." "$CLONE/data/raw/"
}

do_venv() {
  local python
  python="$(find_python)" || { echo "Python 3.12 or newer not found; set PYTHON"; return 1; }
  (cd "$CLONE" && "$python" -m venv .venv)
}

do_pip() { (cd "$CLONE" && "$(venv_python)" -m pip install --quiet --disable-pip-version-check -r requirements.txt); }

do_remove() { rm "$CLONE/predictions.csv"; }

do_target() { (cd "$CLONE" && bash scripts/run_all.sh "$1"); }

do_identical() {
  # --no-filters hashes the raw bytes: a CRLF change counts as a difference.
  local new committed
  new="$(git -C "$CLONE" hash-object --no-filters predictions.csv)" || return 1
  committed="$(git -C "$CLONE" rev-parse HEAD:predictions.csv)" || return 1
  [ "$new" = "$committed" ] || { echo "predictions.csv differs from HEAD"; return 1; }
}

do_tests() { (cd "$CLONE" && "$(venv_python)" -m pytest tests -q -m "not slow"); }

do_start() {
  local i=0
  if port_open; then echo "port $PORT is already in use; set PORT"; return 1; fi
  cd "$CLONE" || return 1
  "$(venv_python)" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" >"$LOGS/api.log" 2>&1 &
  SERVER=$!
  cd "$REPO" || return 1
  until port_open; do
    kill -0 "$SERVER" 2>/dev/null || { echo "the API process exited"; tail -n 20 "$LOGS/api.log"; return 1; }
    i=$((i + 1))
    [ "$i" -lt 60 ] || { echo "the API did not start in 60 s"; return 1; }
    sleep 1
  done
}

do_health() {
  "$(venv_python)" -c '
import json, sys, urllib.request
answer = json.load(urllib.request.urlopen(sys.argv[1] + "/health", timeout=15))
assert answer["status"] == "ok" and answer["model_loaded"] is True
' "$URL"
}

do_predict() {
  (cd "$CLONE" && "$(venv_python)" -c '
import json, sys, urllib.request
sys.path.insert(0, ".")
from src.config import TEAMS
body = json.dumps({"request_text": "My mixer stopped working"}).encode()
request = urllib.request.Request(sys.argv[1] + "/predict", body, {"Content-Type": "application/json"})
assert json.load(urllib.request.urlopen(request, timeout=15))["team"] in TEAMS
' "$URL")
}

do_stop() {
  stop_server
  sleep 1
  if port_open; then echo "port $PORT still answers"; return 1; fi
}

echo "Repo:  $REPO"
echo "Temp:  $WORK"
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
  echo "NOTE   uncommitted changes exist; only the last commit is tested."
fi

step "clone" do_clone
step "git tracking (data/ and models/ untracked, predictions.csv tracked)" do_tracking
step "copy data/raw" do_copy
step "fresh venv" do_venv
step "pip install -r requirements.txt" do_pip
# predictions.csv is deleted first, so the comparison cannot pass on the committed copy.
step "remove committed predictions.csv from the clone" do_remove
for target in clean audit baselines experiments router evaluate train predict; do
  step "pipeline: $target" do_target "$target"
done
step "predictions.csv byte-identical to the committed one" do_identical
step "quick tests (pytest -m 'not slow')" do_tests
step "start API" do_start
step "GET /health" do_health
step "POST /predict" do_predict
step "stop API" do_stop

echo
printf '%s' "$SUMMARY"
if [ "$FAILED" -ne 0 ]; then
  echo "RESULT: FAIL"
  exit 1
fi
echo "RESULT: PASS"
