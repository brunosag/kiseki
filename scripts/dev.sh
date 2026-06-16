#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

backend_pid=""
frontend_pid=""

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

cleanup() {
  local status=$?

  trap - EXIT INT TERM

  if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
    kill "$backend_pid" 2>/dev/null || true
  fi

  if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
    kill "$frontend_pid" 2>/dev/null || true
  fi

  wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
  exit "$status"
}

trap cleanup EXIT INT TERM

require_command uv
require_command npm

if [[ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]]; then
  echo "Frontend dependencies are not installed." >&2
  echo "Run: npm --prefix frontend install" >&2
  exit 1
fi

(
  cd "$BACKEND_DIR"
  uv run fastapi dev main.py
) &
backend_pid=$!

(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
) &
frontend_pid=$!

echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:5173"
echo "Press Ctrl+C to stop both services."

set +e
wait -n "$backend_pid" "$frontend_pid"
status=$?
set -e

exit "$status"
