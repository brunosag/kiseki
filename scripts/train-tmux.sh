#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

session_name="kiseki-train"
log_path=""
train_args=()

usage() {
  cat >&2 <<'EOF'
Usage: scripts/train-tmux.sh [--session NAME] [--log PATH] [--] [kiseki train args...]

Examples:
  scripts/train-tmux.sh --device gpu --optimizer LEEA --iterations 100000
  scripts/train-tmux.sh --session sgd-run -- --optimizer SGD --device cpu
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      if [[ $# -lt 2 ]]; then
        echo "--session requires a value" >&2
        usage
        exit 2
      fi
      session_name="$2"
      shift 2
      ;;
    --log)
      if [[ $# -lt 2 ]]; then
        echo "--log requires a value" >&2
        usage
        exit 2
      fi
      log_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      train_args=("$@")
      break
      ;;
    *)
      train_args=("$@")
      break
      ;;
  esac
done

require_command tmux
require_command uv

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session '$session_name' already exists." >&2
  echo "Attach with: tmux attach -t $session_name" >&2
  exit 1
fi

if [[ -z "$log_path" ]]; then
  log_path="$ROOT_DIR/logs/kiseki-train-$(date +%Y%m%dT%H%M%S).log"
elif [[ "$log_path" != /* ]]; then
  log_path="$ROOT_DIR/$log_path"
fi

mkdir -p "$(dirname "$log_path")"

train_cmd=(uv run kiseki train "${train_args[@]}")
printf -v quoted_train_cmd "%q " "${train_cmd[@]}"
printf -v quoted_log_path "%q" "$log_path"
inner_command="set -o pipefail; $quoted_train_cmd 2>&1 | tee $quoted_log_path; status=\${PIPESTATUS[0]}; echo; echo Log: $quoted_log_path; echo Training command exited with status \$status.; exec bash"
printf -v quoted_inner_command "%q" "$inner_command"

echo "Starting tmux session: $session_name"
echo "Log: $log_path"

tmux new-session -d -s "$session_name" -c "$BACKEND_DIR" "bash -lc $quoted_inner_command"
tmux attach -t "$session_name"
