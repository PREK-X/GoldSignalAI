#!/usr/bin/env bash
#
# GoldSignalAI — quick VPS connect helper.
# Reads VPS_IP / VPS_USER / SSH_KEY_PATH from .env (or positional args).
# Tails recent logs, shows systemd status, then drops into an interactive shell.
#
# Usage:
#   bash deploy/connect_vps.sh                 # use .env values
#   bash deploy/connect_vps.sh <ip> [user] [ssh_key_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$REPO_ROOT/.env"; set +a
fi

VPS_IP="${1:-${VPS_IP:-}}"
VPS_USER="${2:-${VPS_USER:-root}}"
SSH_KEY_PATH="${3:-${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}}"
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

if [[ -z "$VPS_IP" ]]; then
    echo "FATAL: VPS_IP not set (env var or positional arg 1)." >&2
    exit 1
fi

exec ssh -t -i "$SSH_KEY_PATH" "$VPS_USER@$VPS_IP" '
    echo "=== Recent logs ===";
    journalctl -u goldsignalai -n 30 --no-pager 2>/dev/null \
        || tail -n 30 ~/GoldSignalAI/logs/*.log 2>/dev/null \
        || echo "(no logs yet)";
    echo "";
    echo "=== Service status ===";
    systemctl status goldsignalai --no-pager 2>/dev/null | head -15 \
        || echo "(service not installed)";
    echo "";
    cd ~/GoldSignalAI 2>/dev/null || true;
    exec bash -l
'
