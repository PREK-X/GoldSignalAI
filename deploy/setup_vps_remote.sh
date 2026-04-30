#!/usr/bin/env bash
#
# GoldSignalAI — one-command remote VPS deploy.
#
# Provisions a fresh Ubuntu/Debian VPS over SSH:
#   1. installs python3.12 + venv + git
#   2. clones github.com/PREK-X/GoldSignalAI (token-authed)
#   3. creates venv and installs requirements.txt
#   4. scp's local .env to the VPS
#   5. installs and enables the goldsignalai systemd unit
#   6. runs `main.py --health-check` to validate
#
# Usage:
#   # 1) populate .env with VPS_IP / VPS_USER / SSH_KEY_PATH
#   # 2) bash deploy/setup_vps_remote.sh
#   # OR pass positional overrides:
#   #    bash deploy/setup_vps_remote.sh <ip> [user] [ssh_key_path]
#
# Requires: ssh, scp, and (locally) .env at repo root.

set -euo pipefail

# ── Resolve repo root (script may be invoked from anywhere) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load .env (positional args win over .env values) ────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$REPO_ROOT/.env"; set +a
fi

VPS_IP="${1:-${VPS_IP:-}}"
VPS_USER="${2:-${VPS_USER:-root}}"
SSH_KEY_PATH="${3:-${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}}"

# Expand ~ in SSH_KEY_PATH
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

if [[ -z "$VPS_IP" ]]; then
    echo "FATAL: VPS_IP not set (env var or positional arg 1)." >&2
    exit 1
fi
if [[ ! -f "$SSH_KEY_PATH" ]]; then
    echo "FATAL: SSH key not found: $SSH_KEY_PATH" >&2
    exit 1
fi
if [[ ! -f "$REPO_ROOT/.env" ]]; then
    echo "FATAL: $REPO_ROOT/.env missing — copy from deploy/.env.template first." >&2
    exit 1
fi

echo "==> Target: $VPS_USER@$VPS_IP (key: $SSH_KEY_PATH)"

# ── Prompt for GitHub token locally; never store it on the VPS ──────────
read -rsp "GitHub token (clone access for PREK-X/GoldSignalAI) [or press Enter to skip clone if already there]: " GH_TOKEN
echo

SSH_OPTS=(-i "$SSH_KEY_PATH" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
SSH_CMD=(ssh "${SSH_OPTS[@]}" "$VPS_USER@$VPS_IP")
SCP_CMD=(scp "${SSH_OPTS[@]}")

# ── Remote provisioning ─────────────────────────────────────────────────
echo "==> Installing Python + git on remote..."
"${SSH_CMD[@]}" 'bash -s' <<'REMOTE_PROVISION'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip git \
        || sudo apt-get install -y -qq python3 python3-venv python3-pip git
elif command -v dnf >/dev/null; then
    sudo dnf install -y python3.12 git
else
    echo "FATAL: unsupported package manager (need apt-get or dnf)" >&2
    exit 1
fi

mkdir -p "$HOME/GoldSignalAI"
REMOTE_PROVISION

# ── Clone (or pull) the repo on the VPS ─────────────────────────────────
echo "==> Cloning / updating repo on remote..."
"${SSH_CMD[@]}" "GH_TOKEN='$GH_TOKEN' bash -s" <<'REMOTE_CLONE'
set -euo pipefail
cd "$HOME"
if [[ -d GoldSignalAI/.git ]]; then
    cd GoldSignalAI
    if [[ -n "${GH_TOKEN:-}" ]]; then
        git remote set-url origin "https://${GH_TOKEN}@github.com/PREK-X/GoldSignalAI.git"
    fi
    git pull --ff-only
else
    if [[ -z "${GH_TOKEN:-}" ]]; then
        echo "FATAL: no token + no existing clone — cannot proceed." >&2
        exit 1
    fi
    git clone "https://${GH_TOKEN}@github.com/PREK-X/GoldSignalAI.git"
    cd GoldSignalAI
fi

# Strip token from origin so it doesn't persist on disk
git remote set-url origin "https://github.com/PREK-X/GoldSignalAI.git"
REMOTE_CLONE

# ── Build venv + install requirements ───────────────────────────────────
echo "==> Building venv + installing requirements..."
"${SSH_CMD[@]}" 'bash -s' <<'REMOTE_VENV'
set -euo pipefail
cd "$HOME/GoldSignalAI"
PY_BIN="$(command -v python3.12 || command -v python3)"
if [[ ! -d venv ]]; then
    "$PY_BIN" -m venv venv
fi
venv/bin/pip install --upgrade pip --quiet
venv/bin/pip install -r requirements.txt --quiet
REMOTE_VENV

# ── Push local .env ─────────────────────────────────────────────────────
echo "==> Copying local .env to remote..."
"${SCP_CMD[@]}" "$REPO_ROOT/.env" "$VPS_USER@$VPS_IP:~/GoldSignalAI/.env"

# ── Install systemd unit (templated to actual user / home) ──────────────
echo "==> Installing systemd unit..."
"${SSH_CMD[@]}" 'bash -s' <<REMOTE_SYSTEMD
set -euo pipefail
cd "\$HOME/GoldSignalAI"
USER_NAME="\$(whoami)"
HOME_DIR="\$HOME"

sudo tee /etc/systemd/system/goldsignalai.service >/dev/null <<UNIT
[Unit]
Description=GoldSignalAI — XAU/USD Trading Signal Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=\$USER_NAME
WorkingDirectory=\$HOME_DIR/GoldSignalAI
ExecStart=\$HOME_DIR/GoldSignalAI/venv/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=goldsignalai
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable goldsignalai
REMOTE_SYSTEMD

# ── Health check ────────────────────────────────────────────────────────
echo "==> Running remote health-check..."
"${SSH_CMD[@]}" 'cd ~/GoldSignalAI && venv/bin/python main.py --health-check'

# ── Start service + summary ─────────────────────────────────────────────
echo "==> Starting goldsignalai.service..."
"${SSH_CMD[@]}" 'sudo systemctl restart goldsignalai && sleep 2 && systemctl status goldsignalai --no-pager | head -15'

echo ""
echo "================================================================"
echo "  Deploy complete: $VPS_USER@$VPS_IP"
echo "  Status:  ssh -i $SSH_KEY_PATH $VPS_USER@$VPS_IP 'systemctl status goldsignalai'"
echo "  Logs:    ssh -i $SSH_KEY_PATH $VPS_USER@$VPS_IP 'journalctl -u goldsignalai -f'"
echo "  Quick:   bash deploy/connect_vps.sh"
echo "================================================================"
