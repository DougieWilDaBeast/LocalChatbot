#!/bin/bash
# Install and enable the systemd service
# Run as: sudo ./install_service.sh

set -eo pipefail

if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi

REPO_PATH="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$REPO_PATH/nuc-assistant.service"
INSTALL_PATH="/etc/systemd/system/nuc-assistant.service"
PYTHON_BIN="$REPO_PATH/venv/bin/python3"

# The service runs as the user who called sudo, not as root
ACTUAL_USER="${SUDO_USER:-$(logname 2>/dev/null || true)}"
if [ -z "$ACTUAL_USER" ] || [ "$ACTUAL_USER" = "root" ]; then
    echo "[✗] Run this with sudo from your own account (not as root), so the service runs as you"
    exit 1
fi
ACTUAL_UID=$(id -u "$ACTUAL_USER")

[ -x "$PYTHON_BIN" ] || { echo "[✗] $PYTHON_BIN not found. Run ./setup.sh first"; exit 1; }
[ -f "$REPO_PATH/config.yaml" ] || { echo "[✗] config.yaml not found. Run: cp config.example.yaml config.yaml"; exit 1; }

echo "[→] Installing service for user: $ACTUAL_USER"
echo "[→] Repo path: $REPO_PATH"

# Escape characters that are special in a sed replacement (& | \)
esc() { printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'; }

# Render to a temporary file first, so a failure can't leave a broken unit behind
RENDERED=$(mktemp)
trap 'rm -f "$RENDERED"' EXIT
sed \
    -e "s|__SERVICE_USER__|$(esc "$ACTUAL_USER")|g" \
    -e "s|__SERVICE_UID__|$ACTUAL_UID|g" \
    -e "s|__REPO_PATH__|$(esc "$REPO_PATH")|g" \
    -e "s|__PYTHON_BIN__|$(esc "$PYTHON_BIN")|g" \
    "$SERVICE_FILE" > "$RENDERED"
install -m 644 "$RENDERED" "$INSTALL_PATH"

# Start the user's session services (PipeWire) at boot, without a login
loginctl enable-linger "$ACTUAL_USER"

systemctl daemon-reload
systemctl enable nuc-assistant
systemctl restart nuc-assistant

echo "[✓] Service installed and started"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status nuc-assistant"
echo "  sudo journalctl -u nuc-assistant -f"
echo "  sudo systemctl restart nuc-assistant"
echo "  sudo systemctl stop nuc-assistant"
