#!/bin/bash
# Install and enable the systemd service
# Run as: sudo ./install_service.sh

set -e

SERVICE_FILE="nuc-assistant.service"
INSTALL_PATH="/etc/systemd/system/nuc-assistant.service"

# Replace placeholder username with actual user
ACTUAL_USER=$(logname 2>/dev/null || echo "$SUDO_USER")
ACTUAL_UID=$(id -u "$ACTUAL_USER")
REPO_PATH="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$REPO_PATH/venv/bin/python3"
PULSE_RUNTIME_PATH="/run/user/$ACTUAL_UID/pulse"

echo "[→] Installing service for user: $ACTUAL_USER"
echo "[→] Repo path: $REPO_PATH"

sed \
    -e "s|__SERVICE_USER__|$ACTUAL_USER|g" \
    -e "s|__REPO_PATH__|$REPO_PATH|g" \
    -e "s|__PYTHON_BIN__|$PYTHON_BIN|g" \
    -e "s|__PULSE_RUNTIME_PATH__|$PULSE_RUNTIME_PATH|g" \
    "$SERVICE_FILE" > "$INSTALL_PATH"

systemctl daemon-reload
systemctl enable nuc-assistant
systemctl start nuc-assistant

echo "[✓] Service installed and started"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status nuc-assistant"
echo "  sudo journalctl -u nuc-assistant -f"
echo "  sudo systemctl restart nuc-assistant"
echo "  sudo systemctl stop nuc-assistant"
