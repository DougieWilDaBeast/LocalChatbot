#!/bin/bash
# Install and enable the systemd service
# Run as: sudo ./scripts/install_service.sh

set -e

SERVICE_FILE="systemd/nuc-assistant.service"
INSTALL_PATH="/etc/systemd/system/nuc-assistant.service"

# Replace placeholder username with actual user
ACTUAL_USER=$(logname 2>/dev/null || echo "$SUDO_USER")
ACTUAL_HOME=$(eval echo "~$ACTUAL_USER")
REPO_PATH="$(cd "$(dirname "$0")/.." && pwd)"

echo "[→] Installing service for user: $ACTUAL_USER"
echo "[→] Repo path: $REPO_PATH"

sed \
    -e "s|/home/nuc|$ACTUAL_HOME|g" \
    -e "s|User=nuc|User=$ACTUAL_USER|g" \
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
