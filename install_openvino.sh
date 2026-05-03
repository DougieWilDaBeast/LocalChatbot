#!/bin/bash
# =============================================================================
# Intel Iris Xe GPU Acceleration for NUC i5-1135G7
# Accelerates Whisper STT via OpenVINO
# OS: Ubuntu 24.04 LTS (noble)
# =============================================================================

set -e

echo ""
echo "=== Intel Iris Xe Acceleration Setup ==="
echo "    i5-1135G7 | Intel Iris Xe (96 EUs)"
echo ""

# --- 1. Clean up any broken external repos ---
echo "[→] Cleaning stale package sources..."
sudo rm -f /etc/apt/sources.list.d/intel-gpu-focal.list
sudo rm -f /etc/apt/sources.list.d/deadsnakes*
sudo rm -f /usr/share/keyrings/intel-graphics.gpg
sudo apt-get update -qq

# --- 2. Install Intel GPU compute runtime (Ubuntu 24.04 native packages) ---
echo "[→] Installing Intel GPU compute runtime..."
sudo apt-get install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    intel-media-va-driver-non-free \
    libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa \
    libgles2-mesa libglx-mesa0 \
    2>/dev/null || echo "[!] Some GPU packages unavailable — continuing with what installed"

echo "[✓] Intel GPU drivers installed"

# --- 3. Add user to video/render groups ---
sudo usermod -aG video,render $USER
echo "[✓] User added to video and render groups (logout/login required)"

# --- 4. Install OpenVINO ---
echo "[→] Installing OpenVINO toolkit..."
source venv/bin/activate

pip install openvino -q
pip install optimum[openvino] -q

echo "[✓] OpenVINO installed"

# --- 5. Upgrade faster-whisper ---
echo "[→] Upgrading faster-whisper..."
pip install --upgrade faster-whisper -q

echo "[✓] faster-whisper configured"

# --- 6. Test OpenVINO device detection ---
echo "[→] Detecting OpenVINO devices..."
python3 -c "
from openvino.runtime import Core
core = Core()
devices = core.available_devices
print(f'Available OpenVINO devices: {devices}')
for device in devices:
    props = core.get_property(device, 'FULL_DEVICE_NAME')
    print(f'  {device}: {props}')
"

echo ""
echo "=== OpenVINO Setup Complete ==="
echo ""
echo "Your config.yaml already has 'device: auto' which will use OpenVINO."
echo ""
echo "NOTE: You must log out and back in (or reboot) for GPU group changes"
echo "to take effect. Until then, only CPU will be available."
echo ""
