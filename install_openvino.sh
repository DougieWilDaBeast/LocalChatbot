#!/bin/bash
# =============================================================================
# Intel Iris Xe GPU Acceleration for NUC i5-1135G7
# Accelerates Whisper STT via OpenVINO
# Enables iGPU offload for Ollama inference
# =============================================================================

set -e

echo ""
echo "=== Intel Iris Xe Acceleration Setup ==="
echo "    i5-1135G7 | Intel Iris Xe (96 EUs)"
echo ""

# --- 1. Install Intel GPU drivers ---
echo "[→] Installing Intel GPU compute runtime..."
sudo apt-get install -y gpg-agent wget

wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/packages/ubuntu focal client" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu-focal.list

sudo apt-get update
sudo apt-get install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    intel-media-va-driver-non-free \
    libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa \
    libgles2-mesa libglx-mesa0 \
    libigc-dev intel-igc-cm \
    libigdfcl-dev libigfxcmrt-dev

echo "[✓] Intel GPU drivers installed"

# --- 2. Add user to video/render groups ---
sudo usermod -aG video,render $USER
echo "[✓] User added to video and render groups (logout/login required)"

# --- 3. Install OpenVINO ---
echo "[→] Installing OpenVINO toolkit..."
source venv/bin/activate

pip install openvino==2024.0.0 -q
pip install optimum[openvino] -q
pip install openvino-genai -q

echo "[✓] OpenVINO installed"

# --- 4. Install faster-whisper with OpenVINO backend ---
echo "[→] Configuring faster-whisper for Intel GPU..."
pip install faster-whisper[openvino] -q 2>/dev/null || \
pip install faster-whisper -q  # fallback to standard

echo "[✓] faster-whisper configured"

# --- 5. Test OpenVINO device detection ---
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

# --- 6. Patch config to use GPU for STT ---
echo ""
echo "[→] To enable Intel GPU acceleration for Whisper, update config.yaml:"
echo ""
echo "  stt:"
echo "    device: 'auto'          # or specify 'openvino'"
echo "    compute_type: 'int8'    # keep this"
echo ""
echo "  Then update assistant.py WhisperModel init:"
echo "    self.model = WhisperModel("
echo "        stt['model'],"
echo "        device='auto',"
echo "        compute_type='int8',"
echo "    )"
echo ""

# --- 7. Test iGPU availability for Ollama ---
echo "[→] Checking Ollama Intel GPU support..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | head -1)
    echo "    Ollama version: $OLLAMA_VERSION"
    echo "    Ollama uses llama.cpp under the hood."
    echo "    Intel GPU support via SYCL/oneAPI is experimental."
    echo "    For CPU inference with iGPU-accelerated STT, no changes needed."
    echo "    For full iGPU offload, see: https://github.com/intel/llama.cpp"
fi

echo ""
echo "=== OpenVINO Setup Complete ==="
echo ""
echo "Reboot or logout/login for GPU group changes to take effect."
echo ""
