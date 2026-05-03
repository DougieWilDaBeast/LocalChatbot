#!/bin/bash
# =============================================================================
# NUC Offline Voice Assistant - Full Setup Script
# Hardware: Intel NUC i5-1135G7, 32GB RAM, Intel Iris Xe
# OS: Ubuntu 24.04 LTS
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
info() { echo -e "${BLUE}[→]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo ""
echo "============================================="
echo "  NUC Offline Voice Assistant — Setup"
echo "  Intel i5-1135G7 | 32GB RAM | Iris Xe"
echo "============================================="
echo ""

# -----------------------------------------------------------------------------
# 1. System Update
# -----------------------------------------------------------------------------
info "Updating system packages..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq
sudo apt-get install -y \
    curl wget git python3 python3-pip python3-venv \
    portaudio19-dev libportaudio2 libportaudiocpp0 \
    ffmpeg alsa-utils pulseaudio \
    build-essential cmake pkg-config \
    libasound2-dev libsndfile1-dev
log "System packages installed"

# -----------------------------------------------------------------------------
# 2. Ollama
# -----------------------------------------------------------------------------
info "Installing Ollama..."
if command -v ollama &> /dev/null; then
    warn "Ollama already installed, skipping"
else
    curl -fsSL https://ollama.ai/install.sh | sh
    log "Ollama installed"
fi

info "Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 3

info "Pulling Gemma 3 4B model (this will take a few minutes)..."
ollama pull gemma3:4b
log "Gemma 3 4B model ready"

# Optionally pull the newer Gemma 4 E4B
# Uncomment if you want to test the newer model
# warn "Pulling Gemma 4 E4B (larger download)..."
# ollama pull gemma4:e4b

# -----------------------------------------------------------------------------
# 3. Python Virtual Environment
# -----------------------------------------------------------------------------
info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
log "Virtual environment created"

info "Installing Python dependencies..."
# Install all deps except openwakeword (tflite-runtime fails on Python 3.12)
grep -v '^openwakeword==' requirements.txt | pip install -r /dev/stdin -q
# Install openwakeword without pulling tflite-runtime (uses onnxruntime instead)
pip install openwakeword==0.6.0 --no-deps -q
log "Python dependencies installed"

# -----------------------------------------------------------------------------
# 4. Piper TTS
# -----------------------------------------------------------------------------
info "Installing Piper TTS (Python package)..."
pip install piper-tts -q
log "Piper TTS installed"

info "Downloading Piper voice model (en_GB-alba-medium)..."
mkdir -p models/piper
python3 -m piper.download_voices --download-dir models/piper en_GB-alba-medium \
    && log "Piper voice model downloaded" \
    || warn "Voice model download failed (no internet?). Copy .onnx + .onnx.json manually to models/piper/"

# -----------------------------------------------------------------------------
# 5. openWakeWord model
# -----------------------------------------------------------------------------
info "Downloading openWakeWord models..."
python3 -c "
import openwakeword
openwakeword.utils.download_models()
print('openWakeWord models downloaded')
" && log "Wake word models ready" || warn "openWakeWord model download failed — wake word may not work until models are available"

# -----------------------------------------------------------------------------
# 6. Audio device check
# -----------------------------------------------------------------------------
info "Checking audio devices..."
echo ""
echo "--- Available input devices ---"
python3 -c "
import sounddevice as sd
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f'  [{i}] {d[\"name\"]}')
"
echo ""
echo "--- Available output devices ---"
python3 -c "
import sounddevice as sd
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_output_channels'] > 0:
        print(f'  [{i}] {d[\"name\"]}')
"
echo ""
warn "Check the device indices above and update config.yaml accordingly"

# -----------------------------------------------------------------------------
# 7. Test Ollama is responding
# -----------------------------------------------------------------------------
info "Testing Ollama API..."
RESPONSE=$(curl -s http://localhost:11434/api/generate \
    -d '{"model":"gemma3:4b","prompt":"Reply with just the word ready.","stream":false}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])" 2>/dev/null || echo "FAIL")

if [[ "$RESPONSE" == *"FAIL"* ]]; then
    warn "Ollama test failed — check that the service is running: sudo systemctl status ollama"
else
    log "Ollama responding: '$RESPONSE'"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "============================================="
echo "  Setup Complete"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Create config:      cp config.example.yaml config.yaml"
echo "  2. Edit config:        nano config.yaml"
echo "  3. Test the pipeline:  ./test_pipeline.sh"
echo "  4. Run the assistant:  source venv/bin/activate && python3 assistant.py"
echo "  5. Install as service: sudo ./install_service.sh"
echo ""
echo "For Intel Iris Xe GPU acceleration:"
echo "  ./install_openvino.sh"
echo ""
