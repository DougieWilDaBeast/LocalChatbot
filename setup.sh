#!/bin/bash
# =============================================================================
# NUC Offline Voice Assistant - Full Setup Script
# Hardware: Intel NUC i5-1135G7, 32GB RAM, Intel Iris Xe
# OS: Ubuntu 24.04 LTS
# =============================================================================

set -eo pipefail

# Work from the repo folder, wherever the script is run from
cd "$(dirname "$0")"

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
info "Installing system packages..."
sudo apt-get update -qq
# pipewire-audio is Ubuntu 24.04's sound server (already there on Desktop,
# added on Server). Everything else comes as Python wheels, so no compilers.
sudo apt-get install -y \
    curl git python3 python3-pip python3-venv \
    libportaudio2 alsa-utils pipewire-audio zstd
log "System packages installed"

# -----------------------------------------------------------------------------
# 2. Ollama
# -----------------------------------------------------------------------------
info "Installing Ollama..."
if command -v ollama &> /dev/null; then
    warn "Ollama already installed, skipping"
else
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama installed"
fi

info "Starting Ollama service..."
if [ -d /run/systemd/system ]; then
    sudo systemctl enable ollama
    sudo systemctl start ollama
else
    # No systemd (container, WSL without systemd): run the server in the background
    warn "systemd not running, starting 'ollama serve' in the background"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
fi
for _ in $(seq 1 30); do
    curl -s http://localhost:11434/api/tags > /dev/null && break
    sleep 1
done

# Use config.yaml if it exists yet, otherwise the example
CONFIG="config.yaml"
[ -f "$CONFIG" ] || CONFIG="config.example.yaml"
MODEL=$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['model']['name'])" "$CONFIG" 2>/dev/null \
    || echo "gemma4:e4b")

info "Pulling $MODEL model (this will take a few minutes)..."
ollama pull "$MODEL"
log "$MODEL model ready"

# For lower-spec hardware, a lighter model:
# ollama pull gemma3:4b

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
pip install -q -r <(grep -v '^openwakeword==' requirements.txt)
# Install openwakeword without pulling tflite-runtime (uses onnxruntime instead)
pip install openwakeword==0.6.0 --no-deps -q
log "Python dependencies installed"

# -----------------------------------------------------------------------------
# 4. Piper TTS voice (piper-tts itself is installed from requirements.txt)
# -----------------------------------------------------------------------------
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
# 5b. Smart Turn end-of-turn model (BSD-2-Clause, pipecat-ai/smart-turn-v3)
# -----------------------------------------------------------------------------
info "Downloading Smart Turn model..."
mkdir -p models/smart-turn
SMART_TURN=models/smart-turn/smart-turn-v3.2-cpu.onnx
SMART_TURN_SHA256=2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f
if [ -f "$SMART_TURN" ]; then
    warn "Smart Turn model already downloaded, skipping"
# Download to a .part file and check it, so a broken download is never used
elif curl -fsSL -o "$SMART_TURN.part" \
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx" \
    && echo "$SMART_TURN_SHA256  $SMART_TURN.part" | sha256sum -c --quiet; then
    mv "$SMART_TURN.part" "$SMART_TURN"
    log "Smart Turn model ready"
else
    rm -f "$SMART_TURN.part"
    warn "Smart Turn download failed — the assistant will fall back to the silence timer"
fi

# -----------------------------------------------------------------------------
# 5c. Whisper speech-to-text model (so the assistant never needs the internet)
# -----------------------------------------------------------------------------
STT_MODEL=$(python3 -c "from nuc_assistant.config import load_config; print(load_config('$CONFIG')['stt']['model'])" \
    2>/dev/null || echo "small")
info "Downloading Whisper $STT_MODEL model..."
python3 -c "
import sys
from faster_whisper import WhisperModel
WhisperModel(sys.argv[1], device='cpu', compute_type='int8')
" "$STT_MODEL" && log "Whisper model ready" \
    || warn "Whisper download failed — run setup.sh again with internet before going offline"

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
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    log "Created config.yaml from config.example.yaml"
fi
warn "If the default devices are wrong, set audio.input_device / output_device in config.yaml"

# -----------------------------------------------------------------------------
# 7. Test Ollama is responding
# -----------------------------------------------------------------------------
info "Testing Ollama API..."
RESPONSE=$(curl -s http://localhost:11434/api/generate \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Reply with just the word ready.\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":20}}" \
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
echo "  1. Edit config:        nano config.yaml"
echo "  2. Test the pipeline:  ./test_pipeline.sh"
echo "  3. Try it out:         source venv/bin/activate && python3 assistant.py --debug"
echo "  4. Install as service: sudo ./install_service.sh"
echo ""
