#!/bin/bash
# Test each layer of the pipeline individually
# Run: ./test_pipeline.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[✓] $1${NC}"; }
fail() { echo -e "${RED}[✗] $1${NC}"; }
info() { echo -e "${YELLOW}[→] $1${NC}"; }

echo ""
echo "=== NUC Assistant Pipeline Tests ==="
echo ""

source venv/bin/activate 2>/dev/null || { fail "Virtual environment not found. Run setup.sh first."; exit 1; }

# --- 1. Ollama ---
info "Testing Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    RESPONSE=$(curl -s http://localhost:11434/api/generate \
        -d '{"model":"gemma3:4b","prompt":"Say the single word: working","stream":false}' \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','').strip())")
    if [ -n "$RESPONSE" ]; then
        pass "Ollama: '$RESPONSE'"
    else
        fail "Ollama responded but empty output"
    fi
else
    fail "Ollama not running — sudo systemctl start ollama"
fi

# --- 2. faster-whisper ---
info "Testing faster-whisper import..."
python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny', device='cpu', compute_type='int8')
print('faster-whisper loaded successfully')
" && pass "faster-whisper: model loads" || fail "faster-whisper: import failed"

# --- 3. Piper TTS ---
info "Testing Piper TTS..."
if [ -f "/usr/local/bin/piper" ]; then
    VOICE_MODEL="models/piper/en_GB-alba-medium.onnx"
    VOICE_CONFIG="models/piper/en_GB-alba-medium.onnx.json"
    if [ -f "$VOICE_MODEL" ]; then
        echo "Assistant is ready." | piper \
            --model "$VOICE_MODEL" \
            --config "$VOICE_CONFIG" \
            --output_file /tmp/test_tts.wav 2>/dev/null
        if [ -f "/tmp/test_tts.wav" ]; then
            pass "Piper TTS: wav file generated"
            info "Playing test audio..."
            python3 -c "
import soundfile as sf
import sounddevice as sd
data, sr = sf.read('/tmp/test_tts.wav')
sd.play(data, sr)
sd.wait()
print('Audio played')
" && pass "Piper TTS: audio played" || fail "Piper TTS: audio playback failed"
        else
            fail "Piper TTS: no output file generated"
        fi
    else
        fail "Piper voice model not found at $VOICE_MODEL — run setup.sh"
    fi
else
    fail "Piper binary not found at /usr/local/bin/piper — run setup.sh"
fi

# --- 4. Audio devices ---
info "Listing audio devices..."
python3 -c "
import sounddevice as sd
print('Input devices:')
for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] > 0:
        print(f'  [{i}] {d[\"name\"]}')
print('Output devices:')
for i, d in enumerate(sd.query_devices()):
    if d['max_output_channels'] > 0:
        print(f'  [{i}] {d[\"name\"]}')
"
pass "Audio devices listed above — check config.yaml if wrong device is used"

# --- 5. openWakeWord ---
info "Testing openWakeWord..."
python3 -c "
try:
    from openwakeword.model import Model
    print('openWakeWord import OK')
except ImportError as e:
    print(f'openWakeWord not available: {e}')
" && pass "openWakeWord available" || fail "openWakeWord not available"

echo ""
echo "=== Test Complete ==="
echo ""
echo "If all tests pass, run the assistant:"
echo "  source venv/bin/activate && python3 assistant.py --debug --no-wake-word"
echo ""
