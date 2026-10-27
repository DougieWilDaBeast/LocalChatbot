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

# Read settings from config.yaml (falls back to config.example.yaml)
CONFIG="config.yaml"
[ -f "$CONFIG" ] || { info "config.yaml not found, using config.example.yaml"; CONFIG="config.example.yaml"; }
cfg() {
    python3 -c "import sys, yaml; c = yaml.safe_load(open('$CONFIG')); print(eval('c' + sys.argv[1]))" "$1"
}
MODEL=$(cfg "['model']['name']")
OLLAMA_HOST=$(cfg "['model'].get('host', 'http://localhost:11434')")
STT_MODEL=$(cfg "['stt']['model']")
VOICE_MODEL=$(cfg "['tts']['model']")
VOICE_CONFIG=$(cfg "['tts']['config']")

# --- 1. Ollama ---
info "Testing Ollama ($MODEL)..."
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    RESPONSE=$(curl -s "$OLLAMA_HOST/api/generate" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"Say the single word: working\",\"stream\":false}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','').strip())" || true)
    if [ -n "$RESPONSE" ]; then
        pass "Ollama: '$RESPONSE'"
    else
        fail "Ollama responded but gave no output — is the model pulled? ollama pull $MODEL"
    fi
else
    fail "Ollama not running — sudo systemctl start ollama"
fi

# --- 2. faster-whisper ---
info "Testing faster-whisper ($STT_MODEL)..."
python3 -c "
from faster_whisper import WhisperModel
model = WhisperModel('$STT_MODEL', device='cpu', compute_type='int8')
print('faster-whisper loaded successfully')
" && pass "faster-whisper: model loads" || fail "faster-whisper: model failed to load"

# --- 3. Piper TTS ---
info "Testing Piper TTS..."
if [ -f "$VOICE_MODEL" ]; then
    rm -f /tmp/test_tts.wav
    python3 -c "
import wave
from piper import PiperVoice
voice = PiperVoice.load('$VOICE_MODEL', config_path='$VOICE_CONFIG')
with wave.open('/tmp/test_tts.wav', 'wb') as f:
    voice.synthesize_wav('Assistant is ready.', f)
" || true
    if [ -s "/tmp/test_tts.wav" ]; then
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
