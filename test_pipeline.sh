#!/bin/bash
# Test each layer of the pipeline individually, with the settings in config.yaml
# Run: ./test_pipeline.sh

cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILURES=0
pass() { echo -e "${GREEN}[✓] $1${NC}"; }
fail() { echo -e "${RED}[✗] $1${NC}"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}[→] $1${NC}"; }

echo ""
echo "=== NUC Assistant Pipeline Tests ==="
echo ""

source venv/bin/activate 2>/dev/null || { fail "Virtual environment not found. Run setup.sh first."; exit 1; }

# Read settings the way the assistant does: config.yaml (or the example) plus defaults
CONFIG="config.yaml"
[ -f "$CONFIG" ] || { info "config.yaml not found, using config.example.yaml"; CONFIG="config.example.yaml"; }
export CONFIG
python3 -c "from nuc_assistant.config import load_config; import os; load_config(os.environ['CONFIG'])" \
    || { fail "$CONFIG has a problem (see above)"; exit 1; }
cfg() {
    python3 -c "
import os, sys
from nuc_assistant.config import load_config
section, key = sys.argv[1].split('.')
print(load_config(os.environ['CONFIG'])[section][key])" "$1" 2>/dev/null
}
MODEL=$(cfg model.name)
OLLAMA_HOST=$(cfg model.host)

# --- 1. Ollama ---
info "Testing Ollama ($MODEL)..."
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    RESPONSE=$(curl -s "$OLLAMA_HOST/api/generate" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"Say the single word: working\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":20}}" \
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
info "Testing faster-whisper ($(cfg stt.model))..."
python3 -c "
import os
from nuc_assistant.config import load_config
from nuc_assistant.stt import Transcriber
Transcriber(load_config(os.environ['CONFIG']))
" && pass "faster-whisper: model loads" || fail "faster-whisper: model failed to load"

# --- 3. Piper TTS, played on audio.output_device ---
info "Testing Piper TTS..."
python3 -c "
import os
from nuc_assistant.config import load_config
from nuc_assistant.tts import Speaker
Speaker(load_config(os.environ['CONFIG'])).speak('Assistant is ready.')
" && pass "Piper TTS: spoke 'Assistant is ready.' (did you hear it?)" || fail "Piper TTS failed"

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
info "Check config.yaml if the wrong device is used"

# --- 5. Wake word, voice activity and Smart Turn models ---
info "Testing openWakeWord, Silero VAD and Smart Turn..."
python3 -c "
import os
from nuc_assistant.config import load_config
from nuc_assistant.listen import Listener
from nuc_assistant.wakeword import WakeWordDetector
cfg = load_config(os.environ['CONFIG'])
WakeWordDetector(cfg)
listener = Listener(cfg)
assert listener.vad is not None, 'Silero VAD missing'
" && pass "Listening models load" || fail "Listening models failed to load"

echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo "=== All tests passed ==="
    echo ""
    echo "Run the assistant:"
    echo "  source venv/bin/activate && python3 assistant.py --debug"
else
    echo "=== $FAILURES test(s) failed ==="
    exit 1
fi
echo ""
