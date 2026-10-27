# NUC Offline Voice Assistant

A fully offline voice assistant running on an Intel NUC with local LLM inference, wake word detection, speech-to-text, and text-to-speech. No cloud. No subscription. No data leaving the device.

Designed as a local-first voice assistant template you can personalize after cloning.

A personal project by [Will Douglas](https://github.com/DougieWilDaBeast), built to see how far a fully local voice assistant can go on an ordinary mini PC.

---

## Hardware

| Component | Spec                            |
| --------- | ------------------------------- |
| Device    | Intel NUC 11th Gen              |
| CPU       | Intel Core i5-1135G7 @ 2.40GHz  |
| RAM       | 32GB                            |
| GPU       | Intel Iris Xe Graphics (96 EUs) |
| OS        | Ubuntu 24.04 LTS Server         |

---

## Software Stack

| Layer            | Tool                                                        | Purpose                             |
| ---------------- | ----------------------------------------------------------- | ----------------------------------- |
| Model Serving    | [Ollama](https://ollama.com)                                | Runs Gemma locally via REST API     |
| LLM              | Gemma 4 E4B (default) or Gemma 3 4B                         | Conversational intelligence         |
| Speech-to-Text   | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Offline transcription (small model) |
| Wake Word        | [openWakeWord](https://github.com/dscripka/openWakeWord)    | Passive trigger detection           |
| Text-to-Speech   | [Piper TTS](https://github.com/OHF-Voice/piper1-gpl)        | Fast offline voice synthesis        |
| Voice Activity   | [Silero VAD](https://github.com/snakers4/silero-vad) (bundled with openWakeWord) | Hears when you start and stop talking |
| End of Turn      | [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) | Tells a pause from a finished question |
| Orchestration    | Python 3.12                                                 | Pipeline glue                       |
| Auto-start       | systemd                                                     | Runs as a service on boot           |

---

## Architecture

```mermaid
flowchart TD
    Mic([Microphone]) --> Wake[openWakeWord<br/>listens for the wake word]
    Wake -- silence --> Wake
    Wake -- wake word heard --> Turn[Silero VAD + Smart Turn<br/>waits until you finish]
    Turn -- your question --> STT[faster-whisper<br/>speech to text]
    STT -- transcribed text --> LLM[Ollama + Gemma<br/>local LLM]
    LLM -- reply, streamed one sentence at a time --> TTS[Piper<br/>text to speech]
    TTS --> Speaker([Speaker])
    Speaker -- you talk over it, or ask a follow-up within 5 s --> Turn
```

Every step runs on the NUC itself. Replies are streamed to Piper a sentence at a time, so the assistant starts speaking before the LLM has finished thinking.

---

## Project Structure

```
LocalChatbot/
├── README.md
├── config.example.yaml       # Copy to config.yaml and personalize
├── config.yaml               # Local runtime config (do not commit private values)
├── requirements.txt
├── requirements-dev.txt      # pytest and ruff
├── assistant.py              # Entry point: python3 assistant.py
├── nuc_assistant/            # The assistant itself
│   ├── main.py               #   Main conversation loop and command-line options
│   ├── config.py             #   Loads config.yaml, fills defaults, checks values
│   ├── audio.py              #   Mic recording and speaker playback
│   ├── listen.py             #   Voice activity, end of turn (Smart Turn), barge-in
│   ├── wakeword.py           #   Wake word detection (openWakeWord)
│   ├── stt.py                #   Speech-to-text (faster-whisper)
│   ├── llm.py                #   Ollama chat, streamed sentence by sentence
│   └── tts.py                #   Text-to-speech (Piper)
├── setup.sh                  # Full environment setup
├── test_pipeline.sh          # Test each layer individually
├── tests/                    # Unit tests (no mic, speaker or models needed)
├── install_service.sh        # Install systemd service
├── nuc-assistant.service     # systemd template (install script fills local values)
└── HARDWARE.md               # Hardware notes and tuning guidance
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/DougieWilDaBeast/LocalChatbot.git
cd LocalChatbot

# 2. Run the setup script
./setup.sh

# 3. Create your local config from template
cp config.example.yaml config.yaml

# 4. Edit your local config
nano config.yaml

# 5. Test the pipeline
./test_pipeline.sh

# 6. Start as a service
sudo ./install_service.sh
```

## First-Clone Personalization Checklist

After cloning, update these values in `config.yaml`:

- `assistant.name`, and the same name in the first line of `assistant.system_prompt`
- `assistant.system_prompt`: optionally add a line about who the assistant is talking to (an example is in the comments above it)
- `audio.input_device` and `audio.output_device` (if defaults are not correct)
- `model.name` (optional if you want a different local Ollama model)

---

## Model Selection

| Model          | RAM Required | Speed (CPU) | Quality   | Recommended For          |
| -------------- | ------------ | ----------- | --------- | ------------------------ |
| Gemma 3 1B Q4  | ~1.5GB       | Fast        | Good      | Very low-spec hardware   |
| Gemma 3 4B Q4  | ~4GB         | Comfortable | Great     | Lighter alternative      |
| Gemma 4 E4B Q4 | ~6GB         | Comfortable | Excellent | **This build (default)** |

With 32GB RAM this NUC runs either model with significant headroom. Change `model.name` in `config.yaml` and `ollama pull` the model to switch.

---

## Intel GPU Acceleration

Everything currently runs on the CPU. The i5-1135G7's Iris Xe iGPU can be used for acceleration, and this is on the roadmap: Ollama's Vulkan backend for the LLM, and OpenVINO GenAI for Whisper (faster-whisper itself has no Intel GPU backend). See [HARDWARE.md](HARDWARE.md) for notes.

---

## Conversation: barge-in, end of turn and follow-ups

The assistant listens while it talks. If you speak over it for about 300 ms it stops mid-sentence, cancels the rest of the reply, and treats what you said as your next question. It decides you've finished speaking with the Smart Turn model, so a short pause mid-sentence doesn't cut you off, and after each reply it keeps listening for 5 seconds so a follow-up doesn't need the wake word. All of this is under `conversation:` in `config.yaml`.

Barge-in needs the microphone not to hear the assistant's own voice. With a headset or a speakerphone that cancels echo, nothing else is needed. With a separate speaker and mic, turn on echo cancellation in PipeWire (Ubuntu 24.04's sound server), then make the new echo-cancelled microphone the default input in Settings → Sound:

```bash
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat > ~/.config/pipewire/pipewire.conf.d/echo-cancel.conf <<'CONF'
context.modules = [
  { name = libpipewire-module-echo-cancel
    args = { library.name = aec/libspa-aec-webrtc } }
]
CONF
systemctl --user restart pipewire pipewire-pulse
```

If the assistant still interrupts itself, set `conversation.barge_in: false`.

---

## Customisation

The system prompt is configured in `config.yaml` (created from `config.example.yaml`). Edit it to personalize the assistant for whoever is using it: their name, location, interests, and preferred response style.

---

## Development

Unit tests fake the microphone, speaker, Ollama and the speech models, so they run on any machine:

```bash
source venv/bin/activate
pip install -r requirements-dev.txt
ruff check .
pytest
```

GitHub Actions runs the same checks on every push and pull request.

---

## Roadmap

- [x] Base pipeline (STT → LLM → TTS)
- [x] Wake word detection
- [x] systemd auto-start
- [x] Conversation memory (rolling context window)
- [x] Stream LLM replies to TTS sentence by sentence
- [ ] Intel Iris Xe LLM offload via Ollama's Vulkan backend
- [ ] OpenVINO GenAI Whisper inference on the iGPU
- [ ] Physical button trigger (GPIO or USB)
- [ ] 3D printed case (Bambu Lab X1 Carbon)
- [ ] Web UI for config editing

---

## Future Work

### Full-duplex voice models (needs stronger hardware)

Full-duplex speech-to-speech models listen and talk at the same time, so they can be interrupted, make "mm-hm" sounds while you speak, and reply in a few hundred milliseconds. As of October 2026 the open options are Kyutai's [Moshi](https://github.com/kyutai-labs/moshi), NVIDIA's [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1) (built on Moshi), [MiniCPM-o 4.5](https://github.com/tc-mb/llama.cpp-omni) through the llama.cpp-omni fork, and NVIDIA's NemotronLabs VoiceChat ([paper](https://arxiv.org/abs/2609.21967)).

None of them is practical on the NUC. They are 7B to 9B models that target CUDA or Apple Metal, and none supports Intel GPUs. Moshi runs its 7B model about 12.5 times a second without pause, while the i5-1135G7 manages a few tokens a second on the smaller Gemma 4 E4B, so it would run at least 4x slower than real time.

For now the plan is to make the existing pipeline feel more conversational instead: streaming replies, barge-in (talking over the assistant to stop it), smarter end-of-turn detection, and a short follow-up window without the wake word. Revisit full-duplex if the assistant moves to a machine with an NVIDIA GPU with 16 GB or more of VRAM, or if a small full-duplex model with a CPU or Vulkan runtime appears.

Other things worth testing later:

- [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B): a small turn-based speech-to-speech model with CPU GGUF builds, as an experimental mode.
- [Kyutai Pocket TTS](https://kyutai.org/tts/): a 100M-parameter streaming TTS that runs in real time on a CPU, as a more natural alternative to Piper.
- Gemma 4's own audio input, which could replace faster-whisper and save loading one model.

---

## License

GPL-3.0. See [LICENSE](LICENSE). This matches [Piper TTS](https://github.com/OHF-Voice/piper1-gpl), which is also GPL-3.0.
