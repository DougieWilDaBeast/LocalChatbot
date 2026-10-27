# NUC Offline Voice Assistant

A fully offline voice assistant running on an Intel NUC with local LLM inference, wake word detection, speech-to-text, and text-to-speech. No cloud. No subscription. No data leaving the device.

Designed as a local-first voice assistant template you can personalize after cloning.

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
| Orchestration    | Python 3.12                                                 | Pipeline glue                       |
| Auto-start       | systemd                                                     | Runs as a service on boot           |

---

## Architecture

```
[Microphone]
     │
     ▼
[openWakeWord] ──── silence ────► (keep listening)
     │ wake word detected
     ▼
[faster-whisper]
     │ transcribed text
     ▼
[Ollama / Gemma]
     │ response text
     ▼
[Piper TTS]
     │ audio
     ▼
[Speaker]
```

---

## Project Structure

```
local-chatbot/
├── README.md
├── config.example.yaml       # Copy to config.yaml and personalize
├── config.yaml               # Local runtime config (do not commit private values)
├── requirements.txt
├── assistant.py              # Main pipeline orchestrator
├── setup.sh                  # Full environment setup
├── test_pipeline.sh          # Test each layer individually
├── install_service.sh        # Install systemd service
├── nuc-assistant.service     # systemd template (install script fills local values)
└── HARDWARE.md               # Hardware notes and tuning guidance
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

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

- `assistant.name`
- `assistant.system_prompt` placeholders: `[YOUR_NAME]`, `[YOUR_LOCATION]`, and your short profile text
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

## Customisation

The system prompt is configured in `config.yaml` (created from `config.example.yaml`). Edit it to personalize the assistant for whoever is using it: their name, location, interests, and preferred response style.

---

## Roadmap

- [x] Base pipeline (STT → LLM → TTS)
- [x] Wake word detection
- [x] systemd auto-start
- [x] Conversation memory (rolling context window)
- [ ] Stream LLM replies to TTS sentence by sentence
- [ ] Intel Iris Xe LLM offload via Ollama's Vulkan backend
- [ ] OpenVINO GenAI Whisper inference on the iGPU
- [ ] Physical button trigger (GPIO or USB)
- [ ] 3D printed case (Bambu Lab X1 Carbon)
- [ ] Web UI for config editing

---

## License

GPL-3.0. See [LICENSE](LICENSE). This matches [Piper TTS](https://github.com/OHF-Voice/piper1-gpl), which is also GPL-3.0.
