# NUC Offline Voice Assistant

A fully offline voice assistant running on an Intel NUC with local LLM inference, wake word detection, speech-to-text, and text-to-speech. No cloud. No subscription. No data leaving the device.

Built for a family member in a rural location — plug it in and talk to it.

---

## Hardware

| Component | Spec |
|-----------|------|
| Device | Intel NUC 11th Gen |
| CPU | Intel Core i5-1135G7 @ 2.40GHz |
| RAM | 32GB |
| GPU | Intel Iris Xe Graphics (96 EUs) |
| OS | Ubuntu 24.04 LTS Server |

---

## Software Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Model Serving | [Ollama](https://ollama.ai) | Runs Gemma locally via REST API |
| LLM | Gemma 3 4B (Q4) or Gemma 4 E4B | Conversational intelligence |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Offline transcription (small model) |
| Wake Word | [openWakeWord](https://github.com/dscripka/openWakeWord) | Passive trigger detection |
| Text-to-Speech | [Piper TTS](https://github.com/rhasspy/piper) | Fast offline voice synthesis |
| Orchestration | Python 3.12 | Pipeline glue |
| Auto-start | systemd | Runs as a service on boot |
| GPU Acceleration | Intel oneAPI / OpenVINO | iGPU offload for faster inference |

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
nuc-assistant/
├── README.md
├── requirements.txt
├── assistant.py              # Main pipeline orchestrator
├── config/
│   └── config.yaml           # All tuneable settings
├── scripts/
│   ├── setup.sh              # Full environment setup
│   ├── install_openvino.sh   # Intel GPU acceleration setup
│   └── test_pipeline.sh      # Test each layer individually
├── systemd/
│   └── nuc-assistant.service # systemd unit file
└── docs/
    ├── HARDWARE.md           # NUC-specific notes
    ├── OPENVINO.md           # Intel GPU acceleration guide
    └── TROUBLESHOOTING.md    # Common issues
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nuc-assistant.git
cd nuc-assistant

# 2. Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Edit your config
nano config/config.yaml

# 4. Test the pipeline
./scripts/test_pipeline.sh

# 5. Start as a service
sudo ./scripts/install_service.sh
```

---

## Model Selection

| Model | RAM Required | Speed (CPU) | Quality | Recommended For |
|-------|-------------|-------------|---------|-----------------|
| Gemma 3 1B Q4 | ~1.5GB | Fast | Good | Very low-spec hardware |
| Gemma 3 4B Q4 | ~4GB | Comfortable | Great | **This build (default)** |
| Gemma 4 E4B Q4 | ~6GB | Comfortable | Excellent | This build (upgrade) |

With 32GB RAM this NUC runs Gemma 3 4B effortlessly with significant headroom.

---

## Intel GPU Acceleration

The i5-1135G7 includes Intel Iris Xe Graphics which supports inference offloading via Intel oneAPI and OpenVINO. See [docs/OPENVINO.md](docs/OPENVINO.md) for the full acceleration setup.

Expected speedup: **2–4x** over CPU-only inference on the 4B model.

---

## Customisation

The system prompt is configured in `config/config.yaml`. Edit it to personalise the assistant for whoever is using it — their name, location, interests, and preferred response style.

---

## Roadmap

- [x] Base pipeline (STT → LLM → TTS)
- [x] Wake word detection
- [x] systemd auto-start
- [ ] Intel Iris Xe GPU offload via oneAPI
- [ ] OpenVINO-optimised Whisper inference
- [ ] Conversation memory (rolling context window)
- [ ] Physical button trigger (GPIO or USB)
- [ ] 3D printed case (Bambu Lab X1 Carbon)
- [ ] Web UI for config editing
