# Hardware Notes — Intel NUC i5-1135G7

## Specs

| | |
|--|--|
| CPU | Intel Core i5-1135G7 @ 2.40GHz (4 cores / 8 threads) |
| RAM | 32GB DDR4 |
| GPU | Intel Iris Xe Graphics (96 Execution Units) |
| TDP | 28W (low idle power — good for always-on use) |

---

## Model Performance Estimates

| Model | RAM Used | Tokens/sec (CPU) | Tokens/sec (iGPU) |
|-------|----------|-------------------|-------------------|
| Gemma 3 1B Q4 | ~1.5GB | ~15–25 | ~30–45 |
| Gemma 3 4B Q4 | ~4GB | ~5–10 | ~12–20 |
| Gemma 4 E4B Q4 | ~6GB | ~4–8 | ~10–18 |

With 32GB RAM there is no memory pressure at any of these sizes.
5–10 tokens/sec on the 4B model produces a conversational response in 2–4 seconds — acceptable for voice.

---

## Intel Iris Xe Notes

- 96 Execution Units on the i5-1135G7
- Shares system RAM (unified memory) — no dedicated VRAM
- Supported by Intel oneAPI, OpenCL, and OpenVINO
- OpenVINO acceleration for Whisper STT gives the most practical speedup
- Full Ollama iGPU offload requires building llama.cpp with SYCL support (advanced)

---

## Ubuntu 24.04 Audio Setup

If audio is not working out of the box:

```bash
# Check PulseAudio is running
systemctl --user status pulseaudio

# List ALSA devices
aplay -l
arecord -l

# Test speaker output
speaker-test -t wav -c 2

# Test microphone input (Ctrl+C to stop)
arecord -f cd -d 5 /tmp/test.wav && aplay /tmp/test.wav
```

## Headless Boot

Since this NUC runs headless (no monitor), SSH in for all configuration:

```bash
# Enable SSH on Ubuntu during install or after:
sudo apt install openssh-server
sudo systemctl enable ssh

# From your machine:
ssh user@NUC_IP_ADDRESS
```

Find the NUC's IP:
```bash
ip addr show | grep inet
# or from your router's admin page
```
