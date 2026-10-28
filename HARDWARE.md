# Hardware Notes — Intel NUC i5-1135G7

## Specs

|     |                                                      |
| --- | ---------------------------------------------------- |
| CPU | Intel Core i5-1135G7 @ 2.40GHz (4 cores / 8 threads) |
| RAM | 32GB DDR4                                            |
| GPU | Intel Iris Xe Graphics (80 Execution Units)          |
| TDP | 28W (low idle power — good for always-on use)        |

---

## Model Performance

Measured so far in a 4-core cloud VM with no GPU (the NUC's own numbers are still to come):

| Measurement                                          | Result               |
| ---------------------------------------------------- | -------------------- |
| Gemma 4 E4B generation speed (CPU)                   | about 3.4 tokens/sec |
| Gemma 4 E4B memory once loaded (4096 context)        | about 7.5 GB         |
| End of a short question to the assistant starting to speak | 4.5 to 8 s (shared CPU) |
| Talking over the reply to playback stopping          | about 0.7 s          |
| A 500-token reply, start to finish                   | about 150 s          |

Because only the first sentence has to be ready before speaking starts, the wait doesn't grow with the length of the answer. Gemma 3 4B (3.3 GB download) is the lighter option if replies feel slow. Whether the iGPU helps is not measured yet.

---

## Intel Iris Xe Notes

- 80 Execution Units on the i5-1135G7
- Shares system RAM (unified memory) — no dedicated VRAM
- Supported by Intel oneAPI, OpenCL, Vulkan, and OpenVINO
- faster-whisper (CTranslate2) has no Intel GPU backend; Whisper on the iGPU needs OpenVINO GenAI's Whisper pipeline instead
- Ollama has an experimental Vulkan backend that can offload the LLM to the iGPU without building llama.cpp yourself

---

## Ubuntu 24.04 Audio Setup

If audio is not working out of the box:

```bash
# Check PipeWire is running (Ubuntu 24.04's sound server; setup.sh installs it on Server)
systemctl --user status pipewire pipewire-pulse wireplumber
wpctl status

# List ALSA devices
aplay -l
arecord -l

# Test speaker output
speaker-test -t wav -c 2

# Test microphone input (Ctrl+C to stop)
arecord -f cd -d 5 /tmp/test.wav && aplay /tmp/test.wav
```

## Headless Boot

For headless deployments (no monitor), use SSH for configuration:

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
