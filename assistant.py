#!/usr/bin/env python3
"""
NUC Offline Voice Assistant
Intel NUC i5-1135G7 | 32GB RAM | Intel Iris Xe

Pipeline:
  openWakeWord → faster-whisper → Ollama/Gemma → Piper TTS → Speaker

Run:
  python3 assistant.py
  python3 assistant.py --debug
  python3 assistant.py --no-wake-word   (skip wake word, always listening)
"""

import argparse
import io
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import yaml
from faster_whisper import WhisperModel

# openWakeWord import with graceful fallback
try:
    from openwakeword.model import Model as WakeWordModel
    OWW_AVAILABLE = True
except ImportError:
    OWW_AVAILABLE = False
    print("[!] openWakeWord not available — running without wake word detection")


# =============================================================================
# Config
# =============================================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Audio Utilities
# =============================================================================

def play_audio_file(filepath: str, device=None):
    """Play a .wav file through the speaker, resampling to 48kHz for compatibility."""
    data, samplerate = sf.read(filepath)
    target_rate = 48000
    if samplerate != target_rate:
        # Resample using linear interpolation
        duration = len(data) / samplerate
        target_len = int(duration * target_rate)
        indices = np.linspace(0, len(data) - 1, target_len)
        if data.ndim == 1:
            data = np.interp(indices, np.arange(len(data)), data)
        else:
            data = np.column_stack([
                np.interp(indices, np.arange(len(data)), data[:, ch])
                for ch in range(data.shape[1])
            ])
        samplerate = target_rate
    sd.play(data.astype(np.float32), samplerate, device=device)
    sd.wait()


def play_beep(frequency: int = 800, duration: float = 0.15, device=None):
    """Play a short tone to signal state changes."""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.3
    fade = np.linspace(0, 1, int(sample_rate * 0.01))
    tone[:len(fade)] *= fade
    tone[-len(fade):] *= fade[::-1]
    sd.play(tone.astype(np.float32), sample_rate, device=device)
    sd.wait()


def record_until_silence(
    sample_rate: int = 16000,
    max_seconds: int = 8,
    silence_ms: int = 800,
    threshold: int = 500,
    input_device=None,
) -> np.ndarray:
    """Record audio until silence is detected or max duration reached."""
    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    silence_chunks = int(silence_ms / 100)
    max_chunks = int(max_seconds * 10)

    audio_chunks = []
    silent_count = 0
    has_speech = False

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=input_device,
        blocksize=chunk_size,
    ) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            audio_chunks.append(chunk.copy())
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            if rms > threshold:
                has_speech = True
                silent_count = 0
            elif has_speech:
                silent_count += 1
                if silent_count >= silence_chunks:
                    break

    return np.concatenate(audio_chunks, axis=0).flatten()


# =============================================================================
# Speech-to-Text
# =============================================================================

class Transcriber:
    def __init__(self, cfg: dict):
        stt = cfg["stt"]
        print(f"[→] Loading Whisper {stt['model']} model...")
        self.model = WhisperModel(
            stt["model"],
            device=stt["device"],
            compute_type=stt["compute_type"],
        )
        self.language = stt["language"]
        print(f"[✓] Whisper ready")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        # faster-whisper expects float32 normalised audio
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio_f32,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()


# =============================================================================
# LLM (Ollama)
# =============================================================================

class LLMClient:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]["name"]
        self.temperature = cfg["model"]["temperature"]
        self.max_tokens = cfg["model"]["max_tokens"]
        self.system_prompt = cfg["assistant"]["system_prompt"].strip()
        self.memory_size = cfg["behaviour"]["conversation_memory"]
        self.debug = cfg["behaviour"].get("debug_mode", False)
        self.history = deque(maxlen=self.memory_size * 2)  # user+assistant pairs
        self.base_url = "http://localhost:11434"

        print(f"[→] Connecting to Ollama ({self.model})...")
        self._check_connection()
        print(f"[✓] Ollama ready")

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model.split(":")[0] in m for m in models):
                print(f"[!] Model '{self.model}' not found. Run: ollama pull {self.model}")
        except requests.ConnectionError:
            print("[✗] Cannot connect to Ollama. Is it running? sudo systemctl start ollama")
            sys.exit(1)

    def chat(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(list(self.history))

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()

        data = r.json()
        reply = data.get("message", {}).get("content", "")
        if isinstance(reply, list):
            reply = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in reply
            )
        reply = str(reply).strip()

        if not reply:
            alt_reply = data.get("response") or data.get("content") or data.get("output_text")
            if isinstance(alt_reply, str):
                reply = alt_reply.strip()

        # Some models/versions can return empty content from /api/chat.
        # Fall back to /api/generate to avoid silent empty assistant replies.
        if not reply:
            if self.debug:
                print("[!] Empty /api/chat response, falling back to /api/generate")
            gen_payload = {
                "model": self.model,
                "prompt": user_text,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            gen_r = requests.post(
                f"{self.base_url}/api/generate",
                json=gen_payload,
                timeout=60,
            )
            gen_r.raise_for_status()
            gen_data = gen_r.json()
            reply = str(gen_data.get("response", "")).strip()

        if not reply:
            if self.debug:
                print(f"[!] No assistant content from Ollama chat payload: {data}")
            reply = "I'm here, but I couldn't generate a response just now. Please try again."

        self.history.append({"role": "assistant", "content": reply})
        return reply


# =============================================================================
# Text-to-Speech (Piper)
# =============================================================================

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("[!] piper-tts not installed — TTS will be unavailable")


class Speaker:
    def __init__(self, cfg: dict):
        tts = cfg["tts"]
        self.model_path = tts["model"]
        self.config_path = tts["config"]
        self.output_device = cfg["audio"]["output_device"]

        if not PIPER_AVAILABLE:
            print("[!] piper-tts package not installed. Run: pip install piper-tts")
            self.voice = None
            return

        if not Path(self.model_path).exists():
            print(f"[!] Piper voice model not found at {self.model_path}")
            self.voice = None
            return

        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)
        print(f"[✓] Piper TTS ready")

    def speak(self, text: str):
        """Convert text to speech and play through speaker."""
        if not text or not self.voice:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with wave.open(tmp_path, "wb") as wav_file:
                self.voice.synthesize(text, wav_file)

            play_audio_file(tmp_path, device=self.output_device)

        except Exception as e:
            print(f"[!] Piper error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# =============================================================================
# Wake Word
# =============================================================================

class WakeWordDetector:
    def __init__(self, cfg: dict):
        ww = cfg["wake_word"]
        self.enabled = ww["enabled"] and OWW_AVAILABLE
        self.threshold = ww["threshold"]
        self.sample_rate = 16000
        self.chunk_size = 1280  # Required by openWakeWord
        self.input_device = cfg["audio"]["input_device"]

        if self.enabled:
            print(f"[→] Loading wake word model ({ww['model']})...")
            self.model = WakeWordModel(
                wakeword_models=[ww["model"]],
                inference_framework="onnx",
            )
            print(f"[✓] Wake word ready — say '{ww['model'].replace('_', ' ')}'")
        else:
            print("[!] Wake word disabled — press Enter to trigger recording")

    def wait_for_wake_word(self):
        """Block until wake word is detected."""
        if not self.enabled:
            input("[Waiting] Press Enter to speak...")
            return

        print("[Listening for wake word...]")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.input_device,
            blocksize=self.chunk_size,
        ) as stream:
            while True:
                chunk, _ = stream.read(self.chunk_size)
                audio = chunk.flatten()
                prediction = self.model.predict(audio)

                for model_name, score in prediction.items():
                    if score >= self.threshold:
                        print(f"[Wake word detected: {score:.2f}]")
                        return


# =============================================================================
# Main Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NUC Offline Voice Assistant")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.debug:
        cfg["behaviour"]["debug_mode"] = True
    if args.no_wake_word:
        cfg["wake_word"]["enabled"] = False

    debug = cfg["behaviour"]["debug_mode"]

    print("")
    print("==============================================")
    print("  NUC Offline Voice Assistant")
    print(f"  Model: {cfg['model']['name']}")
    print(f"  Wake word: {'enabled' if cfg['wake_word']['enabled'] else 'disabled'}")
    print("==============================================")
    print("")

    # Initialise all components
    transcriber = Transcriber(cfg)
    llm = LLMClient(cfg)
    speaker = Speaker(cfg)
    wake_detector = WakeWordDetector(cfg)

    # Signal ready
    if cfg["behaviour"]["startup_sound"]:
        play_beep(800, 0.1)
        time.sleep(0.05)
        play_beep(1000, 0.1)

    print("\n[Ready]\n")

    # Main conversation loop
    while True:
        try:
            # Wait for trigger
            wake_detector.wait_for_wake_word()

            # Signal that we're listening
            play_beep(1200, 0.08)
            if debug:
                print("[Recording...]")

            # Record speech
            audio = record_until_silence(
                sample_rate=cfg["audio"]["sample_rate"],
                max_seconds=cfg["audio"]["record_seconds"],
                silence_ms=cfg["audio"]["silence_threshold"],
                input_device=cfg["audio"]["input_device"],
            )

            # Transcribe
            text = transcriber.transcribe(audio, cfg["audio"]["sample_rate"])

            if not text:
                if debug:
                    print("[No speech detected]")
                continue

            if debug:
                print(f"[You]: {text}")

            # Get LLM response
            response = llm.chat(text)

            if debug:
                print(f"[Assistant]: {response}")

            # Speak response
            speaker.speak(response)

        except KeyboardInterrupt:
            print("\n\n[Shutting down]")
            play_beep(400, 0.2)
            break
        except Exception as e:
            print(f"[Error]: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            time.sleep(1)
            continue


if __name__ == "__main__":
    main()
