"""Text-to-speech with Piper."""

import queue
import tempfile
import threading
import wave
from pathlib import Path

from .audio import play_audio_file

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
        print("[✓] Piper TTS ready")

    def speak(self, text: str):
        """Convert text to speech and play through speaker."""
        if not text or not self.voice:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with wave.open(tmp_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            play_audio_file(tmp_path, device=self.output_device)

        except Exception as e:
            print(f"[!] Piper error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def speak_stream(self, sentences, debug: bool = False):
        """Speak sentences as they arrive. The LLM keeps generating in a
        background thread while the current sentence plays."""
        pending = queue.Queue()
        done = object()

        def produce():
            try:
                for sentence in sentences:
                    pending.put(sentence)
            except Exception as e:
                pending.put(e)
            pending.put(done)

        threading.Thread(target=produce, daemon=True).start()
        while (item := pending.get()) is not done:
            if isinstance(item, Exception):
                raise item
            if debug:
                print(f"[Assistant]: {item}")
            self.speak(item)
