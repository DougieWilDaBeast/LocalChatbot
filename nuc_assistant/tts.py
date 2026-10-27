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
        self._producer = None

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

    def speak(self, text: str, stop=None):
        """Convert text to speech and play through speaker. Playback is cut
        off if `stop` (a threading.Event) is set."""
        if not text or not self.voice:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with wave.open(tmp_path, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            if stop is None or not stop.is_set():
                play_audio_file(tmp_path, device=self.output_device, stop=stop)

        except Exception as e:
            print(f"[!] Piper error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def speak_stream(self, sentences, debug: bool = False, stop=None):
        """Speak sentences as they arrive. The LLM keeps generating in a
        background thread while the current sentence plays. Setting `stop`
        (a threading.Event) ends playback straight away."""
        # An interrupted reply's thread may still be saving it to the
        # conversation history; let it finish before the next reply starts.
        if self._producer is not None:
            self._producer.join()

        pending = queue.Queue()
        done = object()

        def produce():
            try:
                for sentence in sentences:
                    pending.put(sentence)
            except Exception as e:
                pending.put(e)
            pending.put(done)

        self._producer = threading.Thread(target=produce, daemon=True)
        self._producer.start()
        while stop is None or not stop.is_set():
            try:
                item = pending.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            if debug:
                print(f"[Assistant]: {item}")
            self.speak(item, stop=stop)
