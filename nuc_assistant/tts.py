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


class Speaker:
    def __init__(self, cfg: dict):
        tts = cfg["tts"]
        self.model_path = tts["model"]
        self.config_path = tts["config"]
        self.output_device = cfg["audio"]["output_device"]

        # A voice assistant that can't speak is no use, so stop with a fix-it message
        if not PIPER_AVAILABLE:
            raise RuntimeError("piper-tts isn't installed. Run setup.sh, or: pip install piper-tts")
        for path in (self.model_path, self.config_path):
            if not Path(path).exists():
                raise RuntimeError(f"Piper voice file not found at {path}. Run setup.sh to download it")

        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)
        print("[✓] Piper TTS ready")

    def speak(self, text: str, stop=None):
        """Convert text to speech and play through speaker. Playback is cut
        off if `stop` (a threading.Event) is set."""
        if not text:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            try:
                with wave.open(tmp_path, "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
            except Exception as e:  # skip a sentence Piper can't say, carry on with the rest
                print(f"[!] Piper couldn't say {text!r}: {e}")
                return
            if stop is None or not stop.is_set():
                # A speaker error propagates, so the main loop can see the device is gone
                play_audio_file(tmp_path, device=self.output_device, stop=stop)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def speak_stream(self, sentences, debug: bool = False, stop=None) -> list[str]:
        """Speak sentences as they arrive. The LLM keeps generating in a
        background thread while the current sentence plays. Setting `stop`
        (a threading.Event) ends playback straight away. Returns the sentences
        the user heard, all or part of, for the conversation memory."""
        said = []
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
        while stop is None or not stop.is_set():
            try:
                item = pending.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            if debug:
                print(f"[Assistant]: {item}")
            said.append(item)
            self.speak(item, stop=stop)
        return said
