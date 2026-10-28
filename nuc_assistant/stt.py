"""Speech-to-text with faster-whisper."""

import logging

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


class Transcriber:
    """Turns a recorded turn (16 kHz int16) into text."""

    def __init__(self, cfg: dict):
        stt = cfg["stt"]
        log.info(f"[→] Loading Whisper {stt['model']} model...")
        options = dict(device=stt["device"], compute_type=stt["compute_type"])
        try:
            # setup.sh downloads the model, so normally nothing leaves the machine
            self.model = WhisperModel(stt["model"], local_files_only=True, **options)
        except Exception:
            log.info(f"[→] Whisper {stt['model']} isn't downloaded yet, fetching it (needs internet once)")
            self.model = WhisperModel(stt["model"], **options)
        self.language = stt["language"]
        log.info("[✓] Whisper ready")

    def transcribe(self, audio: np.ndarray) -> str:
        # faster-whisper expects float32 normalised audio
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio_f32,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()
