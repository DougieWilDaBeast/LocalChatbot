"""Speech-to-text with faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel


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
        print("[✓] Whisper ready")

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
