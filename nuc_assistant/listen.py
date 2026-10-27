"""Listening for the user's turn: voice activity detection (Silero, bundled with
openWakeWord), end-of-turn detection (Smart Turn), and listening on while the
assistant transcribes and talks, for barge-in."""

import threading
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from .audio import record_until_silence

try:
    from openwakeword.vad import VAD
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    from faster_whisper.feature_extractor import FeatureExtractor
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

SAMPLE_RATE = 16000
FRAME = 480          # 30 ms, Silero's recommended frame size
FRAME_MS = 30
PRE_ROLL_FRAMES = 10  # keep 300 ms before speech starts, so the first word isn't clipped


def load_vad():
    """Silero VAD, or None if openWakeWord's copy of the model isn't there."""
    if not VAD_AVAILABLE:
        return None
    try:
        return VAD()
    except Exception as e:  # model file missing: run openwakeword.utils.download_models()
        print(f"[!] Silero VAD unavailable ({e})")
        return None


class SmartTurn:
    """Pipecat's Smart Turn v3 model: is the user finished, or just pausing?
    Reads the whole turn so far (up to the last 8 s) and returns P(finished)."""

    SECONDS = 8

    def __init__(self, model_path: str):
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"])
        # Same log-mel features as transformers' WhisperFeatureExtractor(chunk_length=8)
        self.features = FeatureExtractor(chunk_length=self.SECONDS)

    def probability(self, audio: np.ndarray) -> float:
        n = self.SECONDS * SAMPLE_RATE
        x = audio.astype(np.float32)[-n:] / 32768.0
        x = np.pad(x, (n - len(x), 0))  # audio sits at the end of the window
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        feats = self.features(x, padding=0)[None].astype(np.float32)
        return float(self.session.run(None, {"input_features": feats})[0][0][0])


class Listener:
    def __init__(self, cfg: dict):
        self.audio_cfg = cfg["audio"]
        conv = cfg["conversation"]
        self.input_device = self.audio_cfg["input_device"]
        self.vad_threshold = conv["vad_threshold"]
        self.pause_frames = conv["pause_ms"] // FRAME_MS
        self.max_pause_frames = conv["max_pause_ms"] // FRAME_MS
        self.silence_frames = self.audio_cfg["silence_threshold"] // FRAME_MS
        self.max_frames = self.audio_cfg["record_seconds"] * 1000 // FRAME_MS
        self.barge_in_frames = max(1, conv["barge_in_ms"] // FRAME_MS)

        self.vad = load_vad()
        self.monitor_vad = load_vad() if self.vad is not None else None
        self.barge_in = conv["barge_in"] and self.vad is not None
        if self.vad is None:
            print("[!] No voice activity detection: falling back to the silence timer, "
                  "with no barge-in or follow-up listening")
            self.barge_in = False

        self.smart_turn = None
        model = conv["smart_turn_model"]
        if conv["smart_turn"] and self.vad is not None:
            if not FEATURES_AVAILABLE:
                print("[!] faster-whisper missing, Smart Turn disabled")
            elif not Path(model).exists():
                print(f"[!] Smart Turn model not found at {model}, using the silence timer. "
                      "Run setup.sh to download it")
            else:
                self.smart_turn = SmartTurn(model)
                print("[✓] Smart Turn end-of-turn detection ready")

    def is_speech(self, vad, frame: np.ndarray) -> bool:
        return vad.predict(frame, frame_size=FRAME) >= self.vad_threshold

    def turn_finished(self, frames: list, silent: int) -> bool:
        """Decide, after `silent` quiet frames, whether the user has finished."""
        if self.smart_turn is None:
            return silent >= self.silence_frames
        if silent >= self.max_pause_frames:
            return True
        if silent == self.pause_frames:  # ask once per pause
            return self.smart_turn.probability(np.concatenate(frames)) > 0.5
        return False

    def record_turn(self, wait_seconds: float | None = None, initial=None):
        """Record one user turn and return it as int16 samples.

        wait_seconds: give up and return None if no speech starts within this
        time (used for the follow-up window). initial: frames already heard,
        e.g. the start of a barge-in, which continue this turn."""
        if self.vad is None:
            if wait_seconds is not None:
                return None
            return record_until_silence(
                sample_rate=SAMPLE_RATE,
                max_seconds=self.audio_cfg["record_seconds"],
                silence_ms=self.audio_cfg["silence_threshold"],
                threshold=self.audio_cfg["speech_threshold"],
                input_device=self.input_device,
            )

        self.vad.reset_states()
        frames = list(initial or [])
        started = bool(frames)
        pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        wait_frames = None if wait_seconds is None else int(wait_seconds * 1000 / FRAME_MS)
        silent = 0
        waited = 0

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            device=self.input_device, blocksize=FRAME) as stream:
            while True:
                frame = stream.read(FRAME)[0].flatten()
                speech = self.is_speech(self.vad, frame)
                if not started:
                    pre_roll.append(frame)
                    if speech:
                        started = True
                        frames.extend(pre_roll)
                        continue
                    waited += 1
                    if wait_frames is not None and waited >= wait_frames:
                        return None
                    if waited >= self.max_frames:  # nobody spoke at all
                        return np.concatenate(pre_roll)
                    continue

                frames.append(frame)
                silent = 0 if speech else silent + 1
                if len(frames) >= self.max_frames:
                    break
                if silent and self.turn_finished(frames, silent):
                    break

        return np.concatenate(frames)

    def monitor(self):
        """A SpeechMonitor for the time after a turn ends, or None without VAD."""
        return SpeechMonitor(self) if self.vad is not None else None


class SpeechMonitor:
    """Keeps listening after a turn has ended: while Whisper transcribes (the
    user may not have finished after all) and, for barge-in, while the
    assistant talks. Once the user has spoken for barge_in_ms, sets `heard`
    (which stops playback and the LLM), then keeps recording into `frames`
    until stopped, so nothing they say is lost."""

    def __init__(self, listener: Listener):
        self.listener = listener
        self.heard = threading.Event()
        self.frames = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def _run(self):
        listener = self.listener
        vad = listener.monitor_vad
        vad.reset_states()
        recent = deque(maxlen=PRE_ROLL_FRAMES + listener.barge_in_frames)
        run = 0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            device=listener.input_device, blocksize=FRAME) as stream:
            while not self._stop.is_set():
                frame = stream.read(FRAME)[0].flatten()
                if self.heard.is_set():
                    self.frames.append(frame)
                    continue
                recent.append(frame)
                run = run + 1 if listener.is_speech(vad, frame) else 0
                if run >= listener.barge_in_frames:
                    self.frames = list(recent)
                    self.heard.set()
