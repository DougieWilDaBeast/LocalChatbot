"""Microphone recording and speaker playback."""

import numpy as np
import sounddevice as sd
import soundfile as sf


def play_audio_file(filepath: str, device=None, stop=None):
    """Play a .wav file through the speaker, resampling to 48kHz for compatibility.
    If `stop` (a threading.Event) is set during playback, cut it off."""
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
    if stop is not None and stop.wait(len(data) / samplerate):
        sd.stop()
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
