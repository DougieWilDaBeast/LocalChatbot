"""Wake word detection with openWakeWord."""

import re
from pathlib import Path

import sounddevice as sd

# openWakeWord import with graceful fallback
try:
    from openwakeword.model import Model as WakeWordModel
    OWW_AVAILABLE = True
except ImportError:
    OWW_AVAILABLE = False


class WakeWordDetector:
    def __init__(self, cfg: dict):
        ww = cfg["wake_word"]
        self.enabled = ww["enabled"]
        if self.enabled and not OWW_AVAILABLE:
            raise RuntimeError("openWakeWord won't import. Rerun setup.sh, or turn "
                               "wake_word.enabled off to press Enter instead")
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
            # "hey_jarvis" or a path such as models/hey_jarvis_v0.1.onnx -> "hey jarvis"
            phrase = re.sub(r"_v\d.*$", "", Path(ww["model"]).stem).replace("_", " ")
            print(f"[✓] Wake word ready — say '{phrase}'")
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
                        # Clear buffered audio so the same utterance can't re-trigger
                        self.model.reset()
                        return
