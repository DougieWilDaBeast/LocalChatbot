"""Command-line entry point and the main conversation loop."""

import argparse
import sys
import time

from .audio import play_beep, record_until_silence
from .config import ConfigError, load_config
from .llm import LLMClient
from .stt import Transcriber
from .tts import Speaker
from .wakeword import WakeWordDetector


def main():
    parser = argparse.ArgumentParser(description="NUC Offline Voice Assistant")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        sys.exit(f"[✗] {args.config} not found. Create it with: cp config.example.yaml config.yaml")
    except ConfigError as e:
        sys.exit(f"[✗] {args.config}: {e}")

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
                threshold=cfg["audio"].get("speech_threshold", 500),
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

            # Get the LLM response and speak it sentence by sentence
            speaker.speak_stream(llm.chat_stream(text), debug=debug)

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
