"""Command-line entry point and the main conversation loop."""

import argparse
import logging
import sys
import time

import requests

from . import log as logs
from .audio import play_beep
from .config import ConfigError, load_config
from .listen import Listener
from .llm import LLMClient
from .stt import Transcriber
from .tts import Speaker
from .wakeword import WakeWordDetector

log = logging.getLogger(__name__)

# After this many failed turns in a row (e.g. the mic was unplugged), exit so
# systemd restarts the assistant, which also makes PortAudio look for devices again
MAX_FAILURES = 5
SORRY = "Sorry, I couldn't get an answer just now. Please try again."


def main():
    parser = argparse.ArgumentParser(description="NUC Offline Voice Assistant")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    args = parser.parse_args()
    logs.setup(args.debug)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        sys.exit(f"[✗] {args.config} not found. Create it with: cp config.example.yaml config.yaml")
    except (ConfigError, OSError) as e:
        sys.exit(f"[✗] {args.config}: {e}")

    if args.debug:
        cfg["behaviour"]["debug_mode"] = True
    if args.no_wake_word:
        cfg["wake_word"]["enabled"] = False

    logs.setup(cfg["behaviour"]["debug_mode"])

    log.info("==============================================")
    log.info(f"  NUC Offline Voice Assistant: {cfg['assistant']['name']}")
    log.info(f"  Model: {cfg['model']['name']}")
    log.info(f"  Wake word: {'enabled' if cfg['wake_word']['enabled'] else 'disabled'}")
    log.info("==============================================")

    try:
        run(cfg)
    except KeyboardInterrupt:
        log.info("[Shutting down]")


def run(cfg: dict):
    """Load every component, then hold conversations until stopped."""
    out = cfg["audio"]["output_device"]
    try:
        transcriber = Transcriber(cfg)
        llm = LLMClient(cfg)
        speaker = Speaker(cfg)
        wake_detector = WakeWordDetector(cfg)
        listener = Listener(cfg)

        # Signal ready
        if cfg["behaviour"]["startup_sound"]:
            play_beep(800, 0.1, device=out)
            time.sleep(0.05)
            play_beep(1000, 0.1, device=out)
    except Exception as e:
        log.debug("Startup failed", exc_info=True)
        sys.exit(f"[✗] Couldn't start: {e}")
    follow_up = cfg["conversation"]["follow_up_seconds"]

    log.info("[Ready]")

    # Main conversation loop
    audio = None  # a turn already heard: a barge-in or a follow-up question
    failures = 0
    while True:
        try:
            if audio is None:
                # Wait for trigger
                wake_detector.wait_for_wake_word()

                # Signal that we're listening
                play_beep(1200, 0.08, device=out)
                log.debug("[Recording...]")
                audio = listener.record_turn()

            # Keep listening while Whisper transcribes: if the user carries on
            # talking, the turn ended too early and continues. With barge-in,
            # the same monitor then listens while the reply plays.
            monitor = listener.monitor()
            if monitor is not None:
                monitor.start()
            try:
                text = transcriber.transcribe(audio)
                failures = 0  # the mic and Whisper are working

                if monitor is not None and monitor.heard.is_set():
                    monitor.stop()
                    if len(audio) < listener.max_samples:
                        log.debug("[Still talking, listening...]")
                        audio = listener.record_turn(initial=[audio, *monitor.frames])
                        continue
                    # Already at record_seconds (e.g. a TV is on): answer what we have,
                    # with a fresh monitor for barge-in
                    monitor = listener.monitor().start()
                audio = None

                if not text:
                    log.debug("[No speech detected]")
                    continue

                log.debug(f"[You]: {text}")

                # Get the LLM response and speak it sentence by sentence. With
                # barge-in, talking over it stops the reply and starts a new turn.
                stop = monitor.heard if listener.barge_in else None
                if stop is None and monitor is not None:
                    monitor.stop()
                try:
                    said = speaker.speak_stream(llm.chat_stream(text, stop=stop), stop=stop)
                    llm.remember(text, " ".join(said))
                except requests.RequestException as e:
                    # Ollama stopped, timed out or failed: say so rather than go quiet
                    log.error(f"[Error]: no reply from Ollama: {e}")
                    speaker.speak(SORRY)
                    continue
                if stop is not None and stop.is_set():
                    log.debug("[Interrupted, listening...]")
                    llm.cancel()
                    monitor.stop()
                    audio = listener.record_turn(initial=monitor.frames)
                    continue
            finally:
                if monitor is not None:
                    monitor.stop()

            # Follow-up window: keep listening briefly without the wake word
            if follow_up:
                log.debug(f"[Listening for a follow-up ({follow_up}s)...]")
                audio = listener.record_turn(wait_seconds=follow_up)

        except KeyboardInterrupt:
            log.info("[Shutting down]")
            play_beep(400, 0.2, device=out)
            break
        except EOFError:
            # input() with nothing to read: running as a service with the wake word off
            sys.exit("[✗] No keyboard to press Enter on. Turn wake_word.enabled on to run as a service")
        except Exception as e:
            failures += 1
            log.exception(f"[Error]: {e}")
            if failures >= MAX_FAILURES:
                sys.exit(f"[✗] {failures} errors in a row, stopping (systemd will restart it)")
            time.sleep(failures)
            audio = None
            continue
