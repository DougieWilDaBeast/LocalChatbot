"""Load config.yaml, fill in defaults and catch mistakes before anything starts."""

import copy

import yaml

# Used for any setting config.yaml leaves out. Matches config.example.yaml.
DEFAULTS = {
    "assistant": {},
    "model": {
        "host": "http://localhost:11434",
        "temperature": 0.7,
        "max_tokens": 500,
        "context_window": 4096,
        "think": False,
        "keep_alive": -1,
        "timeout": 300,
    },
    "stt": {
        "model": "small",
        "language": "en",
        "device": "cpu",
        "compute_type": "int8",
    },
    "wake_word": {
        "enabled": True,
        "model": "hey_jarvis",
        "threshold": 0.5,
    },
    "tts": {},
    "audio": {
        "input_device": None,
        "output_device": None,
        "sample_rate": 16000,
        "channels": 1,
        "record_seconds": 8,
        "silence_threshold": 500,
        "speech_threshold": 500,
    },
    "behaviour": {
        "startup_sound": True,
        "listening_indicator": True,
        "conversation_memory": 6,
        "debug_mode": False,
    },
}

# Settings with no sensible default
REQUIRED = [
    ("assistant", "system_prompt"),
    ("model", "name"),
    ("tts", "model"),
    ("tts", "config"),
]


class ConfigError(ValueError):
    pass


def merge_defaults(user: dict) -> dict:
    """Return DEFAULTS with the user's settings layered on top."""
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in (user or {}).items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def validate(cfg: dict):
    missing = [f"{s}.{k}" for s, k in REQUIRED if not cfg.get(s, {}).get(k)]
    if missing:
        raise ConfigError(f"missing required setting(s): {', '.join(missing)}")
    if not 0 <= cfg["wake_word"]["threshold"] <= 1:
        raise ConfigError("wake_word.threshold must be between 0 and 1")
    if cfg["model"]["max_tokens"] <= 0:
        raise ConfigError("model.max_tokens must be positive")
    if cfg["behaviour"]["conversation_memory"] < 0:
        raise ConfigError("behaviour.conversation_memory can't be negative")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = merge_defaults(yaml.safe_load(f))
    validate(cfg)
    return cfg
