"""Load config.yaml, fill in defaults and catch mistakes before anything starts."""

import copy
from numbers import Real
from pathlib import Path

import yaml

# Used for any setting config.yaml leaves out. Matches config.example.yaml.
DEFAULTS = {
    "assistant": {
        "name": "Jarvis",
    },
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
        "record_seconds": 8,
        "silence_threshold": 500,
        "speech_threshold": 500,
    },
    "conversation": {
        "barge_in": True,
        "barge_in_ms": 300,
        "vad_threshold": 0.5,
        "smart_turn": True,
        "smart_turn_model": "models/smart-turn/smart-turn-v3.2-cpu.onnx",
        "pause_ms": 500,
        "max_pause_ms": 2000,
        "follow_up_seconds": 5,
    },
    "behaviour": {
        "startup_sound": True,
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

# Settings older configs may still have that nothing reads any more
RETIRED = {
    ("audio", "sample_rate"),       # always 16 kHz: every speech model needs it
    ("audio", "channels"),          # always mono
    ("behaviour", "listening_indicator"),
}

NUMBER = "a number"
WHOLE = "a whole number"
TEXT = "text"
FLAG = "true or false"
DEVICE = "a device number, a device name or null"
DURATION = "a number of seconds, or text such as \"30m\""

# What each setting must be, and the range it must fall in
RULES = {
    ("assistant", "name"): (TEXT, None),
    ("assistant", "system_prompt"): (TEXT, None),
    ("model", "name"): (TEXT, None),
    ("model", "host"): (TEXT, None),
    ("model", "temperature"): (NUMBER, (0, 2)),
    ("model", "max_tokens"): (WHOLE, (1, None)),
    ("model", "context_window"): (WHOLE, (256, None)),
    ("model", "think"): (FLAG, None),
    ("model", "keep_alive"): (DURATION, None),
    ("model", "timeout"): (NUMBER, (1, None)),
    ("stt", "model"): (TEXT, None),
    ("stt", "language"): (TEXT, None),
    ("stt", "device"): (TEXT, None),
    ("stt", "compute_type"): (TEXT, None),
    ("wake_word", "enabled"): (FLAG, None),
    ("wake_word", "model"): (TEXT, None),
    ("wake_word", "threshold"): (NUMBER, (0.01, 1)),
    ("tts", "model"): (TEXT, None),
    ("tts", "config"): (TEXT, None),
    ("audio", "input_device"): (DEVICE, None),
    ("audio", "output_device"): (DEVICE, None),
    ("audio", "record_seconds"): (NUMBER, (1, 120)),
    ("audio", "silence_threshold"): (WHOLE, (100, None)),
    ("audio", "speech_threshold"): (NUMBER, (1, 32767)),
    ("conversation", "barge_in"): (FLAG, None),
    ("conversation", "barge_in_ms"): (WHOLE, (30, None)),
    ("conversation", "vad_threshold"): (NUMBER, (0.01, 1)),
    ("conversation", "smart_turn"): (FLAG, None),
    ("conversation", "smart_turn_model"): (TEXT, None),
    ("conversation", "pause_ms"): (WHOLE, (30, None)),
    ("conversation", "max_pause_ms"): (WHOLE, (30, None)),
    ("conversation", "follow_up_seconds"): (NUMBER, (0, None)),
    ("behaviour", "startup_sound"): (FLAG, None),
    ("behaviour", "conversation_memory"): (WHOLE, (0, None)),
    ("behaviour", "debug_mode"): (FLAG, None),
}


class ConfigError(ValueError):
    pass


def merge_defaults(user) -> dict:
    """Return DEFAULTS with the user's settings layered on top."""
    if user is None:
        user = {}
    if not isinstance(user, dict):
        raise ConfigError("the file should hold sections such as 'model:' and 'audio:'")
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in user.items():
        if values is None:  # a section heading with nothing under it
            values = {}
        if not isinstance(values, dict):
            raise ConfigError(f"'{section}' should be a section with settings under it")
        cfg.setdefault(section, {}).update(values)
    return cfg


def _is_kind(value, kind) -> bool:
    is_number = isinstance(value, Real) and not isinstance(value, bool)
    if kind == NUMBER:
        return is_number
    if kind == WHOLE:
        return is_number and float(value).is_integer()
    if kind == TEXT:
        return isinstance(value, str)
    if kind == FLAG:
        return isinstance(value, bool)
    if kind == DEVICE:
        return value is None or isinstance(value, str) or (is_number and float(value).is_integer())
    if kind == DURATION:
        return isinstance(value, str) or is_number
    raise AssertionError(kind)


def unknown_settings(cfg: dict) -> list[str]:
    """Settings the assistant doesn't know, most likely typos. Warned about, not fatal."""
    known = set(RULES) | RETIRED
    return [f"{s}.{k}" for s, values in cfg.items() for k in values
            if (s, k) not in known and s in DEFAULTS] + \
           [s for s in cfg if s not in DEFAULTS]


def validate(cfg: dict):
    missing = [f"{s}.{k}" for s, k in REQUIRED if cfg.get(s, {}).get(k) in (None, "")]
    if missing:
        raise ConfigError(f"missing required setting(s): {', '.join(missing)}")

    for (section, key), (kind, limits) in RULES.items():
        value = cfg[section].get(key)
        if value is None and kind != DEVICE:
            continue  # only REQUIRED settings can be missing, checked above
        name = f"{section}.{key}"
        if not _is_kind(value, kind):
            hint = " (remove the quotes)" if isinstance(value, str) and kind != TEXT else ""
            raise ConfigError(f"{name} should be {kind}, not {value!r}{hint}")
        if limits:
            low, high = limits
            if low is not None and value < low:
                raise ConfigError(f"{name} must be at least {low}, not {value}")
            if high is not None and value > high:
                raise ConfigError(f"{name} must be at most {high}, not {value}")

    if not cfg["model"]["host"].startswith(("http://", "https://")):
        raise ConfigError("model.host should start with http:// (e.g. http://localhost:11434)")
    conv = cfg["conversation"]
    if conv["max_pause_ms"] <= conv["pause_ms"]:
        raise ConfigError("conversation.max_pause_ms must be longer than conversation.pause_ms")


# Model files, given relative to the repo in config.example.yaml
MODEL_PATHS = [("tts", "model"), ("tts", "config"), ("conversation", "smart_turn_model")]


def resolve_model_paths(cfg: dict, config_dir: Path):
    """Let relative model paths work when the assistant is started from another
    folder: if a path isn't found from here, look next to config.yaml."""
    for section, key in MODEL_PATHS:
        path = Path(cfg[section][key])
        if not path.is_absolute() and not path.exists() and (config_dir / path).exists():
            cfg[section][key] = str(config_dir / path)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"not valid YAML: {e}") from None
    cfg = merge_defaults(raw)
    validate(cfg)
    resolve_model_paths(cfg, Path(path).resolve().parent)
    for name in unknown_settings(cfg):
        print(f"[!] {path}: unknown setting '{name}' ignored (a typo?)")
    return cfg
