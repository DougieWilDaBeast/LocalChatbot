"""Unit tests for the nuc_assistant package. Audio devices, Ollama and the speech models
are all faked, so these run anywhere without a mic, speaker or GPU."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import requests
import yaml

from nuc_assistant import audio, config, listen, llm, stt, tts, wakeword

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return config.load_config(REPO / "config.example.yaml")


# =============================================================================
# Sentence splitting
# =============================================================================

@pytest.mark.parametrize("text, expected", [
    ("Hello there. How are", "Hello there."),
    ("Done.", "Done."),
    ("A! B? C d", "A! B?"),
    ("Line one.\nLine two half", "Line one."),
    ("no full stop here", "no full stop here"),
])
def test_trim_to_sentence(text, expected):
    assert llm.trim_to_sentence(text) == expected


@pytest.mark.parametrize("buffer, expected", [
    ("Hello there. How are", ("Hello there.", " How are")),
    ("Is it? Yes", ("Is it?", " Yes")),
    ("Pi is 3.14 roughly", ("", "Pi is 3.14 roughly")),
    ("Wait.", ("", "Wait.")),  # no whitespace yet, more text may follow
    ("\n\nHello. x", ("Hello.", " x")),
    ("First line\nSecond", ("First line", "Second")),
])
def test_pop_sentence(buffer, expected):
    assert llm.pop_sentence(buffer) == expected


# =============================================================================
# LLM client
# =============================================================================

class FakeResponse:
    def __init__(self, json_data=None, lines=()):
        self._json = json_data
        self._lines = lines

    def raise_for_status(self):
        pass

    def json(self):
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def stream_lines(pieces, done_reason="stop"):
    lines = [{"message": {"content": p}, "done": False} for p in pieces]
    lines.append({"message": {"content": ""}, "done": True, "done_reason": done_reason})
    return lines


class FakeOllama:
    """Records requests and replies like Ollama's REST API."""

    def __init__(self, chat_pieces=(), done_reason="stop", generate_reply="",
                 models=("gemma4:e4b",)):
        self.chat_pieces = chat_pieces
        self.done_reason = done_reason
        self.generate_reply = generate_reply
        self.models = models
        self.posts = []

    def get(self, url, timeout=None):
        assert url.endswith("/api/tags")
        return FakeResponse({"models": [{"name": m} for m in self.models]})

    def post(self, url, json=None, timeout=None, stream=False):
        self.posts.append((url, json))
        if url.endswith("/api/chat"):
            return FakeResponse(lines=stream_lines(self.chat_pieces, self.done_reason))
        return FakeResponse({"response": self.generate_reply, "done_reason": "stop"})


@pytest.fixture
def ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(llm.requests, "get", fake.get)
    monkeypatch.setattr(llm.requests, "post", fake.post)
    return fake


def test_llm_warms_up_model_at_startup(cfg, ollama):
    llm.LLMClient(cfg)
    url, payload = ollama.posts[0]
    assert url.endswith("/api/generate")
    assert payload == {"model": "gemma4:e4b", "keep_alive": -1}


def test_llm_warns_when_model_missing(cfg, ollama, capsys):
    ollama.models = ("gemma3:4b",)
    llm.LLMClient(cfg)
    assert "ollama pull gemma4:e4b" in capsys.readouterr().out


def test_llm_exits_when_ollama_down(cfg, monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError()
    monkeypatch.setattr(llm.requests, "get", refuse)
    with pytest.raises(SystemExit):
        llm.LLMClient(cfg)


def test_chat_stream_yields_sentences_as_they_complete(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["The capital", " of France is Paris.", " It is", " lovely in spring."]
    assert list(client.chat_stream("Capital of France?")) == [
        "The capital of France is Paris.",
        "It is lovely in spring.",
    ]


def test_chat_stream_request_settings(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["Hi."]
    list(client.chat_stream("Hello"))
    url, payload = ollama.posts[-1]
    assert url.endswith("/api/chat")
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["keep_alive"] == -1
    assert payload["options"]["num_predict"] == cfg["model"]["max_tokens"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "Hello"}


def test_chat_stream_drops_sentence_cut_off_by_token_limit(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["Once upon a time. ", "There was a keep"]
    ollama.done_reason = "length"
    assert list(client.chat_stream("Story")) == ["Once upon a time."]


def test_chat_stream_keeps_unpunctuated_ending_when_not_cut_off(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["Sure. ", "Here you go"]
    assert list(client.chat_stream("Hi")) == ["Sure.", "Here you go"]


def test_chat_stream_falls_back_to_generate_when_chat_is_empty(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.generate_reply = "Fallback answer."
    assert list(client.chat_stream("Hi")) == ["Fallback answer."]
    assert ollama.posts[-1][0].endswith("/api/generate")


def test_chat_stream_apologises_when_nothing_comes_back(cfg, ollama):
    client = llm.LLMClient(cfg)
    reply = list(client.chat_stream("Hi"))
    assert len(reply) == 1 and "couldn't generate a response" in reply[0]


def test_chat_keeps_rolling_history(cfg, ollama):
    cfg["behaviour"]["conversation_memory"] = 1
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["First answer."]
    assert client.chat("First question") == "First answer."
    ollama.chat_pieces = ["Second answer."]
    client.chat("Second question")
    assert list(client.history) == [
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer."},
    ]


# =============================================================================
# Speaker
# =============================================================================

@pytest.fixture
def speaker(cfg, monkeypatch):
    cfg["tts"]["model"] = "does/not/exist.onnx"  # skip loading a real voice
    spk = tts.Speaker(cfg)
    said = []
    monkeypatch.setattr(spk, "speak", lambda text, stop=None: said.append(text))
    return spk, said


def test_speak_stream_speaks_each_sentence_in_order(speaker):
    spk, said = speaker
    spk.speak_stream(iter(["One.", "Two.", "Three."]))
    assert said == ["One.", "Two.", "Three."]


def test_speak_stream_reraises_llm_errors(speaker):
    spk, said = speaker

    def sentences():
        yield "Before the error."
        raise requests.Timeout("too slow")

    with pytest.raises(requests.Timeout):
        spk.speak_stream(sentences())
    assert said == ["Before the error."]


# =============================================================================
# Audio
# =============================================================================

class FakeInputStream:
    """Plays back a fixed int16 signal in whatever block size is read."""

    def __init__(self, signal, **kwargs):
        self.signal = signal
        self.pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        chunk = self.signal[self.pos:self.pos + n]
        self.pos += n
        if len(chunk) < n:
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), np.int16)])
        return chunk.reshape(-1, 1), False


def test_record_until_silence_stops_after_speech_then_silence(monkeypatch):
    rate = 16000
    speech = (np.sin(np.linspace(0, 2000, rate)) * 5000).astype(np.int16)  # 1s loud
    signal = np.concatenate([np.zeros(rate // 2, np.int16), speech, np.zeros(rate * 5, np.int16)])
    monkeypatch.setattr(audio.sd, "InputStream",
                        lambda **kw: FakeInputStream(signal, **kw))
    recorded = audio.record_until_silence(rate, max_seconds=8, silence_ms=500)
    # 0.5s lead-in + 1s speech + 0.5s of trailing silence
    assert len(recorded) == pytest.approx(2.0 * rate, abs=0.1 * rate)


def test_record_until_silence_respects_max_length(monkeypatch):
    rate = 16000
    signal = np.full(rate * 20, 5000, np.int16)  # never goes quiet
    monkeypatch.setattr(audio.sd, "InputStream",
                        lambda **kw: FakeInputStream(signal, **kw))
    recorded = audio.record_until_silence(rate, max_seconds=3)
    assert len(recorded) == 3 * rate


def test_play_audio_file_resamples_to_48k(tmp_path, monkeypatch):
    import soundfile as sf
    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(22050, np.float32), 22050)  # 1s at 22.05kHz
    played = {}
    monkeypatch.setattr(audio.sd, "play",
                        lambda data, rate, device=None: played.update(n=len(data), rate=rate))
    monkeypatch.setattr(audio.sd, "wait", lambda: None)
    audio.play_audio_file(str(path))
    assert played == {"n": 48000, "rate": 48000}


# =============================================================================
# Speech-to-text and wake word
# =============================================================================

def test_transcriber_normalises_audio_and_joins_segments(cfg, monkeypatch):
    seen = {}

    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            seen["language"] = kwargs["language"]
            segments = [type("S", (), {"text": " Hello"}), type("S", (), {"text": " world. "})]
            return iter(segments), None

    monkeypatch.setattr(stt, "WhisperModel", FakeWhisper)
    transcriber = stt.Transcriber(cfg)
    text = transcriber.transcribe(np.array([0, 16384, -32768], np.int16))
    assert text == "Hello world."
    assert seen["audio"].dtype == np.float32
    assert seen["audio"].tolist() == [0.0, 0.5, -1.0]
    assert seen["language"] == "en"


def test_wake_word_returns_on_detection_and_resets(cfg, monkeypatch):
    scores = iter([0.1, 0.2, 0.9])

    class FakeWakeModel:
        resets = 0

        def __init__(self, **kwargs):
            pass

        def predict(self, audio):
            assert len(audio) == 1280
            return {"hey_jarvis": next(scores)}

        def reset(self):
            FakeWakeModel.resets += 1

    monkeypatch.setattr(wakeword, "OWW_AVAILABLE", True)
    monkeypatch.setattr(wakeword, "WakeWordModel", FakeWakeModel, raising=False)
    monkeypatch.setattr(wakeword.sd, "InputStream",
                        lambda **kw: FakeInputStream(np.zeros(16000, np.int16)))
    detector = wakeword.WakeWordDetector(cfg)
    detector.wait_for_wake_word()
    assert FakeWakeModel.resets == 1


# =============================================================================
# Config
# =============================================================================

def test_config_fills_in_missing_settings_with_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "assistant": {"system_prompt": "Be brief."},
        "model": {"name": "gemma3:1b", "max_tokens": 120},
        "tts": {"model": "voice.onnx", "config": "voice.onnx.json"},
    }))
    cfg = config.load_config(path)
    assert cfg["model"]["name"] == "gemma3:1b"
    assert cfg["model"]["max_tokens"] == 120
    assert cfg["model"]["keep_alive"] == -1
    assert cfg["audio"]["speech_threshold"] == 500
    assert cfg["wake_word"]["model"] == "hey_jarvis"


def test_config_example_matches_defaults(cfg):
    for section, values in config.DEFAULTS.items():
        for key, default in values.items():
            assert cfg[section][key] == default, f"{section}.{key}"


@pytest.mark.parametrize("change, message", [
    (lambda c: c["model"].pop("name"), "model.name"),
    (lambda c: c["tts"].pop("model"), "tts.model"),
    (lambda c: c["wake_word"].update(threshold=5), "threshold"),
    (lambda c: c["model"].update(max_tokens=0), "max_tokens"),
])
def test_config_rejects_bad_settings(tmp_path, change, message):
    with open(REPO / "config.example.yaml") as f:
        raw = yaml.safe_load(f)
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(config.ConfigError, match=message):
        config.load_config(path)


# =============================================================================
# Barge-in, end-of-turn detection and follow-ups
# =============================================================================

def test_chat_stream_stops_when_interrupted_and_keeps_what_was_said(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["First part. ", "Second part. ", "Third part."]
    stop = threading.Event()
    reply = []
    for sentence in client.chat_stream("Tell me", stop=stop):
        reply.append(sentence)
        stop.set()  # the user talks over the first sentence
    assert reply == ["First part."]
    assert list(client.history)[-2:] == [
        {"role": "user", "content": "Tell me"},
        {"role": "assistant", "content": "First part."},
    ]


def test_chat_stream_forgets_question_interrupted_before_any_answer(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["Never heard."]
    stop = threading.Event()
    stop.set()
    assert list(client.chat_stream("Tell me", stop=stop)) == []
    assert list(client.history) == []
    assert not ollama.posts[-1][0].endswith("/api/generate")  # no fallback request


def test_speak_stream_stops_speaking_when_interrupted(speaker):
    spk, said = speaker
    stop = threading.Event()

    def speak(text, stop=None):
        said.append(text)
        stop.set()

    spk.speak = speak
    spk.speak_stream(iter(["One.", "Two.", "Three."]), stop=stop)
    assert said == ["One."]


def test_play_audio_file_cuts_off_when_stopped(tmp_path, monkeypatch):
    import soundfile as sf
    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(48000 * 10, np.float32), 48000)  # 10s
    calls = []
    monkeypatch.setattr(audio.sd, "play", lambda *a, **kw: calls.append("play"))
    monkeypatch.setattr(audio.sd, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(audio.sd, "wait", lambda: calls.append("wait"))
    stop = threading.Event()
    stop.set()
    audio.play_audio_file(str(path), stop=stop)
    assert calls == ["play", "stop", "wait"]


class FakeVAD:
    """Calls any loud frame speech."""

    def reset_states(self):
        pass

    def predict(self, frame, frame_size=480):
        return 1.0 if np.abs(frame).mean() > 1000 else 0.0


def frames_of(*parts):
    """Build a signal from (seconds, loud?) parts."""
    rate = listen.SAMPLE_RATE
    return np.concatenate([
        np.full(int(sec * rate), 5000 if loud else 0, np.int16) for sec, loud in parts])


@pytest.fixture
def listener(cfg, monkeypatch):
    cfg["conversation"]["smart_turn"] = False
    monkeypatch.setattr(listen, "load_vad", FakeVAD)

    def use_signal(signal):
        monkeypatch.setattr(listen.sd, "InputStream",
                            lambda **kw: FakeInputStream(signal, **kw))
    return listen.Listener(cfg), use_signal


def test_record_turn_ends_after_silence_without_smart_turn(listener):
    lis, use_signal = listener
    use_signal(frames_of((0.5, False), (1.0, True), (5, False)))
    turn = lis.record_turn()
    # 300 ms pre-roll + 1s speech + silence_threshold (500 ms)
    assert len(turn) / listen.SAMPLE_RATE == pytest.approx(1.8, abs=0.06)


def test_record_turn_asks_smart_turn_and_waits_out_a_mid_sentence_pause(listener):
    lis, use_signal = listener
    asked = []

    class NotFinished:
        def probability(self, audio):
            asked.append(len(audio))
            return 0.1

    lis.smart_turn = NotFinished()
    use_signal(frames_of((1.0, True), (5, False)))
    turn = lis.record_turn()
    assert len(asked) == 1  # asked once at pause_ms (500 ms), then waited for max_pause_ms
    assert len(turn) / listen.SAMPLE_RATE == pytest.approx(1.0 + 2.0, abs=0.06)


def test_record_turn_ends_early_when_smart_turn_says_finished(listener):
    lis, use_signal = listener
    lis.smart_turn = type("Done", (), {"probability": lambda self, audio: 0.9})()
    use_signal(frames_of((1.0, True), (5, False)))
    turn = lis.record_turn()
    assert len(turn) / listen.SAMPLE_RATE == pytest.approx(1.0 + 0.5, abs=0.06)


def test_record_turn_gives_up_when_no_follow_up(listener):
    lis, use_signal = listener
    use_signal(frames_of((10, False)))
    assert lis.record_turn(wait_seconds=2) is None


def test_record_turn_continues_from_barge_in_audio(listener):
    lis, use_signal = listener
    use_signal(frames_of((0.5, True), (5, False)))
    start = [np.full(listen.FRAME, 5000, np.int16)] * 10
    turn = lis.record_turn(initial=start)
    assert np.array_equal(turn[:listen.FRAME * 10], np.concatenate(start))


def test_barge_in_monitor_hears_user_and_keeps_their_audio(listener):
    lis, use_signal = listener
    use_signal(frames_of((1.0, False), (2.0, True)))
    with lis.barge_in_monitor() as monitor:
        assert monitor.heard.wait(2)
    speech_frames = sum(np.abs(f).mean() > 1000 for f in monitor.frames)
    assert speech_frames == lis.barge_in_frames  # stopped as soon as it was sure


def test_barge_in_monitor_ignores_short_noises(listener):
    lis, use_signal = listener
    use_signal(np.tile(frames_of((0.1, True), (0.5, False)), 10))
    with lis.barge_in_monitor() as monitor:
        assert not monitor.heard.wait(0.5)


def test_smart_turn_pads_audio_into_an_8_second_window(monkeypatch):
    seen = {}

    class FakeSession:
        def __init__(self, *a, **kw):
            pass

        def run(self, outputs, inputs):
            seen["features"] = inputs["input_features"]
            return [np.array([[0.75]], np.float32)]

    monkeypatch.setattr(listen.ort, "InferenceSession", FakeSession)
    model = listen.SmartTurn("model.onnx")
    assert model.probability(frames_of((1.0, True))) == pytest.approx(0.75)
    assert seen["features"].shape == (1, 80, 800)
    assert seen["features"].dtype == np.float32
