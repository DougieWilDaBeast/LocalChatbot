"""Unit tests for assistant.py. Audio devices, Ollama and the speech models
are all faked, so these run anywhere without a mic, speaker or GPU."""

import json
from pathlib import Path

import numpy as np
import pytest
import requests
import yaml

import assistant

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(REPO / "config.example.yaml") as f:
        return yaml.safe_load(f)


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
    assert assistant.trim_to_sentence(text) == expected


@pytest.mark.parametrize("buffer, expected", [
    ("Hello there. How are", ("Hello there.", " How are")),
    ("Is it? Yes", ("Is it?", " Yes")),
    ("Pi is 3.14 roughly", ("", "Pi is 3.14 roughly")),
    ("Wait.", ("", "Wait.")),  # no whitespace yet, more text may follow
    ("\n\nHello. x", ("Hello.", " x")),
    ("First line\nSecond", ("First line", "Second")),
])
def test_pop_sentence(buffer, expected):
    assert assistant.pop_sentence(buffer) == expected


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
    monkeypatch.setattr(assistant.requests, "get", fake.get)
    monkeypatch.setattr(assistant.requests, "post", fake.post)
    return fake


def test_llm_warms_up_model_at_startup(cfg, ollama):
    assistant.LLMClient(cfg)
    url, payload = ollama.posts[0]
    assert url.endswith("/api/generate")
    assert payload == {"model": "gemma4:e4b", "keep_alive": -1}


def test_llm_warns_when_model_missing(cfg, ollama, capsys):
    ollama.models = ("gemma3:4b",)
    assistant.LLMClient(cfg)
    assert "ollama pull gemma4:e4b" in capsys.readouterr().out


def test_llm_exits_when_ollama_down(cfg, monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError()
    monkeypatch.setattr(assistant.requests, "get", refuse)
    with pytest.raises(SystemExit):
        assistant.LLMClient(cfg)


def test_chat_stream_yields_sentences_as_they_complete(cfg, ollama):
    llm = assistant.LLMClient(cfg)
    ollama.chat_pieces = ["The capital", " of France is Paris.", " It is", " lovely in spring."]
    assert list(llm.chat_stream("Capital of France?")) == [
        "The capital of France is Paris.",
        "It is lovely in spring.",
    ]


def test_chat_stream_request_settings(cfg, ollama):
    llm = assistant.LLMClient(cfg)
    ollama.chat_pieces = ["Hi."]
    list(llm.chat_stream("Hello"))
    url, payload = ollama.posts[-1]
    assert url.endswith("/api/chat")
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["keep_alive"] == -1
    assert payload["options"]["num_predict"] == cfg["model"]["max_tokens"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "Hello"}


def test_chat_stream_drops_sentence_cut_off_by_token_limit(cfg, ollama):
    llm = assistant.LLMClient(cfg)
    ollama.chat_pieces = ["Once upon a time. ", "There was a keep"]
    ollama.done_reason = "length"
    assert list(llm.chat_stream("Story")) == ["Once upon a time."]


def test_chat_stream_keeps_unpunctuated_ending_when_not_cut_off(cfg, ollama):
    llm = assistant.LLMClient(cfg)
    ollama.chat_pieces = ["Sure. ", "Here you go"]
    assert list(llm.chat_stream("Hi")) == ["Sure.", "Here you go"]


def test_chat_stream_falls_back_to_generate_when_chat_is_empty(cfg, ollama):
    llm = assistant.LLMClient(cfg)
    ollama.generate_reply = "Fallback answer."
    assert list(llm.chat_stream("Hi")) == ["Fallback answer."]
    assert ollama.posts[-1][0].endswith("/api/generate")


def test_chat_stream_apologises_when_nothing_comes_back(cfg, ollama):
    llm = assistant.LLMClient(cfg)
    reply = list(llm.chat_stream("Hi"))
    assert len(reply) == 1 and "couldn't generate a response" in reply[0]


def test_chat_keeps_rolling_history(cfg, ollama):
    cfg["behaviour"]["conversation_memory"] = 1
    llm = assistant.LLMClient(cfg)
    ollama.chat_pieces = ["First answer."]
    assert llm.chat("First question") == "First answer."
    ollama.chat_pieces = ["Second answer."]
    llm.chat("Second question")
    assert list(llm.history) == [
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer."},
    ]


# =============================================================================
# Speaker
# =============================================================================

@pytest.fixture
def speaker(cfg, monkeypatch):
    cfg["tts"]["model"] = "does/not/exist.onnx"  # skip loading a real voice
    spk = assistant.Speaker(cfg)
    said = []
    monkeypatch.setattr(spk, "speak", said.append)
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
    monkeypatch.setattr(assistant.sd, "InputStream",
                        lambda **kw: FakeInputStream(signal, **kw))
    audio = assistant.record_until_silence(rate, max_seconds=8, silence_ms=500)
    # 0.5s lead-in + 1s speech + 0.5s of trailing silence
    assert len(audio) == pytest.approx(2.0 * rate, abs=0.1 * rate)


def test_record_until_silence_respects_max_length(monkeypatch):
    rate = 16000
    signal = np.full(rate * 20, 5000, np.int16)  # never goes quiet
    monkeypatch.setattr(assistant.sd, "InputStream",
                        lambda **kw: FakeInputStream(signal, **kw))
    audio = assistant.record_until_silence(rate, max_seconds=3)
    assert len(audio) == 3 * rate


def test_play_audio_file_resamples_to_48k(tmp_path, monkeypatch):
    import soundfile as sf
    path = tmp_path / "tone.wav"
    sf.write(path, np.zeros(22050, np.float32), 22050)  # 1s at 22.05kHz
    played = {}
    monkeypatch.setattr(assistant.sd, "play",
                        lambda data, rate, device=None: played.update(n=len(data), rate=rate))
    monkeypatch.setattr(assistant.sd, "wait", lambda: None)
    assistant.play_audio_file(str(path))
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

    monkeypatch.setattr(assistant, "WhisperModel", FakeWhisper)
    transcriber = assistant.Transcriber(cfg)
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

    monkeypatch.setattr(assistant, "OWW_AVAILABLE", True)
    monkeypatch.setattr(assistant, "WakeWordModel", FakeWakeModel, raising=False)
    monkeypatch.setattr(assistant.sd, "InputStream",
                        lambda **kw: FakeInputStream(np.zeros(16000, np.int16)))
    detector = assistant.WakeWordDetector(cfg)
    detector.wait_for_wake_word()
    assert FakeWakeModel.resets == 1
