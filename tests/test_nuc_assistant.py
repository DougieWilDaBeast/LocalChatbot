"""Unit tests for the nuc_assistant package. Audio devices, Ollama and the speech models
are all faked, so these run anywhere without a mic, speaker or GPU."""

import json
import threading
import time
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
    ("Dr. Smith said hello. x", ("Dr. Smith said hello.", " x")),
    ("Bring fruit, e.g. apples. x", ("Bring fruit, e.g. apples.", " x")),
    ("He said \"stop.\" Then", ("He said \"stop.\"", " Then")),
    ("1. Apples and pears.\n", ("1. Apples and pears.", "\n")),
    ("In 1999. Then", ("In 1999.", " Then")),
    ("a, " * 60 + "b", ("a, " * 59 + "a,", " b")),  # long sentence: split at a comma
])
def test_pop_sentence(buffer, expected):
    assert llm.pop_sentence(buffer) == expected


# =============================================================================
# LLM client
# =============================================================================

class FakeResponse:
    def __init__(self, json_data=None, lines=(), status_code=200):
        self._json = json_data
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = "Error"
        self.closed = False

    def close(self):
        self.closed = True

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
    assert payload == {"model": "gemma4:e4b", "keep_alive": -1, "options": {"num_ctx": 4096}}


def test_llm_exits_when_model_missing(cfg, ollama):
    ollama.models = ("gemma3:4b",)
    with pytest.raises(SystemExit, match="ollama pull gemma4:e4b"):
        llm.LLMClient(cfg)


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


def test_chat_stream_apologises_when_nothing_comes_back(cfg, ollama):
    client = llm.LLMClient(cfg)
    assert list(client.chat_stream("Hi")) == [llm.APOLOGY]


def test_chat_stream_raises_ollama_errors_mid_reply(cfg, ollama, monkeypatch):
    client = llm.LLMClient(cfg)
    lines = [{"message": {"content": "First. Second half"}, "done": False},
             {"error": "model runner has unexpectedly stopped"}]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(lines=lines))
    reply = client.chat_stream("Hi")
    assert next(reply) == "First."
    with pytest.raises(llm.OllamaError, match="unexpectedly stopped"):
        next(reply)


def test_chat_stream_reports_ollama_http_errors(cfg, ollama, monkeypatch):
    client = llm.LLMClient(cfg)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(
        {"error": "model 'gemma4:e4b' not found"}, status_code=404))
    with pytest.raises(llm.OllamaError, match="not found"):
        list(client.chat_stream("Hi"))


def test_memory_zero_still_sends_the_question(cfg, ollama):
    cfg["behaviour"]["conversation_memory"] = 0
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["Hi."]
    client.chat("Hello")
    assert ollama.posts[-1][1]["messages"][-1] == {"role": "user", "content": "Hello"}
    assert len(ollama.posts[-1][1]["messages"]) == 2


def test_failed_request_leaves_no_question_in_memory(cfg, ollama, monkeypatch):
    client = llm.LLMClient(cfg)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(status_code=500))
    with pytest.raises(llm.OllamaError):
        client.chat("This one fails")
    assert list(client.history) == []


@pytest.mark.parametrize("text, expected", [
    ("**Paris** is the capital.", "Paris is the capital."),
    ("- Apples\n- Pears", "Apples Pears"),
    ("Sunny 😊 and 20°C.", "Sunny and 20°C."),
    ("See https://example.com for more.", "See a link for more."),
    ("## Summary", "Summary"),
])
def test_clean_for_speech(text, expected):
    assert llm.clean_for_speech(text) == expected


def test_chat_keeps_rolling_history(cfg, ollama):
    cfg["behaviour"]["conversation_memory"] = 1
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["First answer."]
    assert client.chat("First question") == "First answer."
    ollama.chat_pieces = ["Second answer."]
    client.chat("Second question")
    assert list(client.history) == [("Second question", "Second answer.")]
    ollama.chat_pieces = ["Third answer."]
    client.chat("Third question")
    # The oldest exchange is forgotten whole: the request starts with a question
    assert [m["role"] for m in ollama.posts[-1][1]["messages"]] == [
        "system", "user", "assistant", "user"]


# =============================================================================
# Speaker
# =============================================================================

@pytest.fixture
def speaker(cfg, monkeypatch, tmp_path):
    # A stand-in voice, so no real one is loaded
    for key in ("model", "config"):
        cfg["tts"][key] = str(tmp_path / f"voice.{key}")
        Path(cfg["tts"][key]).touch()
    monkeypatch.setattr(tts, "PiperVoice", type("V", (), {"load": staticmethod(lambda *a, **kw: None)}),
                        raising=False)
    monkeypatch.setattr(tts, "PIPER_AVAILABLE", True)
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
    (lambda c: c["model"].update(max_tokens="500"), "remove the quotes"),
    (lambda c: c["model"].update(temperature="0.7"), "model.temperature should be a number"),
    (lambda c: c["conversation"].update(barge_in="false"), "barge_in should be true or false"),
    (lambda c: c["audio"].update(record_seconds=0), "record_seconds must be at least"),
    (lambda c: c["conversation"].update(pause_ms=0), "pause_ms must be at least"),
    (lambda c: c["conversation"].update(max_pause_ms=400), "max_pause_ms must be longer"),
    (lambda c: c["model"].update(host="localhost:11434"), "http://"),
    (lambda c: c.update(behaviour=["not", "a", "section"]), "'behaviour' should be a section"),
])
def test_config_rejects_bad_settings(tmp_path, change, message):
    with open(REPO / "config.example.yaml") as f:
        raw = yaml.safe_load(f)
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(config.ConfigError, match=message):
        config.load_config(path)


def test_config_treats_an_empty_section_as_all_defaults(tmp_path):
    raw = yaml.safe_load((REPO / "config.example.yaml").read_text())
    raw["conversation"] = None  # "conversation:" with everything under it deleted
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert config.load_config(path)["conversation"] == config.DEFAULTS["conversation"]


def test_config_reports_invalid_yaml_in_one_line(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n\tname: x\n")
    with pytest.raises(config.ConfigError, match="not valid YAML"):
        config.load_config(path)


def test_config_warns_about_typos(tmp_path, capsys):
    raw = yaml.safe_load((REPO / "config.example.yaml").read_text())
    raw["wake_word"]["treshold"] = 0.9
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    config.load_config(path)
    assert "wake_word.treshold" in capsys.readouterr().out


def test_config_example_has_no_unknown_settings(capsys):
    config.load_config(REPO / "config.example.yaml")
    assert "unknown setting" not in capsys.readouterr().out


def test_config_finds_models_next_to_config_from_another_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config.load_config(REPO / "config.example.yaml")
    if (REPO / "models/smart-turn").exists():  # only after setup.sh has run
        assert Path(cfg["conversation"]["smart_turn_model"]).exists()
    raw = yaml.safe_load((REPO / "config.example.yaml").read_text())
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf/voice.onnx").touch()
    raw["tts"]["model"] = "voice.onnx"
    (tmp_path / "conf/config.yaml").write_text(yaml.safe_dump(raw))
    cfg = config.load_config(tmp_path / "conf/config.yaml")
    assert cfg["tts"]["model"] == str(tmp_path / "conf/voice.onnx")


# =============================================================================
# Barge-in, end-of-turn detection and follow-ups
# =============================================================================

def test_chat_stream_stops_when_interrupted(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["First part. ", "Second part. ", "Third part."]
    stop = threading.Event()
    reply = []
    for sentence in client.chat_stream("Tell me", stop=stop):
        reply.append(sentence)
        stop.set()  # the user talks over the first sentence
    assert reply == ["First part."]


def test_chat_stream_says_nothing_when_interrupted_before_any_answer(cfg, ollama):
    client = llm.LLMClient(cfg)
    ollama.chat_pieces = ["Never heard."]
    stop = threading.Event()
    stop.set()
    assert list(client.chat_stream("Tell me", stop=stop)) == []


def test_speak_stream_returns_only_what_was_heard(speaker):
    spk, said = speaker
    stop = threading.Event()

    def speak(text, stop=None):
        said.append(text)
        if text == "Two.":
            stop.set()  # the user cuts in during the second sentence

    spk.speak = speak
    heard = spk.speak_stream(iter(["One.", "Two.", "Three.", "Four."]), stop=stop)
    assert heard == ["One.", "Two."]


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


def test_speech_monitor_hears_user_and_keeps_their_audio(listener):
    lis, use_signal = listener
    use_signal(frames_of((1.0, False), (2.0, True)))
    with lis.monitor() as monitor:
        assert monitor.heard.wait(2)
        time.sleep(0.2)  # it keeps recording after hearing them, until stopped
    heard = np.concatenate(monitor.frames)
    assert np.count_nonzero(heard) == 2 * listen.SAMPLE_RATE  # every word kept


def test_speech_monitor_ignores_short_noises(listener):
    lis, use_signal = listener
    use_signal(np.tile(frames_of((0.1, True), (0.5, False)), 10))
    with lis.monitor() as monitor:
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



def test_turn_continues_when_user_keeps_talking_during_transcription(monkeypatch):
    from nuc_assistant import main as app

    first, rest = np.ones(10, np.int16), np.full(5, 2, np.int16)
    transcribed, asked = [], []

    class FakeMonitor:
        def __init__(self):
            self.heard = threading.Event()
            self.frames = [rest]

        def start(self):
            return self

        def stop(self):
            pass

    class FakeListener:
        barge_in = True
        max_samples = 16000 * 8
        monitors = []

        def __init__(self, cfg):
            pass

        def monitor(self):
            self.monitors.append(FakeMonitor())
            return self.monitors[-1]

        def record_turn(self, wait_seconds=None, initial=None):
            if wait_seconds is not None:
                raise KeyboardInterrupt  # end the run at the follow-up window
            return np.concatenate(initial) if initial else first

    class FakeTranscriber:
        def __init__(self, cfg):
            pass

        def transcribe(self, audio):
            transcribed.append(audio)
            if len(transcribed) == 1:
                # The user carries on talking while the first part is transcribed
                FakeListener.monitors[-1].heard.set()
                return "Sorry, stop. Actually,"
            return "Sorry, stop. Actually, what is two plus two?"

    class FakeLLM:
        def __init__(self, cfg):
            pass

        def chat_stream(self, text, stop=None):
            asked.append(text)
            return iter(["Four."])

        def remember(self, text, reply):
            pass

    class FakeSpeaker:
        def __init__(self, cfg):
            pass

        def speak_stream(self, sentences, debug=False, stop=None):
            return list(sentences)

    class FakeWake:
        def __init__(self, cfg):
            pass

        def wait_for_wake_word(self):
            pass

    monkeypatch.setattr(app, "Listener", FakeListener)
    monkeypatch.setattr(app, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(app, "LLMClient", FakeLLM)
    monkeypatch.setattr(app, "Speaker", FakeSpeaker)
    monkeypatch.setattr(app, "WakeWordDetector", FakeWake)
    monkeypatch.setattr(app, "play_beep", lambda *a, **kw: None)
    monkeypatch.setattr(app.sys, "argv",
                        ["assistant.py", "--config", str(REPO / "config.example.yaml")])
    app.main()

    # One turn, nothing lost: the LLM gets the whole question once
    assert np.array_equal(transcribed[1], np.concatenate([first, rest]))
    assert asked == ["Sorry, stop. Actually, what is two plus two?"]


# =============================================================================
# Main loop: failures
# =============================================================================

@pytest.fixture
def app(monkeypatch):
    """main() with every component faked. Tests set `app.turns` (what record_turn
    returns, in order; then the run ends) and can replace any fake."""
    from nuc_assistant import main as app

    class FakeListener:
        barge_in = False
        max_samples = 16000 * 8

        def __init__(self, cfg):
            pass

        def monitor(self):
            return None

        def record_turn(self, wait_seconds=None, initial=None):
            if not app.turns:
                raise KeyboardInterrupt
            return app.turns.pop(0)

    class FakeTranscriber:
        def __init__(self, cfg):
            pass

        def transcribe(self, audio):
            return "What time is it?"

    class FakeLLM:
        def __init__(self, cfg):
            self.remembered = []
            app.llm = self

        def chat_stream(self, text, stop=None):
            return iter(["It is noon."])

        def remember(self, text, reply):
            self.remembered.append((text, reply))

    class FakeSpeaker:
        def __init__(self, cfg):
            self.said = []
            app.speaker = self

        def speak_stream(self, sentences, debug=False, stop=None):
            heard = list(sentences)
            self.said.extend(heard)
            return heard

        def speak(self, text, stop=None):
            self.said.append(text)

    class FakeWake:
        def __init__(self, cfg):
            pass

        def wait_for_wake_word(self):
            pass

    monkeypatch.setattr(app, "Listener", FakeListener)
    monkeypatch.setattr(app, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(app, "LLMClient", FakeLLM)
    monkeypatch.setattr(app, "Speaker", FakeSpeaker)
    monkeypatch.setattr(app, "WakeWordDetector", FakeWake)
    monkeypatch.setattr(app, "play_beep", lambda *a, **kw: None)
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    monkeypatch.setattr(app.sys, "argv",
                        ["assistant.py", "--config", str(REPO / "config.example.yaml")])
    app.turns = [np.ones(10, np.int16)]
    return app


def test_main_remembers_what_was_said(app):
    app.main()
    assert app.speaker.said == ["It is noon."]
    assert app.llm.remembered == [("What time is it?", "It is noon.")]


def test_main_says_sorry_when_ollama_fails(app, monkeypatch):
    def failing(self, text, stop=None):
        raise requests.ConnectionError("Ollama stopped")
        yield
    monkeypatch.setattr(app.LLMClient, "chat_stream", failing)
    app.main()
    assert app.speaker.said == [app.SORRY]
    assert app.llm.remembered == []


def test_main_exits_after_repeated_device_errors(app, monkeypatch):
    def unplugged(self, wait_seconds=None, initial=None):
        raise OSError("Error querying device -1")
    monkeypatch.setattr(app.Listener, "record_turn", unplugged)
    with pytest.raises(SystemExit, match="errors in a row"):
        app.main()


def test_main_exits_without_a_keyboard_when_wake_word_is_off(app, monkeypatch):
    def no_stdin(self):
        raise EOFError
    monkeypatch.setattr(app.WakeWordDetector, "wait_for_wake_word", no_stdin)
    with pytest.raises(SystemExit, match="No keyboard"):
        app.main()


def test_main_exits_cleanly_when_a_component_fails_to_load(app, monkeypatch):
    def broken(self, cfg):
        raise RuntimeError("Piper voice file not found at x.onnx")
    monkeypatch.setattr(app.Speaker, "__init__", broken)
    with pytest.raises(SystemExit, match="Couldn't start: Piper voice file not found"):
        app.main()


def test_speaker_refuses_to_start_without_a_voice(cfg):
    cfg["tts"]["model"] = "does/not/exist.onnx"
    with pytest.raises(RuntimeError, match="voice file not found"):
        tts.Speaker(cfg)


def test_wake_word_required_when_enabled(cfg, monkeypatch):
    monkeypatch.setattr(wakeword, "OWW_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="openWakeWord"):
        wakeword.WakeWordDetector(cfg)
