"""Chat with the local LLM through Ollama's REST API."""

import json
import re
import sys
import unicodedata
from collections import deque

import requests

# A sentence ends at . ! or ? (plus any closing quote or bracket) followed by
# whitespace, or at a line break. Not after a list number ("2.") or a common
# abbreviation ("Dr.", "e.g."), which would make Piper pause mid-sentence.
SENTENCE_END = re.compile(
    r"(?<![\s(]\d)(?<!^\d)(?<![\s(]\d\d)"
    r"(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bSt)(?<!\bMrs)(?<!\bNo)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)"
    r"[.!?]+[\"'”’)\]]*(?=\s)|\n"
)
# Speak a long sentence in pieces rather than wait for its end: once this
# many characters have built up, split after the last comma, colon or semicolon.
LONG_SENTENCE = 150
CLAUSE_END = re.compile(r"[,;:—](?=\s)")

MARKDOWN = re.compile(r"[*_#`>|~]+")
LIST_MARKER = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
URL = re.compile(r"https?://\S+|www\.\S+")

APOLOGY = "Sorry, I don't have an answer for that. Please try again."


class OllamaError(requests.RequestException):
    """Ollama answered, but with an error instead of a reply."""


def clean_for_speech(text: str) -> str:
    """Strip what Piper would read out literally: markdown symbols, list
    bullets, emoji and URLs. Models do this now and then despite the prompt."""
    text = URL.sub("a link", text)
    text = LIST_MARKER.sub("", text)
    text = MARKDOWN.sub("", text)
    text = "".join(c for c in text
                   if c == "°" or unicodedata.category(c) not in ("So", "Cs", "Co"))
    return " ".join(text.split())


def trim_to_sentence(text: str) -> str:
    """Cut text back to its last complete sentence, if it has one."""
    ends = [m.end() for m in SENTENCE_END.finditer(text + " ")]
    return text[:ends[-1]].strip() if ends else text


def pop_sentence(buffer: str) -> tuple[str, str]:
    """Split the first complete sentence off buffer. Returns (sentence, rest);
    sentence is empty if buffer doesn't hold a complete one yet."""
    for match in SENTENCE_END.finditer(buffer):
        sentence = buffer[:match.end()].strip()
        if sentence:  # skip blank lines
            return sentence, buffer[match.end():]
    if len(buffer) > LONG_SENTENCE:
        clauses = [m.end() for m in CLAUSE_END.finditer(buffer)]
        if clauses:
            return buffer[:clauses[-1]].strip(), buffer[clauses[-1]:]
    return "", buffer


class LLMClient:
    """Streams replies from Ollama and keeps a rolling memory of the conversation."""

    def __init__(self, cfg: dict):
        self.model = cfg["model"]["name"]
        self.temperature = cfg["model"]["temperature"]
        self.max_tokens = cfg["model"]["max_tokens"]
        self.context_window = cfg["model"]["context_window"]
        self.think = cfg["model"]["think"]
        self.timeout = cfg["model"]["timeout"]
        self.keep_alive = cfg["model"]["keep_alive"]
        self.system_prompt = cfg["assistant"]["system_prompt"].strip()
        # (question, answer) pairs, so the oldest exchange is forgotten whole
        self.history = deque(maxlen=cfg["behaviour"]["conversation_memory"])
        self.base_url = cfg["model"]["host"].rstrip("/")
        self._response = None

        print(f"[→] Connecting to Ollama ({self.model})...")
        self._check_connection()
        self._warm_up()
        print("[✓] Ollama ready")

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            models = {m["name"] for m in r.json().get("models", [])}
        except requests.ConnectionError:
            sys.exit(f"[✗] Cannot connect to Ollama at {self.base_url}. Is it running? "
                     "sudo systemctl start ollama")
        except (requests.RequestException, ValueError) as e:
            sys.exit(f"[✗] Ollama at {self.base_url} isn't answering properly: {e}")
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in models:
            sys.exit(f"[✗] Model '{self.model}' isn't downloaded. Run: ollama pull {self.model}")

    def _warm_up(self):
        """Load the model now, so the first question doesn't wait for it,
        and keep it loaded for keep_alive (-1 = until Ollama stops)."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": self.keep_alive,
                      "options": {"num_ctx": self.context_window}},
                timeout=self.timeout,
            ).raise_for_status()
        except requests.RequestException as e:
            print(f"[!] Could not preload {self.model}: {e}")

    def _options(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.context_window,
        }

    def messages(self, user_text: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for question, answer in self.history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": user_text})
        return messages

    def remember(self, user_text: str, reply: str):
        """Add an exchange to the conversation memory. Called with what the
        user actually heard, which is less than the reply if they cut in."""
        if reply:
            self.history.append((user_text, reply))

    def chat(self, user_text: str) -> str:
        """Get the whole reply as one string, and remember it."""
        reply = " ".join(self.chat_stream(user_text))
        self.remember(user_text, reply)
        return reply

    def cancel(self):
        """Drop the reply being streamed, which makes Ollama stop generating it."""
        response = self._response
        if response is not None:
            response.close()

    def chat_stream(self, user_text: str, stop=None):
        """Yield the reply one sentence at a time, as Ollama generates it.
        If `stop` (a threading.Event) is set, e.g. because the user talked over
        the reply, stop reading: leaving the request drops the connection,
        which makes Ollama stop generating. The caller decides what to
        remember(), since only it knows how much of the reply was heard."""
        payload = {
            "model": self.model,
            "messages": self.messages(user_text),
            "stream": True,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": self._options(),
        }

        spoken = 0
        buffer = ""
        finished = truncated = False
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as r:
            if not r.ok:
                raise OllamaError(f"{r.status_code} from Ollama: {error_message(r)}")
            self._response = r
            try:
                for line in r.iter_lines():
                    if stop is not None and stop.is_set():
                        return
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        raise OllamaError(f"unreadable line from Ollama: {line[:80]!r}") from None
                    if "error" in chunk:  # e.g. the model ran out of memory mid-reply
                        raise OllamaError(chunk["error"])
                    buffer += chunk.get("message", {}).get("content", "")
                    sentence, buffer = pop_sentence(buffer)
                    while sentence:
                        sentence = clean_for_speech(sentence)
                        if sentence:
                            spoken += 1
                            yield sentence
                        sentence, buffer = pop_sentence(buffer)
                    if chunk.get("done"):
                        finished = True
                        truncated = chunk.get("done_reason") == "length"
            finally:
                self._response = None

        if stop is not None and stop.is_set():
            return
        if not finished:
            raise OllamaError("the reply stopped before Ollama said it was done")

        # Whatever is left had no sentence ending. If the reply hit max_tokens
        # it was cut off mid-sentence, so drop it unless it's all we have.
        rest = clean_for_speech(buffer)
        if rest and not (truncated and spoken):
            spoken += 1
            yield rest
        if not spoken:
            yield APOLOGY


def error_message(r) -> str:
    """Ollama's own explanation of a failed request, e.g. 'model not found'."""
    try:
        return r.json()["error"]
    except (ValueError, KeyError, TypeError):
        return r.reason
