"""Chat with the local LLM through Ollama's REST API."""

import json
import re
import sys
from collections import deque

import requests


SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")


def trim_to_sentence(text: str) -> str:
    """Cut text back to its last complete sentence, if it has one."""
    if text.endswith((".", "!", "?")):
        return text
    end = max(text.rfind(p) for p in (". ", "! ", "? ", ".\n", "!\n", "?\n"))
    return text[:end + 1] if end > 0 else text


def pop_sentence(buffer: str) -> tuple[str, str]:
    """Split the first complete sentence off buffer. Returns (sentence, rest);
    sentence is empty if buffer doesn't hold a complete one yet."""
    for match in SENTENCE_END.finditer(buffer):
        sentence = buffer[:match.end()].strip()
        if sentence:  # skip blank lines
            return sentence, buffer[match.end():]
    return "", buffer


class LLMClient:
    def __init__(self, cfg: dict):
        self.model = cfg["model"]["name"]
        self.temperature = cfg["model"]["temperature"]
        self.max_tokens = cfg["model"]["max_tokens"]
        self.context_window = cfg["model"].get("context_window", 4096)
        self.think = cfg["model"].get("think", False)
        self.timeout = cfg["model"].get("timeout", 300)
        self.keep_alive = cfg["model"].get("keep_alive", -1)
        self.system_prompt = cfg["assistant"]["system_prompt"].strip()
        self.memory_size = cfg["behaviour"]["conversation_memory"]
        self.debug = cfg["behaviour"].get("debug_mode", False)
        self.history = deque(maxlen=self.memory_size * 2)  # user+assistant pairs
        self.base_url = cfg["model"].get("host", "http://localhost:11434").rstrip("/")

        print(f"[→] Connecting to Ollama ({self.model})...")
        self._check_connection()
        self._warm_up()
        print("[✓] Ollama ready")

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = {m["name"] for m in r.json().get("models", [])}
            wanted = self.model if ":" in self.model else f"{self.model}:latest"
            if wanted not in models:
                print(f"[!] Model '{self.model}' not found. Run: ollama pull {self.model}")
        except requests.ConnectionError:
            print("[✗] Cannot connect to Ollama. Is it running? sudo systemctl start ollama")
            sys.exit(1)

    def _warm_up(self):
        """Load the model now, so the first question doesn't wait for it,
        and keep it loaded for keep_alive (-1 = until Ollama stops)."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": self.keep_alive},
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

    def chat(self, user_text: str) -> str:
        """Get the whole reply as one string."""
        return " ".join(self.chat_stream(user_text))

    def chat_stream(self, user_text: str, stop=None):
        """Yield the reply one sentence at a time, as Ollama generates it.
        If `stop` (a threading.Event) is set, e.g. because the user talked over
        the reply, stop reading: leaving the request drops the connection,
        which makes Ollama stop generating."""
        self.history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(list(self.history))

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": self._options(),
        }

        spoken = []
        buffer = ""
        truncated = False
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if stop is not None and stop.is_set():
                    break
                if not line:
                    continue
                chunk = json.loads(line)
                buffer += chunk.get("message", {}).get("content", "")
                sentence, buffer = pop_sentence(buffer)
                while sentence:
                    spoken.append(sentence)
                    yield sentence
                    sentence, buffer = pop_sentence(buffer)
                if chunk.get("done"):
                    truncated = chunk.get("done_reason") == "length"

        if stop is not None and stop.is_set():
            # Interrupted: remember the part of the reply generated before the
            # user cut in, or forget the question if no answer had started.
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            elif self.history:
                self.history.pop()
            return

        # Whatever is left had no sentence ending. If the reply hit max_tokens
        # it was cut off mid-sentence, so drop it unless it's all we have.
        rest = buffer.strip()
        if rest and not (truncated and spoken):
            spoken.append(rest)
            yield rest

        # Some models/versions can return empty content from /api/chat.
        # Fall back to /api/generate to avoid silent empty assistant replies.
        if not spoken:
            if self.debug:
                print("[!] Empty /api/chat response, falling back to /api/generate")
            reply = self._generate(user_text)
            if not reply:
                reply = "I'm here, but I couldn't generate a response just now. Please try again."
            spoken.append(reply)
            yield reply

        self.history.append({"role": "assistant", "content": " ".join(spoken)})

    def _generate(self, user_text: str) -> str:
        gen_payload = {
            "model": self.model,
            "prompt": user_text,
            "system": self.system_prompt,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": self._options(),
        }
        gen_r = requests.post(
            f"{self.base_url}/api/generate",
            json=gen_payload,
            timeout=self.timeout,
        )
        gen_r.raise_for_status()
        gen_data = gen_r.json()
        reply = str(gen_data.get("response", "")).strip()
        if reply and gen_data.get("done_reason") == "length":
            # Hit max_tokens mid-sentence: speak only the complete sentences
            reply = trim_to_sentence(reply)
        return reply
