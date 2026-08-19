"""model.py — Ollama LLM client with streaming, sentence splitting, and history.

Wraps the ``ollama`` Python package to provide:
  - Streaming token generation (``chat_stream``)
  - Sentence-aware streaming (``stream_sentences``) for low-latency TTS dispatch
  - Conversation history (last N turns, configurable)
  - Live-reloaded system prompt from SOUL.md
"""

import re

from ollama import Client

from config import (
    MAX_HISTORY,
    MAX_TOKENS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    SOUL_PATH,
    TEMPERATURE,
)

_SENT_END = re.compile(r"(?<=[.!?])\s+")


# ── System prompt ────────────────────────────────────────────────


def _load_soul() -> str:
    """Read the system prompt from SOUL.md (re-read each turn for live edits)."""
    if SOUL_PATH.exists():
        return SOUL_PATH.read_text().strip()
    return "You are a helpful voice assistant. Speak naturally and concisely."


# ── Sentence accumulator ────────────────────────────────────────


class SentenceAccumulator:
    """Accumulates streaming LLM text and yields complete sentences.

    Short fragments (< *min_chars*) are merged with the next sentence to
    avoid TTS artifacts on abbreviations like "Mr." or "Dr.".
    """

    def __init__(self, min_chars: int = 20):
        self.min_chars = min_chars
        self._buffer = ""
        self._carry = ""

    def feed(self, text: str) -> list[str]:
        """Feed a new text chunk. Returns zero or more complete sentences."""
        self._buffer += text
        sentences: list[str] = []

        while True:
            m = _SENT_END.search(self._buffer)
            if not m:
                break
            fragment = self._buffer[: m.start() + 1].strip()
            self._buffer = self._buffer[m.end() :]

            # Merge short fragments
            self._carry = (
                (self._carry + " " + fragment).strip() if self._carry else fragment
            )
            if len(self._carry) >= self.min_chars:
                sentences.append(self._carry)
                self._carry = ""

        return sentences

    def flush(self) -> str | None:
        """Flush any remaining text as a final sentence."""
        remainder = self._buffer.strip()
        if self._carry:
            remainder = (self._carry + " " + remainder).strip()
        self._buffer = ""
        self._carry = ""
        return remainder if remainder else None


# ── Model class ──────────────────────────────────────────────────


class Model:
    """Ollama LLM interface with conversation history and streaming."""

    def __init__(self):
        print(f"Connecting to Ollama at {OLLAMA_HOST}…")
        self._client = Client(host=OLLAMA_HOST)
        self._history: list[dict] = []

        try:
            self._client.list()
            print(f"  Ollama connected ✓ (model: {OLLAMA_MODEL})")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ Could not reach Ollama ({e})")
            print(f"    Make sure Ollama is running at {OLLAMA_HOST}")

    # ── message construction ─────────────────────────────────────

    def _build_messages(self, user_text: str) -> list[dict]:
        """Build the chat message list: system prompt + history + current input."""
        messages: list[dict] = []

        soul = _load_soul()
        if soul:
            messages.append({"role": "system", "content": soul})

        for turn in self._history[-MAX_HISTORY:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": user_text})
        return messages

    # ── chat modes ───────────────────────────────────────────────

    def chat(self, user_text: str) -> str:
        """Blocking chat — sends the full message, waits for the full response."""
        messages = self._build_messages(user_text)
        response = self._client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        )
        reply = response["message"]["content"]
        self._history.append({"user": user_text, "assistant": reply})
        return reply

    def chat_stream(self, user_text: str):
        """Streaming chat — yields raw text chunks as they arrive from Ollama."""
        messages = self._build_messages(user_text)
        full_response = ""

        for chunk in self._client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        ):
            content = chunk["message"]["content"]
            if content:
                full_response += content
                yield content

        self._history.append({"user": user_text, "assistant": full_response})

    def stream_sentences(self, user_text: str):
        """Stream the LLM response as complete sentences.

        Each yielded string is a full sentence ready for TTS synthesis.
        Short fragments are merged to avoid TTS artifacts.
        """
        acc = SentenceAccumulator()

        for chunk in self.chat_stream(user_text):
            yield from acc.feed(chunk)

        remainder = acc.flush()
        if remainder:
            yield remainder
