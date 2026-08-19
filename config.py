"""Configuration constants for VirtualMlx."""

import os
from pathlib import Path

# ── Audio ────────────────────────────────────────────────────────
SAMPLE_RATE = 16000           # 16 kHz for all speech models
CHUNK_SAMPLES = 512           # 32ms blocks (Silero VAD requirement)
TTS_SAMPLE_RATE = 24000       # Kokoro's native output rate

# ── VAD ──────────────────────────────────────────────────────────
VAD_THRESHOLD = 0.5           # Speech probability threshold
SILENCE_MS = 700              # Silence before turn check (ms)

# ── Smart Turn ───────────────────────────────────────────────────
SMART_TURN_THRESHOLD = 0.5    # Turn completion threshold
SMART_TURN_MAX_SECONDS = 8    # Max audio window for prediction

# ── TTS ──────────────────────────────────────────────────────────
TTS_VOICE = "am_michael"      # US English male (Kokoro v1.0)
TTS_SPEED = 1.0
TTS_LANG = "en-us"
SENTENCE_MIN_CHARS = 20       # Min chars before dispatching a sentence
TTS_VOLUME = int(os.getenv("TTS_VOLUME", "12"))  # 1-20 loudness (10 = normal)

# ── LLM ──────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_HISTORY = 10              # Conversation turns in context
MAX_TOKENS = 300
TEMPERATURE = 0.7

# ── MCP / Websearch (parallel.ai — remote streamable-HTTP MCP) ──
# basic mode is free & anonymous; API key optional for higher rate limits.
PARALLEL_SEARCH_URL = os.getenv("PARALLEL_SEARCH_URL", "https://search.parallel.ai/mcp")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")

# ── Microphone ───────────────────────────────────────────────────
MIC_NAME = "UGREEN"           # Substring to match in device name
MIC_DEVICE_INDEX: int | None = None  # None = auto-detect by MIC_NAME

# ── Barehands ────────────────────────────────────────────────────
BAREHANDS_DIR = Path("/home/oj2/projects/clones/barehands")
BAREHANDS_PORT = int(os.getenv("BAREHANDS_PORT", "8794"))
BAREHANDS_HOST = os.getenv("BAREHANDS_HOST", "127.0.0.1")

# ── Paths ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
SYSTEM_PATH = PROJECT_DIR / "SYSTEM.md"
MODEL_CACHE_DIR = PROJECT_DIR / ".model_cache"
