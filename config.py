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

# ── LLM ──────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_HISTORY = 10              # Conversation turns in context
MAX_TOKENS = 300
TEMPERATURE = 0.7

# ── Microphone ───────────────────────────────────────────────────
MIC_NAME = "UGREEN"           # Substring to match in device name
MIC_DEVICE_INDEX: int | None = None  # None = auto-detect by MIC_NAME

# ── Paths ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
SOUL_PATH = PROJECT_DIR / "SYSTEM.md"
MODEL_CACHE_DIR = PROJECT_DIR / ".model_cache"
