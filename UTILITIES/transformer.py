"""transformer.py — Moonshine STT and Kokoro TTS.

Handles the two transformation steps in the pipeline:
  audio → text  (Moonshine, local, CPU)
  text  → audio (Kokoro v1.0 ONNX, local, CPU)
"""

import urllib.request

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.text import Text

from config import (
    MODEL_CACHE_DIR,
    SAMPLE_RATE,
    TTS_LANG,
    TTS_SAMPLE_RATE,
    TTS_SPEED,
    TTS_VOICE,
    TTS_VOLUME,
)

console = Console()

# ── Kokoro model auto-download ───────────────────────────────────

_KOKORO_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)
_KOKORO_MODEL_URL = f"{_KOKORO_BASE}/kokoro-v1.0.onnx"
_KOKORO_VOICES_URL = f"{_KOKORO_BASE}/voices-v1.0.bin"


def _ensure_kokoro_files() -> tuple[str, str]:
    """Download Kokoro ONNX model + voices if not already cached."""
    cache = MODEL_CACHE_DIR / "kokoro"
    cache.mkdir(parents=True, exist_ok=True)

    model_path = cache / "kokoro-v1.0.onnx"
    voices_path = cache / "voices-v1.0.bin"

    if not model_path.exists():
        console.print("  Downloading Kokoro model (~300 MB)…")
        urllib.request.urlretrieve(_KOKORO_MODEL_URL, model_path)

    if not voices_path.exists():
        console.print("  Downloading Kokoro voices (~300 MB)…")
        urllib.request.urlretrieve(_KOKORO_VOICES_URL, voices_path)

    return str(model_path), str(voices_path)


# ── Loudness bar helper ──────────────────────────────────────────


def _loudness_bar(level: int, volume: int = TTS_VOLUME, max_lvl: int = 20) -> Text:
    """Build a 1-20 loudness indicator: filled blocks + empty + labels."""
    filled = "█" * level
    empty = "░" * (max_lvl - level)
    return Text.assemble(
        ("🔊 Vol:", "bold"),
        (f"{volume:>2}", "bold yellow"),
        (" | Loudness: ", ""),
        (filled, "bold green"),
        (empty, "dim"),
        (f" {level}/{max_lvl}", "bold"),
    )


def _rms_to_level(samples: np.ndarray, max_lvl: int = 20) -> int:
    """Map a chunk's RMS amplitude to a 1-20 level."""
    if len(samples) == 0:
        return 1
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    # Speech RMS typically 0.02-0.30; map that range onto 1-20
    level = int(1 + (rms / 0.30) * (max_lvl - 1))
    return max(1, min(max_lvl, level))


# ── Transformer class ────────────────────────────────────────────


class Transformer:
    """Speech-to-text (Moonshine) and text-to-speech (Kokoro)."""

    def __init__(self):
        # ── Moonshine STT ──
        from moonshine_voice import Transcriber, get_model_for_language

        console.print("Loading Moonshine STT…")
        ms_path, ms_arch = get_model_for_language("en")
        self._moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)
        console.print("  Moonshine loaded [green]✓[/]")

        # ── Kokoro TTS ──
        from kokoro_onnx import Kokoro

        console.print("Loading Kokoro TTS…")
        model_file, voices_file = _ensure_kokoro_files()
        self._kokoro = Kokoro(model_file, voices_file)
        console.print(f"  Kokoro loaded [green]✓[/] (voice: {TTS_VOICE}, "
                      f"vol: {TTS_VOLUME}/20)")

    # ── STT ──────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 audio array to text via Moonshine."""
        result = self._moonshine.transcribe_without_streaming(
            audio.tolist(), SAMPLE_RATE
        )
        return " ".join(line.text for line in result.lines if line.text).strip()

    # ── TTS ──────────────────────────────────────────────────────

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text → (audio_samples, sample_rate)."""
        if not text.strip():
            return np.array([], dtype=np.float32), TTS_SAMPLE_RATE

        samples, sr = self._kokoro.create(
            text, voice=TTS_VOICE, speed=TTS_SPEED, lang=TTS_LANG
        )
        return samples, sr

    def speak(self, text: str, on_chunk=None):
        """Synthesize text and play it through the speakers via sounddevice.

        Applies the configured volume gain (TTS_VOLUME / 10) and shows a
        live 1-20 loudness indicator during playback.  Optional
        ``on_chunk(samples)`` callback receives each audio chunk so callers
        (e.g. board.py) can stream a waveform to the barehands ring.
        """
        import sounddevice as sd

        samples, sr = self.synthesize(text)
        if len(samples) == 0:
            return

        # Apply volume gain (1-20 scale, 10 = normal 1.0x) + clip to avoid artifacts
        gain = TTS_VOLUME / 10.0
        samples = np.clip(samples * gain, -1.0, 1.0)
        data = samples.reshape(-1, 1).astype(np.float32)

        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
        stream.start()
        try:
            with Live(_loudness_bar(1), console=console, refresh_per_second=20) as live:
                # 4096 samples ≈ 170ms at 24kHz
                for i in range(0, len(data), 4096):
                    chunk = data[i : i + 4096]
                    stream.write(chunk)
                    flat = chunk.flatten()
                    live.update(_loudness_bar(_rms_to_level(flat)))
                    if on_chunk is not None:
                        on_chunk(flat)
        finally:
            stream.stop()
            stream.close()
