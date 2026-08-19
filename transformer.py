"""transformer.py — Moonshine STT and Kokoro TTS.

Handles the two transformation steps in the pipeline:
  audio → text  (Moonshine, local, CPU)
  text  → audio (Kokoro v1.0 ONNX, local, CPU)
"""

import urllib.request

import numpy as np

from config import (
    MODEL_CACHE_DIR,
    SAMPLE_RATE,
    TTS_LANG,
    TTS_SAMPLE_RATE,
    TTS_SPEED,
    TTS_VOICE,
)

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
        print("  Downloading Kokoro model (~300 MB)…")
        urllib.request.urlretrieve(_KOKORO_MODEL_URL, model_path)

    if not voices_path.exists():
        print("  Downloading Kokoro voices (~300 MB)…")
        urllib.request.urlretrieve(_KOKORO_VOICES_URL, voices_path)

    return str(model_path), str(voices_path)


# ── Transformer class ────────────────────────────────────────────


class Transformer:
    """Speech-to-text (Moonshine) and text-to-speech (Kokoro)."""

    def __init__(self):
        # ── Moonshine STT ──
        from moonshine_voice import Transcriber, get_model_for_language

        print("Loading Moonshine STT…")
        ms_path, ms_arch = get_model_for_language("en")
        self._moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)
        print("  Moonshine loaded ✓")

        # ── Kokoro TTS ──
        from kokoro_onnx import Kokoro

        print("Loading Kokoro TTS…")
        model_file, voices_file = _ensure_kokoro_files()
        self._kokoro = Kokoro(model_file, voices_file)
        print(f"  Kokoro loaded ✓ (voice: {TTS_VOICE})")

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

    def speak(self, text: str):
        """Synthesize text and play it through the speakers via sounddevice."""
        import sounddevice as sd

        samples, sr = self.synthesize(text)
        if len(samples) == 0:
            return

        data = samples.reshape(-1, 1).astype(np.float32)

        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
        stream.start()
        try:
            # 4096 samples ≈ 170ms at 24kHz — allows interrupt checks between frames
            for i in range(0, len(data), 4096):
                stream.write(data[i : i + 4096])
        finally:
            stream.stop()
            stream.close()
