"""listener.py — Microphone capture with Silero VAD and Smart Turn v3.

Handles the full listen pipeline:
  sounddevice InputStream → Silero VAD (speech gate) → Smart Turn (turn-end gate) → utterance buffer
"""

import queue

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.text import Text

from config import (
    CHUNK_SAMPLES,
    MIC_DEVICE_INDEX,
    MIC_NAME,
    MODEL_CACHE_DIR,
    SAMPLE_RATE,
    SILENCE_MS,
    SMART_TURN_MAX_SECONDS,
    SMART_TURN_THRESHOLD,
    VAD_THRESHOLD,
)

console = Console()


# ── Microphone discovery ────────────────────────────────────────


def _find_microphone() -> tuple[int, dict]:
    """Find the target microphone by name substring or explicit index.

    Returns ``(device_index, device_info)``.
    Raises ``RuntimeError`` if no suitable device is found.
    """
    import sounddevice as sd

    devices = sd.query_devices()

    # Explicit index takes priority
    if MIC_DEVICE_INDEX is not None:
        info = sd.query_devices(MIC_DEVICE_INDEX)
        console.print(f"  [bold]Mic:[/] Using device {MIC_DEVICE_INDEX}: "
                      f"[cyan]{info['name']}[/]")
        return MIC_DEVICE_INDEX, info

    # Search by name substring (case-insensitive)
    search = MIC_NAME.lower()
    candidates: list[tuple[int, dict]] = []
    for idx, dev in enumerate(devices):
        if search in dev["name"].lower() and dev["max_input_channels"] > 0:
            candidates.append((idx, dev))

    if not candidates:
        console.print(f"  [red bold]✗ No microphone matching '{MIC_NAME}'[/]")
        console.print("  Available input devices:")
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                console.print(f"    [{idx}] {dev['name']}  "
                              f"(inputs={dev['max_input_channels']}, "
                              f"sr={dev['default_samplerate']:.0f})")
        raise RuntimeError(f"No microphone matching '{MIC_NAME}'")

    idx, info = candidates[0]
    return idx, info


def _print_mic_details(device_index: int, info: dict):
    """Print detailed microphone information at startup."""
    console.print("  [bold]Microphone Details[/]")
    console.print(f"    Index:          [cyan]{device_index}[/]")
    console.print(f"    Name:           [cyan]{info['name']}[/]")
    console.print(f"    Input channels: [cyan]{info['max_input_channels']}[/]")
    console.print(f"    Default SR:     [cyan]{info['default_samplerate']:.0f} Hz[/]")
    console.print(f"    Latency:        [cyan]{info['default_low_input_latency']:.4f}s "
                  f"(low) / {info['default_high_input_latency']:.4f}s (high)[/]")
    host_api = info.get("hostapi", "?")
    console.print(f"    Host API:       [cyan]{host_api}[/]")


# ── Silero VAD helper ───────────────────────────────────────────


def _vad_prob(vad, chunk: np.ndarray) -> float:
    """Run Silero VAD on one 32ms chunk. Returns speech probability 0..1."""
    tensor = torch.from_numpy(chunk)
    p = vad(tensor, SAMPLE_RATE)
    return p.item() if hasattr(p, "item") else p


# ── Smart Turn loader ────────────────────────────────────────────


def _load_smart_turn():
    """Load Smart Turn v3 ONNX model for conversation-end detection.

    Returns a ``predict(audio) -> float`` callable, or ``None`` if loading
    fails (in which case we fall back to simple silence-based detection).
    """
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        console.print("  Downloading / loading Smart Turn v3…")
        model_path = hf_hub_download(
            repo_id="onnx-community/smart-turn-v3-ONNX",
            filename="model.onnx",
            cache_dir=str(MODEL_CACHE_DIR / "smart_turn"),
        )

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

        def predict(audio: np.ndarray) -> float:
            max_samples = SMART_TURN_MAX_SECONDS * SAMPLE_RATE
            audio = audio[-max_samples:]
            features = extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                max_length=max_samples,
                padding="max_length",
                return_attention_mask=False,
                return_tensors="np",
            )
            out = session.run(
                None,
                {"input_features": features.input_features.astype(np.float32)},
            )
            return float(out[0].flatten()[0])

        console.print("  Smart Turn v3 loaded [green]✓[/]")
        return predict

    except Exception as e:  # noqa: BLE001
        console.print(f"  [yellow]⚠ Smart Turn unavailable ({e})[/]")
        console.print("    Falling back to simple silence-based turn detection.")
        return None


# ── Listener class ───────────────────────────────────────────────


class Listener:
    """Captures complete utterances from the microphone.

    Uses Silero VAD as the first gate ("is someone talking?") and Smart Turn
    as the second gate ("are they actually done?").  Falls back to silence-
    only detection when Smart Turn is unavailable.
    """

    def __init__(self):
        import sounddevice as sd
        from silero_vad import load_silero_vad

        # ── Discover and validate microphone ──
        self._device_index, self._device_info = _find_microphone()
        _print_mic_details(self._device_index, self._device_info)

        # Validate sample rate support
        actual_rate = self._validate_sample_rate(sd)
        if actual_rate != SAMPLE_RATE:
            console.print(f"  [yellow]⚠ Mic doesn't support {SAMPLE_RATE} Hz, "
                          f"using {actual_rate} Hz with resampling[/]")
        self._actual_rate = actual_rate

        # ── Load models ──
        console.print("Loading Silero VAD…")
        self.vad = load_silero_vad(onnx=True)
        console.print("  Silero VAD loaded [green]✓[/]")

        console.print("Loading Smart Turn…")
        self.smart_turn = _load_smart_turn()

        # How many silent 32ms chunks before we consider running Smart Turn
        self._silence_limit = int(SILENCE_MS / (CHUNK_SAMPLES / SAMPLE_RATE * 1000))
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()

    # ── sample rate validation ───────────────────────────────────

    def _validate_sample_rate(self, sd) -> int:
        """Check which sample rates the mic supports. Return best match."""
        preferred = [SAMPLE_RATE, 48000, 44100]
        for rate in preferred:
            try:
                stream = sd.InputStream(
                    device=self._device_index,
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=CHUNK_SAMPLES,
                )
                stream.start()
                stream.stop()
                stream.close()
                return rate
            except Exception:  # noqa: BLE001
                continue
        # Last resort: use device default
        return int(self._device_info["default_samplerate"])

    # ── resampling helper ────────────────────────────────────────

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        """Resample a chunk from the mic's native rate to SAMPLE_RATE."""
        if self._actual_rate == SAMPLE_RATE:
            return chunk
        ratio = SAMPLE_RATE / self._actual_rate
        new_len = int(len(chunk) * ratio)
        idx = np.linspace(0, len(chunk) - 1, new_len)
        return np.interp(idx, np.arange(len(chunk)), chunk).astype(np.float32)

    # ── public API ───────────────────────────────────────────────

    def listen(self) -> np.ndarray:
        """Block until a complete utterance is captured.

        Returns a float32 numpy array of the full utterance at 16 kHz.
        """
        import sounddevice as sd

        buf: list[np.ndarray] = []
        speaking = False
        silent_chunks = 0

        def callback(indata, frames, time_info, status):
            # Runs in PortAudio's C thread — copy immediately
            chunk = indata[:, 0].copy()
            if self._actual_rate != SAMPLE_RATE:
                chunk = self._resample(chunk)
            self._audio_q.put(chunk)

        sd.default.latency = "high"

        with sd.InputStream(
            device=self._device_index,
            samplerate=self._actual_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        ):
            # Rich live display for listening state
            passive = Text("● PASSIVE LISTENING", style="dim")
            active = Text("● RECEIVING AUDIO", style="bold green")

            with Live(passive, console=console, refresh_per_second=10) as live:
                while True:
                    chunk = self._audio_q.get()
                    prob = _vad_prob(self.vad, chunk)

                    if prob > VAD_THRESHOLD:
                        # Speech detected
                        live.update(active)
                        if not speaking:
                            speaking = True
                            buf.clear()
                        buf.append(chunk)
                        silent_chunks = 0

                    else:
                        live.update(passive)
                        if speaking:
                            # Silence after speech — keep accumulating
                            buf.append(chunk)
                            silent_chunks += 1

                            if silent_chunks >= self._silence_limit:
                                # Gate 2: Smart Turn (if available)
                                if self.smart_turn and buf:
                                    turn_prob = self.smart_turn(np.concatenate(buf))
                                    if turn_prob < SMART_TURN_THRESHOLD:
                                        # Not done — just pausing mid-thought
                                        silent_chunks = 0
                                        continue

                                # Turn is complete
                                self.vad.reset_states()
                                audio = np.concatenate(buf)
                                dur = len(audio) / SAMPLE_RATE
                                console.print(f"  Captured [bold]{dur:.1f}s[/] of audio")
                                return audio
