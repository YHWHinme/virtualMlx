"""listener.py — Microphone capture with Silero VAD and Smart Turn v3.

Handles the full listen pipeline:
  sounddevice InputStream → Silero VAD (speech gate) → Smart Turn (turn-end gate) → utterance buffer
"""

import queue

import numpy as np
import torch

from config import (
    CHUNK_SAMPLES,
    MODEL_CACHE_DIR,
    SAMPLE_RATE,
    SILENCE_MS,
    SMART_TURN_MAX_SECONDS,
    SMART_TURN_THRESHOLD,
    VAD_THRESHOLD,
)

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

        print("  Downloading / loading Smart Turn v3…")
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

        print("  Smart Turn v3 loaded ✓")
        return predict

    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ Smart Turn unavailable ({e})")
        print("    Falling back to simple silence-based turn detection.")
        return None


# ── Listener class ───────────────────────────────────────────────


class Listener:
    """Captures complete utterances from the microphone.

    Uses Silero VAD as the first gate ("is someone talking?") and Smart Turn
    as the second gate ("are they actually done?").  Falls back to silence-
    only detection when Smart Turn is unavailable.
    """

    def __init__(self):
        from silero_vad import load_silero_vad

        print("Loading Silero VAD…")
        self.vad = load_silero_vad(onnx=True)
        print("  Silero VAD loaded ✓")

        print("Loading Smart Turn…")
        self.smart_turn = _load_smart_turn()

        # How many silent 32ms chunks before we consider running Smart Turn
        self._silence_limit = int(SILENCE_MS / (CHUNK_SAMPLES / SAMPLE_RATE * 1000))
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()

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
            self._audio_q.put(indata[:, 0].copy())

        sd.default.latency = "high"

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        ):
            print("\n  Listening…")
            while True:
                chunk = self._audio_q.get()
                prob = _vad_prob(self.vad, chunk)

                if prob > VAD_THRESHOLD:
                    # Speech detected
                    if not speaking:
                        speaking = True
                        buf.clear()
                    buf.append(chunk)
                    silent_chunks = 0

                elif speaking:
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
                        print(f"  Captured {dur:.1f}s of audio")
                        return audio
