"""main.py — VirtualMlx orchestrator.

The listen → transcribe → think → speak loop.
"""

from listener import Listener
from model import Model
from transformer import Transformer


class VirtualMlx:
    """The main voice agent. Boots all subsystems, then enters the main loop."""

    def __init__(self):
        print("═" * 50)
        print("  VirtualMlx — Starting Up")
        print("═" * 50)

        self.listener = Listener()
        self.transformer = Transformer()
        self.model = Model()

        print("═" * 50)
        print("  Ready. Speak anytime.")
        print("  Press Ctrl+C to exit.")
        print("═" * 50)

        self._run()

    # ── main loop ────────────────────────────────────────────────

    def _run(self):
        """listen → transcribe → think → speak, forever."""
        while True:
            try:
                # ── Listen ──
                audio = self.listener.listen()

                # ── Transcribe ──
                print("  Transcribing…")
                text = self.transformer.transcribe(audio)
                if not text:
                    print("  (empty transcription, skipping)")
                    continue
                print(f"  You: {text}")

                # ── Think + Speak (streamed sentence-by-sentence) ──
                print("  Thinking…")
                spoken_any = False

                for sentence in self.model.stream_sentences(text):
                    print(f"  Mlx: {sentence}")
                    self.transformer.speak(sentence)
                    spoken_any = True

                if not spoken_any:
                    print("  (no response generated)")

            except KeyboardInterrupt:
                print("\n  Bye!")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  Error: {e}")
                continue


if __name__ == "__main__":
    VirtualMlx()
