"""main.py — VirtualMlx orchestrator.

The listen → transcribe → think → speak loop, reflected on the
barehands glass-board ring when the board is running.
"""

from rich.console import Console
from rich.panel import Panel

import board
from UTILITIES.listener import Listener
from UTILITIES.model import Model
from UTILITIES.transformer import Transformer

console = Console()


class VirtualMlx:
    """The main voice agent. Boots all subsystems, then enters the main loop."""

    def __init__(self):
        console.print(Panel.fit(
            "[bold]VirtualMlx[/] — Starting Up",
            border_style="bright_blue",
        ))

        # Check for the barehands board (degrade gracefully if absent)
        if board.is_connected():
            console.print("  barehands board [green]connected ✓[/]")
            self._board = True
        else:
            console.print("  [yellow]⚠ barehands board not running — voice-only mode[/]")
            console.print("    Start it with: [dim]uv run board.py[/]")
            self._board = False

        self.listener = Listener()
        self.transformer = Transformer()
        self.model = Model()

        if self._board:
            board.set_ring_state("idle")
            board.set_mood("green")

        console.print(Panel.fit(
            "Ready. Speak anytime.\n[dim]Press Ctrl+C to exit.[/]",
            border_style="green",
        ))

        self._run()

    # ── main loop ────────────────────────────────────────────────

    def _run(self):
        """listen → transcribe → think → speak, forever."""
        while True:
            try:
                # ── Listen ──
                if self._board:
                    board.set_ring_state("listening")
                    board.set_mood("amber")
                audio = self.listener.listen()

                # ── Transcribe ──
                if self._board:
                    board.set_ring_state("thinking")
                console.print("  [dim]Transcribing…[/]")
                text = self.transformer.transcribe(audio)
                if not text:
                    console.print("  [yellow](empty transcription, skipping)[/]")
                    continue
                console.print(f"  [bold cyan]You:[/] {text}")

                # ── Think + Speak (streamed sentence-by-sentence) ──
                if self._board:
                    board.set_ring_state("speaking")
                    board.set_mood("green")
                console.print("  [dim]Thinking…[/]")
                spoken_any = False

                for sentence in self.model.stream_sentences(text):
                    console.print(f"  [bold magenta]Mlx:[/] {sentence}")
                    self.transformer.speak(
                        sentence,
                        on_chunk=lambda s: board.set_wave(_extract_waveform(s))
                        if self._board else None,
                    )
                    spoken_any = True

                if self._board:
                    board.set_wave([])
                if not spoken_any:
                    console.print("  [yellow](no response generated)[/]")

            except KeyboardInterrupt:
                console.print("\n  [bold]Bye![/]")
                break
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red bold]Error:[/] {e}")
                continue
            finally:
                if self._board:
                    board.set_ring_state("idle")


# ── waveform helper ──────────────────────────────────────────────


def _extract_waveform(samples, bins: int = 64) -> list[float]:
    """Convert a float32 audio chunk into 64 amplitude values 0..1."""
    if len(samples) == 0:
        return []
    step = max(1, len(samples) // bins)
    return [float(abs(samples[i])) for i in range(0, len(samples), step)][:bins]


if __name__ == "__main__":
    VirtualMlx()
