"""main.py — VirtualMlx orchestrator.

The listen → transcribe → think → speak loop.
"""

from rich.console import Console
from rich.panel import Panel

from listener import Listener
from model import Model
from transformer import Transformer

console = Console()


class VirtualMlx:
    """The main voice agent. Boots all subsystems, then enters the main loop."""

    def __init__(self):
        console.print(Panel.fit(
            "[bold]VirtualMlx[/] — Starting Up",
            border_style="bright_blue",
        ))

        self.listener = Listener()
        self.transformer = Transformer()
        self.model = Model()

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
                audio = self.listener.listen()

                # ── Transcribe ──
                console.print("  [dim]Transcribing…[/]")
                text = self.transformer.transcribe(audio)
                if not text:
                    console.print("  [yellow](empty transcription, skipping)[/]")
                    continue
                console.print(f"  [bold cyan]You:[/] {text}")

                # ── Think + Speak (streamed sentence-by-sentence) ──
                console.print("  [dim]Thinking…[/]")
                spoken_any = False

                for sentence in self.model.stream_sentences(text):
                    console.print(f"  [bold magenta]Mlx:[/] {sentence}")
                    self.transformer.speak(sentence)
                    spoken_any = True

                if not spoken_any:
                    console.print("  [yellow](no response generated)[/]")

            except KeyboardInterrupt:
                console.print("\n  [bold]Bye![/]")
                break
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red bold]Error:[/] {e}")
                continue


if __name__ == "__main__":
    VirtualMlx()
