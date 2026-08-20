"""main.py — VirtualMlx orchestrator.

The listen → transcribe → think → speak loop, now powered by a LangChain
agent (with MCP websearch tools) and reflected on the barehands ring.
"""

import asyncio

from rich.console import Console
from rich.panel import Panel

import board
from UTILITIES.keyboard import BreakInController
from UTILITIES.listener import Listener
from UTILITIES.model import Model
from UTILITIES.transformer import Transformer

console = Console()


class VirtualMlx:
    """The main voice agent. Boots subsystems, then enters the main loop."""

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
        self._break = BreakInController()
        self._break.enter_raw()
        self._listening = True  # toggled by Ctrl-T (see config.TOGGLE_LISTEN_KEY)
        # model is set up async in run()

    async def run(self):
        """Async entry — sets up the LangChain agent, then loops."""
        self.model = await Model.create()

        if self._board:
            board.set_ring_state("idle")
            board.set_mood("green")

        console.print(Panel.fit(
            "Ready. Speak anytime.\n[dim]Press Ctrl-T to pause/resume listening, "
            "Ctrl-B to interrupt, Ctrl+C to exit.[/]",
            border_style="green",
        ))

        while True:
            try:
                # ── NOTE: Apply any pending listen-toggle (Ctrl-T) request ──
                # The watcher thread sets the toggle event any time the key is
                # pressed. If listen() returned empty it already saw the toggle;
                # if we were paused we poll it here too.
                if self._break.consume_toggle():
                    self._listening = not self._listening
                    if self._listening:
                        console.print("  [bold green]▶ Listening resumed[/]")
                    else:
                        console.print("  [bold yellow]⏸  Listening paused "
                                      "(Ctrl-T to resume)[/]")
                        if self._board:
                            board.set_ring_state("idle")
                            board.set_mood("red")

                if not self._listening:
                    # Paused — wait quietly for the next toggle, no mic capture.
                    await asyncio.sleep(0.05)
                    continue

                # ── NOTE: Listen (sync — blocks the loop, fine: no async work
                #     happens until the agent runs) ──
                if self._board:
                    board.set_ring_state("listening")
                    board.set_mood("amber")
                audio = self.listener.listen(
                    stop_check=self._break.toggle_is_set
                )

                # Empty capture ⇒ interrupted by toggle (handled above on the
                # next iteration). Skip transcription.
                if audio.size == 0:
                    continue

                # ── NOTE: Transcribe ──
                if self._board:
                    board.set_ring_state("thinking")
                console.print("  [dim]Transcribing…[/]")
                text = self.transformer.transcribe(audio)
                if not text:
                    console.print("  [yellow](empty transcription, skipping)[/]")
                    continue
                console.print(f"  [bold cyan]You:[/] {text}")

                # ── NOTE: Think + Speak (async stream → sync TTS) ──
                if self._board:
                    board.set_ring_state("speaking")
                    board.set_mood("green")
                console.print("  [dim]Thinking…[/]")
                spoken_any = False

                # Arm Ctrl-B break-in for the whole think+speak phase.
                self._break.arm()
                try:
                    async for sentence in self.model.stream_sentences(
                        text, break_check=self._break.is_set
                    ):
                        console.print(f"  [bold magenta]Mlx:[/] {sentence}")
                        self.transformer.speak(
                            sentence,
                            on_chunk=lambda s: board.set_wave(_extract_waveform(s))
                            if self._board else None,
                            break_check=self._break.is_set,
                        )
                        spoken_any = True
                        if self._break.triggered:
                            break
                finally:
                    self._break.disarm()

                if self._break.triggered:
                    console.print("  [bold yellow]⏹  interrupted[/]")

                if self._board:
                    board.set_wave([])
                if not spoken_any and not self._break.triggered:
                    console.print("  [yellow](no response generated)[/]")

            except KeyboardInterrupt:
                console.print("\n  [bold]Bye![/]")
                break
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red bold]Error:[/] {e}")
                self._break.disarm()
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
    mlx = VirtualMlx()
    try:
        asyncio.run(mlx.run())
    finally:
        mlx._break.exit_raw()
