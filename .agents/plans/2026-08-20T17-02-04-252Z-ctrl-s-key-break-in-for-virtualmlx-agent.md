# Plan: Ctrl-B Key Break-In (Barge-In) for VirtualMlx

## Goal
While the agent is **thinking** (LLM streaming) or **speaking** (TTS playback), pressing **Ctrl-B** should:
1. Immediately stop TTS audio playback (cut the current sentence mid-frame).
2. Cancel the in-flight LLM stream (stop consuming tokens).
3. Loop straight back to `listener.listen()` so the user can speak again — no restart, no lost state.

This mirrors the barge-in design documented in `INSIGHTS/INSIGHT_VOICELOOP.md` §9 (keypoll in TTS write loop) and §13 (cbreak terminal mode), minus the WebRTC AEC path (VirtualMlx has no AEC; voice barge-in is out of scope for this pass — key-based only).

## Key choice: Ctrl-B (0x02)
Originally requested Ctrl-S (0x13), but 0x13 is **XOFF** — the terminal line discipline intercepts it by default (with `IXON` set) and pauses output, so plain `tty.setcbreak()` never delivers it. That would require manually clearing `IXON`/`IXOFF`/`IXANY` in `c_iflag`, which is fragile (a botched restore can wedge the terminal until `stty sane`).

**Decision (user-approved): use Ctrl-B (0x02)** — no flow-control conflict, works with plain `tty.setcbreak()`, zero extra flag wrangling. The INSIGHT doc's pattern uses "any keypress"; Ctrl-B is an equally convenient barge-in key.

## Design

### New module: `UTILITIES/keyboard.py`
A small, self-contained break-in controller. Keeps terminal plumbing out of `main.py`/`transformer.py`.

```python
# UTILITIES/keyboard.py
import select, sys, termios, threading

# Ctrl-B byte
BREAK_KEY = 0x02          # Ctrl-B — no flow-control conflict

class BreakInController:
    """Owns terminal raw-mode + a stdin watcher thread.

    Usage:
        ctrl = BreakInController()
        ctrl.enter_raw()            # at startup, once
        ...
        ctrl.arm()                  # start watching (begin speak/think phase)
        ctrl.disarm()               # stop watching (phase over)
        ctrl.exit_raw()             # restore terminal at shutdown
    """
    def __init__(self):
        self._event = threading.Event()
        self._armed = threading.Event()
        self._old = None
        self._thread = threading.Thread(target=self._watch, daemon=True)

    # public ─────────────────────────────────────────────
    @property
    def triggered(self) -> bool:           # poll from TTS/LLM loops
        return self._event.is_set()

    def is_set(self) -> bool:              # alias
        return self._event.is_set()

    def arm(self):
        self._event.clear()
        self._armed.set()
        if not self._thread.is_alive():
            self._thread.start()

    def disarm(self):
        self._armed.clear()

    def reset(self):
        self._event.clear()

    # terminal ───────────────────────────────────────────
    def enter_raw(self):
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # cbreak: no line buffering, no echo  (Ctrl-B needs no IXON wrangling)
        new[3] &= ~termios.ECHO
        new[3] &= ~termios.ICANON
        termios.tcsetattr(fd, termios.TCSANOW, new)
        # NOTE: Ctrl-B (0x02) has no flow-control conflict, so we do NOT
        # touch IXON/IXOFF/IXANY. This keeps terminal setup safe & reversible.

    def exit_raw(self):
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    # watcher thread ─────────────────────────────────────
    def _watch(self):
        while True:
            self._armed.wait()                 # block until armed
            # poll stdin while armed
            while self._armed.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.02)  # 20ms poll
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch and ord(ch) == BREAK_KEY:
                    self._event.set()
                    self._armed.clear()         # one-shot
                    break
            self._event.wait() if False else None  # noop; loop back to wait
```

**Notes on the watcher:**
- Daemon thread, spawned once, lives for program lifetime.
- Only consumes CPU while armed (armed only during think+speak phase, not during listen).
- 20ms poll → feels instant for a human keypress and is negligible overhead.

### Wire into `main.py` (`VirtualMlx`)
- Construct `self._break = BreakInController()` in `__init__`; call `enter_raw()` after board init, `exit_raw()` in a `finally`/shutdown path.
- In the think+speak phase:
  - `self._break.arm()` before the `async for` loop.
  - `self._break.disarm()` after the loop (normal end or break).
  - If `self._break.triggered` → print `⏹  interrupted` , skip remaining speak, set ring → listening, `continue` (back to listen).
- Guard the existing `KeyboardInterrupt` handler so a genuine Ctrl-C still exits (Ctrl-C is 0x03; still delivered fine in cbreak since `ISIG` handling — see caveat below).

### Wire into `UTILITIES/transformer.py` (`Transformer.speak`)
- Accept an optional `break_check: Callable[[], bool] | None`.
- Between every `4096`-sample write (the existing per-frame loop), call `break_check()`; if truthy → `stream.stop()`, `break`. This is exactly INSIGHT §9's pattern, substituting our event poll for `select.select([sys.stdin],...)`.
- Keep the existing loudness-bar / `on_chunk` behavior intact; just add the early-exit.
- Signature: `def speak(self, text, on_chunk=None, break_check=None):`

### Wire into `UTILITIES/model.py` (`Model.stream_sentences`)
- Accept `break_check` (passed through from `main.py`, or read a shared event).
- Inside `async for chunk, _metadata in self._agent.astream(...)`:
  - `if break_check and break_check(): break`
  - Optionally call `.aclose()` on the async generator for a cleaner cancellation.
- After break, do **not** flush remainder — we want silence, not trailing audio.

### Call site change in `main.py`
```python
self._break.arm()
try:
    async for sentence in self.model.stream_sentences(
        text, break_check=self._break.is_set
    ):
        console.print(f"  [bold magenta]Mlx:[/] {sentence}")
        self.transformer.speak(
            sentence,
            on_chunk=...,
            break_check=self._break.is_set,
        )
        if self._break.triggered:
            break
finally:
    self._break.disarm()
    if self._board:
        board.set_wave([])
```

## Scope / Non-goals (this pass)
- **No WebRTC AEC / voice barge-in.** Pure key-based interrupt. Adding AEC is a separate, larger task (livekit APM, ref-signal alignment — see INSIGHT §10).
- **No mid-LLM token cancellation token.** We just stop iterating the LangGraph stream; underlying generation ends when the generator is closed. True hard-cancel of the Ollama request is out of scope.
- **Ctrl-S during the listening phase** = no-op (watcher disarmed; listener owns the mic). If we want Ctrl-S to abort a long silence too, that's a follow-up.
- **Ctrl-S during transcription** (the brief Moonshine call) — interrupt lands on first TTS frame or first LLM chunk; acceptable.

## Caveats / things to verify during build
1. **No IXON work needed** (we switched to Ctrl-B). But still verify the byte arrives: run, start speaking, press Ctrl-B. If nothing happens, confirm the watcher thread armed and `select` sees stdin.
2. **Ctrl-C must still exit.** In cbreak (ICANON off) `ISIG` is still on by default, so Ctrl-C still raises `KeyboardInterrupt`. We do **not** clear `ISIG`, so this should work. Verify the `except KeyboardInterrupt` path in `run()` still fires.
3. **Terminal restore on crash.** Wrap everything in `try/finally` and call `exit_raw()` in `run()`'s outer finally. A hard kill (-9) can't restore the terminal; user runs `stty sane` or `reset`.
4. **Thread safety of the `Event`.** `threading.Event` is fine across the audio thread (none here) / watcher thread / asyncio loop (single thread). No lock needed.
5. **`sys.stdin.read(1)` after arm** — if the user types other keys while armed, they're consumed and discarded. Acceptable; we only care about Ctrl-B.
6. **No AEC means the mic may pick up TTS** — but we disarm before listening, and the listener's VAD will start fresh. The agent's own voice could falsely trigger VAD *while* TTS is still ringing in the room. Out of scope here; document as a known limitation.

## Files to touch
| File | Change |
|------|--------|
| `UTILITIES/keyboard.py` | **NEW** — `BreakInController` (raw mode + IXON clear + watcher thread + event) |
| `main.py` | Construct controller, `enter_raw()`/`exit_raw()`, arm/disarm around think+speak, break → `continue` |
| `UTILITIES/transformer.py` | `speak(..., break_check=None)`; poll between 4096-sample writes |
| `UTILITIES/model.py` | `stream_sentences(..., break_check=None)`; check each async chunk, break early |
| `config.py` | (optional) `BREAK_IN_KEY = 0x02` (Ctrl-B) constant |

## Acceptance test
1. Run `uv run main.py`.
2. Say something → agent starts speaking (multi-sentence).
3. Press **Ctrl-B** mid-sentence → audio cuts immediately, no terminal freezes, ring → listening, mic captures the next utterance without restart.
4. Press Ctrl-B during the "Thinking…" phase (before any audio) → LLM stream stops, immediately back to listening.
5. Ctrl-C still exits cleanly with "Bye!" and terminal restored.

## Resolved
- **Key = Ctrl-B (0x02)**, per user. No flow-control/IXON handling required.
