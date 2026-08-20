"""keyboard.py — Ctrl-B break-in (barge-in) controller.

Puts the terminal into cbreak mode so a keypress is delivered immediately
(no Enter needed), then runs a lightweight daemon thread that polls stdin
while *armed*. When the break-in key (Ctrl-B, byte 0x02) is seen, a
``threading.Event`` is set; the TTS playback loop and the LLM stream loop
both poll that event and bail out early.

Modeled on the barge-in design in ``INSIGHTS/INSIGHT_VOICELOOP.md`` (§9
stdin poll between audio frames, §13 cbreak terminal setup), minus the
WebRTC AEC path — this is key-based interrupt only.

Why Ctrl-B (0x02) and not Ctrl-S (0x13): Ctrl-S is XOFF and is swallowed
by the terminal line discipline when ``IXON`` is set. Clearing ``IXON``
manually is fragile (a botched restore can wedge the terminal until
``stty sane``). Ctrl-B has no flow-control conflict, so plain cbreak is
sufficient and the setup is fully reversible.
"""

import select
import sys
import termios
import threading

from config import BREAK_IN_KEY, TOGGLE_LISTEN_KEY

console = None  # imported lazily to avoid rich-on-import surprises


class BreakInController:
    """Owns terminal raw-mode + a stdin watcher thread.

    Lifecycle::

        ctrl = BreakInController()
        ctrl.enter_raw()           # once, at startup
        ...
        ctrl.arm()                 # begin a think+speak phase
        ...                        # TTS/LLM loops poll ctrl.is_set()
        ctrl.disarm()              # phase over
        ...
        ctrl.exit_raw()            # once, at shutdown

    The watcher thread is created on first ``arm()`` and is a daemon, so it
    never blocks interpreter shutdown.
    """

    def __init__(
        self,
        key: int = BREAK_IN_KEY,
        toggle_key: int = TOGGLE_LISTEN_KEY,
    ):
        self._key = key
        self._toggle_key = toggle_key
        self._event = threading.Event()        # break-in (one-shot per arm)
        self._toggle = threading.Event()        # listen toggle request
        self._armed = threading.Event()
        self._old_termios = None
        self._thread: threading.Thread | None = None

    # ── public poll API ────────────────────────────────────────

    def is_set(self) -> bool:
        """True once the break-in key has been seen since the last arm()."""
        return self._event.is_set()

    # alias kept for readability at call sites
    triggered = property(lambda self: self._event.is_set())

    # ── listen-toggle API ──────────────────────────────────────

    def toggle_is_set(self) -> bool:
        """True if the listen-toggle key has been pressed (not yet consumed)."""
        return self._toggle.is_set()

    def consume_toggle(self) -> bool:
        """Atomically check-and-clear the toggle request. Returns True if set."""
        was = self._toggle.is_set()
        self._toggle.clear()
        return was

    def arm(self) -> None:
        """Begin watching for the break-in key (start of a phase).

        The watcher thread is always running once ``enter_raw`` succeeded, so
        this just (re)arms the one-shot break-in event. We still start the
        thread as a fallback for callers that skipped ``enter_raw``.
        """
        self._event.clear()
        self._armed.set()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def disarm(self) -> None:
        """Stop watching (end of a phase). Leaves the event alone so callers
        can still read a just-triggered break after disarming."""
        self._armed.clear()

    def reset(self) -> None:
        """Clear any pending break signal (call before re-arming)."""
        self._event.clear()

    # ── terminal setup ─────────────────────────────────────────

    def enter_raw(self) -> None:
        """Switch stdin to cbreak mode (no echo, no line buffering).

        We deliberately keep ``ISIG`` on so Ctrl-C still raises
        ``KeyboardInterrupt``. We do **not** touch ``IXON``/``IXOFF`` because
        Ctrl-B has no flow-control conflict.
        """
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, ValueError):
            # stdin isn't a real tty (e.g. piped/redirected) — nothing to do
            return
        try:
            self._old_termios = termios.tcgetattr(fd)
        except termios.error:
            return  # not a terminal

        new = termios.tcgetattr(fd)
        # lflag (index 3): disable echo + canonical (line) mode
        new[3] &= ~termios.ECHO
        new[3] &= ~termios.ICANON
        # ISIG stays on → Ctrl-C still works.
        # IXON/IXOFF untouched → Ctrl-B / Ctrl-T are delivered unchanged.
        termios.tcsetattr(fd, termios.TCSANOW, new)

        # Start the always-on stdin watcher so the listen-toggle key works
        # even outside an armed (think/speak) phase — e.g. while the mic is
        # capturing audio. Daemon thread, so it never blocks shutdown.
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def exit_raw(self) -> None:
        """Restore the original terminal settings."""
        if self._old_termios is None:
            return
        try:
            termios.tcsetattr(
                sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios
            )
        except (termios.error, ValueError):
            pass
        finally:
            self._old_termios = None

    # ── watcher thread ─────────────────────────────────────────

    def _watch(self) -> None:
        """Background loop: poll stdin continuously.

        Always watches for the listen-toggle key; watches for the break-in
        key only while armed. Uses a short ``select`` timeout so it reacts
        within ~20ms and burns negligible CPU when idle.
        """
        while True:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            except (OSError, ValueError):
                # stdin isn't selectable (piped/redirected) — give up
                return
            if not readable:
                continue
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                return
            if not ch:
                continue
            o = ord(ch)
            if o == self._toggle_key:
                self._toggle.set()
            elif o == self._key and self._armed.is_set():
                self._event.set()
                self._armed.clear()  # one-shot per arm()
