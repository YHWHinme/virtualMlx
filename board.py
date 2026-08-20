"""board.py — VirtualMlx's bridge to the barehands glass board.

Wraps the three barehands seams:
  1. Ring state   (face)  → write tiny files to barehands/state/
  2. Board cmds   (hands) → POST /cmd
  3. Board state  (eyes)  → GET /state

Running this file directly (``python board.py``) starts the barehands
HTTP server as a **blocking foreground process** in the current terminal
— run it in one window, ``uv run main.py`` in another.  The agent checks
for a connection at startup and degrades gracefully to voice-only if the
board isn't running.
"""

import json
import time
import urllib.request
from pathlib import Path

from rich.console import Console

from config import BAREHANDS_DIR, BAREHANDS_HOST, BAREHANDS_PORT

console = Console()

# ── URLs ─────────────────────────────────────────────────────────

_BASE = f"http://{BAREHANDS_HOST}:{BAREHANDS_PORT}"


def _url(path: str) -> str:
    return f"{_BASE}{path}"


# ── Connection check ─────────────────────────────────────────────


def is_connected() -> bool:
    """Return True if the barehands server is reachable."""
    try:
        urllib.request.urlopen(_url("/config"), timeout=1)
        return True
    except Exception:  # noqa: BLE001
        return False


# ── NOTE: State file helpers (Seam 1: the ring face) ───────────────────

_STATE_DIR = Path(BAREHANDS_DIR) / "state"


def _ensure_state_dir():
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def set_ring_state(state: str):
    """Set the ring animation: idle | listening | thinking | speaking."""
    _ensure_state_dir()
    if state in ("idle", "listening", "thinking", "speaking"):
        (_STATE_DIR / "state").write_text(state)


def set_mood(mood: str):
    """Set ring color: green | amber | red."""
    _ensure_state_dir()
    (_STATE_DIR / "mood.json").write_text(
        json.dumps({"mood": mood, "ts": time.time()})
    )


def set_wave(samples: list[float]):
    """Stream audio waveform samples (0..1, up to 64 values) to the ring."""
    _ensure_state_dir()
    (_STATE_DIR / "wave.json").write_text(
        json.dumps({"samples": samples[:64], "ts": time.time()})
    )


# ── NOTE: Board commands (Seam 2: the agent's hands) ───────────────────


def cmd(action: str, **kwargs) -> bool:
    """Send a board command. Returns True on success."""
    payload = {"a": action, **kwargs}
    try:
        req = urllib.request.Request(
            _url("/cmd"),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 204
    except Exception:  # noqa: BLE001
        return False


def add_card(title: str, body: str = "", file: str = "", open: bool = False) -> bool:
    return cmd("add_card", title=title, body=body, file=file, open=open)


def present(title: str, body: str = "", src: str = "", file: str = "") -> bool:
    return cmd("present", title=title, body=body, src=src, file=file)


def add_img(src: str, title: str = "") -> bool:
    return cmd("add_img", src=src, title=title)


def clear() -> bool:
    return cmd("clear")


# ── NOTE: Board state (Seam 3: the agent's eyes) ───────────────────────


def get_board_state() -> dict | None:
    """Read what's currently on the board."""
    try:
        req = urllib.request.Request(
            _url("/state"),
            headers={"Cache-Control": "no-store"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def describe_board() -> str:
    """Human-readable summary of the board for the LLM context."""
    state = get_board_state()
    if not state:
        return "The board server is not running."
    items = state.get("items", [])
    if not items:
        return "The board is empty."

    lines = [f"ON THE BOARD — {len(items)} item(s):"]
    for i in items:
        t = i.get("type", "?")
        title = i.get("title", "")
        src = i.get("src", "")
        pos = _zone(i.get("x", 0.5), i.get("y", 0.5))
        grabbed = " [IN USER'S HAND]" if i.get("g") else ""
        if t == "card":
            lines.append(f'  - card "{title}" @ {pos}{grabbed}')
        elif t == "panel":
            lines.append(f'  - open note "{title}" @ {pos}{grabbed}')
        elif t in ("img", "model"):
            lines.append(f'  - {t} "{src}" @ {pos}{grabbed}')
        elif t == "orb":
            lines.append(f'  - orb "{title}"')
        else:
            lines.append(f'  - {t} "{title}" @ {pos}{grabbed}')
    return "\n".join(lines)


def _zone(x: float, y: float) -> str:
    h = "left" if x < 0.33 else ("center" if x < 0.67 else "right")
    v = "top" if y < 0.33 else ("middle" if y < 0.67 else "bottom")
    return "center" if (h == "center" and v == "middle") else f"{v}-{h}"


# ── Run as a blocking server ─────────────────────────────────────


def run_server():
    """Start the barehands HTTP server in this process (blocks forever).

    This runs barehands' own ``server.py`` in-process via ``runpy`` so the
    server logs print to THIS terminal and Ctrl+C stops it directly.
    """
    import runpy

    server_path = Path(BAREHANDS_DIR) / "server.py"
    console.print(f"[bold]Starting barehands server[/] ({server_path})…")
    console.print(f"  tracker: http://{BAREHANDS_HOST}:{BAREHANDS_PORT}/stage.html")
    console.print(f"  render:  http://{BAREHANDS_HOST}:{BAREHANDS_PORT}/stage.html?role=render")
    console.print("[dim]Ctrl+C to stop.[/]")
    runpy.run_path(str(server_path), run_name="__main__")


if __name__ == "__main__":
    run_server()
