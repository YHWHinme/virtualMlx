# IMPLEMENTATION: Wiring barehands into a Voice Agent

> This document explains how to integrate barehands' hand-tracked glass interface into a voice agent pipeline like Voice Loop. It covers the architectural seams, the state protocol, the command protocol, and the concrete code changes needed to give a voice agent hands and a face on the glass board.

**Assumed base:** A working voice agent with the Voice Loop architecture — mic capture, STT, LLM generation, TTS output. The LLM backend is abstracted as "your LLM client" (could be a local model, a remote API, a LangChain chain, etc.).

---

## 1. What You're Building

After integration, the user has:

1. **Voice interaction** — speak to the agent, hear responses (existing Voice Loop pipeline)
2. **A glass board** — their notes, images, and 3D models float on screen, manipulated by hand gestures
3. **A ring face** — the on-screen ring reflects what the agent is doing (idle, listening, thinking, speaking) with real audio waveform data
4. **Agent hands** — the agent can stage cards, images, and models on the board in response to voice commands
5. **Agent eyes** — the agent can read what's on the board before responding

```
User speaks → Mic → STT → LLM ──┬──→ TTS → Speakers
                                  │
                                  ├──→ write state/state (the face)
                                  ├──→ write state/wave.json (the waveform)
                                  ├──→ POST /cmd (put things on the board)
                                  └──→ GET /state (see what's on the board)
```

---

## 2. The Three Seams

barehands communicates through three dead-simple protocols. Your voice agent needs to touch all three:

### Seam 1: The Ring State (face)

**Protocol:** Write tiny files to `state/` directory. The ring polls `GET /orb` every 500ms.

| File | Format | When to write |
|------|--------|---------------|
| `state/state` | Plain text: `idle`, `listening`, `thinking`, `speaking` | On every pipeline state transition |
| `state/mood.json` | `{"mood": "green"\|"amber"\|"red", "ts": <unix>}` | On mood changes (optional) |
| `state/wave.json` | `{"samples": [0..1 × 64], "ts": <unix>}` | Every ~100ms during TTS playback |

### Seam 2: The Board Commands (hands)

**Protocol:** `POST http://127.0.0.1:<port>/cmd` with JSON body.

```json
{"a": "add_card", "title": "Hello", "body": "The agent says hi"}
{"a": "present", "title": "The Plan", "body": "Three steps..."}
{"a": "add_img", "src": "misc/diagram.png"}
{"a": "hand", "src": "models/engine.glb"}
{"a": "explode"}
{"a": "clear"}
{"a": "reset"}
```

### Seam 3: The Board State (eyes)

**Protocol:** `GET http://127.0.0.1:<port>/state` returns the tracker's last scene heartbeat.

---

## 3. Integration Architecture

### 3.1 Process Model

barehands runs as a **separate process** alongside your voice agent. They communicate over localhost HTTP. No shared memory, no imports, no coupling.

```
Process 1: voice_agent.py          Process 2: barehands server.py
┌──────────────────────────────┐   ┌──────────────────────────────┐
│  Mic → STT → LLM → TTS      │   │  stage.html (tracker)        │
│                              │   │  stage.html?role=render      │
│  Writes: state/state         │   │                              │
│  Writes: state/wave.json     │   │  Reads: /orb (ring state)    │
│  POSTs: /cmd (board commands)│   │  POSTs: /state (scene)       │
│  GETs:  /state (board eyes)  │──►│  Serves: notes, media, etc.  │
└──────────────────────────────┘   └──────────────────────────────┘
         localhost HTTP only
```

**Why separate processes?** barehands is stdlib Python + a browser page. Your voice agent is a completely different stack (numpy, torch, ML models, audio I/O). Keeping them separate means:
- barehands never crashes your voice agent and vice versa
- You can develop and debug each independently
- The HTTP boundary is trivially testable with `curl`

### 3.2 The Bridge Module

Create a `board.py` module in your voice agent project that wraps the three seams:

```python
"""board.py — bridge between the voice agent and the barehands board."""

import json
import time
import urllib.request
from pathlib import Path

BAREHANDS_DIR = Path(__file__).parent.parent / "barehands"  # adjust path
BAREHANDS_PORT = 8794
STATE_DIR = BAREHANDS_DIR / "state"

# ── Seam 1: Ring State ─────────────────────────────────────────

def set_state(state: str):
    """Write the agent's state to the ring. state: idle|listening|thinking|speaking"""
    (STATE_DIR / "state").write_text(state)

def set_mood(mood: str):
    """Write mood: green|amber|red"""
    (STATE_DIR / "mood.json").write_text(
        json.dumps({"mood": mood, "ts": time.time()})
    )

def set_wave(samples: list[float]):
    """Write audio waveform samples (0..1, up to 64 values)."""
    (STATE_DIR / "wave.json").write_text(
        json.dumps({"samples": samples[:64], "ts": time.time()})
    )

# ── Seam 2: Board Commands ─────────────────────────────────────

def cmd(action: str, **kwargs) -> bool:
    """Send a command to the board. Returns True on success."""
    payload = {"a": action, **kwargs}
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{BAREHANDS_PORT}/cmd",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 204
    except Exception:
        return False

def add_card(title: str, body: str = "", file: str = "", open: bool = False,
             hand: bool = False) -> bool:
    return cmd("add_card", title=title, body=body, file=file, open=open, hand=hand)

def add_img(src: str, title: str = "") -> bool:
    return cmd("add_img", src=src, title=title)

def present(title: str, body: str = "", src: str = "", file: str = "",
            open: bool = False) -> bool:
    return cmd("present", title=title, body=body, src=src, file=file, open=open)

def hand(src: str, title: str = "") -> bool:
    """Deliver an item to the user's reach."""
    return cmd("hand", src=src, title=title)

def give(src: str, title: str = "") -> bool:
    """Deliver an item INTO the user's hand."""
    return cmd("give", src=src, title=title)

def explode() -> bool:
    return cmd("explode")

def assemble() -> bool:
    return cmd("assemble")

def clear() -> bool:
    return cmd("clear")

def reset() -> bool:
    return cmd("reset")

# ── Seam 3: Board State ────────────────────────────────────────

def get_board_state() -> dict | None:
    """Read what's currently on the board."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{BAREHANDS_PORT}/state",
            headers={"Cache-Control": "no-store"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None

def describe_board() -> str:
    """Human-readable summary of what's on the board (for the LLM)."""
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
            lines.append(f"  - card \"{title}\" @ {pos}{grabbed}")
        elif t == "panel":
            lines.append(f"  - open note \"{title}\" @ {pos}{grabbed}")
        elif t in ("img", "model"):
            lines.append(f"  - {t} \"{src}\" @ {pos}{grabbed}")
        elif t == "orb":
            lines.append(f"  - orb \"{title}\"")
        else:
            lines.append(f"  - {t} \"{title}\" @ {pos}{grabbed}")
    return "\n".join(lines)

def _zone(x, y):
    h = "left" if x < 0.33 else ("center" if x < 0.67 else "right")
    v = "top" if y < 0.33 else ("middle" if y < 0.67 else "bottom")
    return "center" if (h == "center" and v == "middle") else f"{v}-{h}"
```

---

## 4. Wiring the Voice Pipeline

### 4.1 State Transitions — The Ring Face

Hook into your voice agent's main loop to write state at each phase:

```python
# In your main loop:

import board

while True:
    # ── LISTENING ──
    board.set_state("listening")
    audio = capture_speech()  # your existing mic capture + VAD

    # ── THINKING ──
    board.set_state("thinking")
    board.set_mood("amber")
    transcript = your_stt(audio)

    # ── GENERATING ──
    # The LLM might want to put things on the board as part of its response.
    # See section 5 for how to give the LLM board awareness.
    response = your_llm(transcript)

    # ── SPEAKING ──
    board.set_state("speaking")
    board.set_mood("green")

    # During TTS, stream the waveform to the ring
    for sentence in split_sentences(response):
        samples = your_tts(sentence)
        # Extract waveform samples for the ring visualization
        board.set_wave(extract_waveform(samples))
        play_audio(samples)

    # ── IDLE ──
    board.set_state("idle")
```

### 4.2 Waveform Streaming

The ring reads `state/wave.json` every ~100ms during `speaking` state and only uses it if the timestamp is less than 0.6s old. You need to update it continuously during TTS playback:

```python
def play_with_waveform(audio_samples, sample_rate):
    """Play audio while streaming waveform to the ring."""
    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    for i in range(0, len(audio_samples), chunk_size):
        chunk = audio_samples[i:i + chunk_size]
        # Compute 64 amplitude values from this chunk
        step = max(1, len(chunk) // 64)
        samples = [abs(chunk[j]) for j in range(0, len(chunk), step)][:64]
        board.set_wave(samples)
        # Write this audio chunk to your output stream
        output_stream.write(chunk)
    # Clear the wave after speaking
    board.set_wave([])
```

### 4.3 The Startup Sequence

```python
import subprocess
import time
import board

def start_barehands():
    """Start the barehands server if it's not already running."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8794/config", timeout=1)
        print("  barehands server already running")
        return None  # already running, don't manage the process
    except Exception:
        pass

    proc = subprocess.Popen(
        ["python3", "server.py"],
        cwd=str(BAREHANDS_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for it to come up
    for _ in range(20):
        try:
            urllib.request.urlopen("http://127.0.0.1:8794/config", timeout=0.5)
            print("  barehands server started")
            return proc
        except Exception:
            time.sleep(0.25)
    print("  ⚠ barehands server failed to start")
    return proc
```

---

## 5. Giving the LLM Board Awareness

The LLM needs to know two things:
1. **What the board can do** (so it knows when to stage something)
2. **What's on the board right now** (so it doesn't talk about things that aren't there)

### 5.1 System Prompt Augmentation

Add this block to your system prompt (equivalent to Voice Loop's `SOUL.md`):

```markdown
## The Glass Board

A hand-tracked glass board runs on this machine. The user moves notes, images,
and 3D models on screen with their bare hands. You have hands and eyes on it.

### When to use the board
When the user asks to SEE something ("show me", "put it up", "pull up my notes
on X"), don't answer with a wall of text: find the thing, put it on the glass,
and say what you put up. The board is your show-and-tell.

### Board commands (your hands)
Use the board_command function to interact with the board:
- present(title, body): spotlight something center stage
- add_card(title, body): stage a card
- add_img(src): stage an image from media/
- hand(src): deliver to the user's reach
- give(src): deliver INTO the user's hand
- explode(): part a 3D model
- clear(): wipe the board
- reset(): ring center stage

### Board state (your eyes)
Use describe_board() before commenting on what's on the board. The user moves
things by hand, so never trust memory.

### The airlock law
Only files inside the media/ folder can be staged. To show a new image, it must
already be in media/.
```

### 5.2 Tool / Function Definitions

If your LLM client supports tool calling (function calling), define these:

```python
board_tools = [
    {
        "name": "present",
        "description": "Put something center stage on the glass board, spotlighted. Use when the user asks to see something.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Content to display"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "add_card",
        "description": "Stage a card on the glass board.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "describe_board",
        "description": "Read what's currently on the glass board. Call this before commenting on the board's contents.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "add_image",
        "description": "Stage an image from the media/ folder on the board.",
        "parameters": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "Path relative to media/, e.g. 'misc/diagram.png'"},
                "title": {"type": "string"},
            },
            "required": ["src"],
        },
    },
    {
        "name": "explode_model",
        "description": "Explode a 3D model on the board into its component parts.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "clear_board",
        "description": "Remove everything from the glass board.",
        "parameters": {"type": "object", "properties": {}},
    },
]
```

### 5.3 Tool Dispatch

After the LLM responds, check for tool calls and dispatch them:

```python
def handle_tool_calls(tool_calls):
    """Dispatch LLM tool calls to the board bridge."""
    results = []
    for call in tool_calls:
        name = call["name"]
        args = call.get("arguments", {})

        if name == "present":
            ok = board.present(args.get("title", ""), args.get("body", ""))
            results.append({"role": "tool", "content": "presented" if ok else "failed"})

        elif name == "add_card":
            ok = board.add_card(args.get("title", ""), args.get("body", ""))
            results.append({"role": "tool", "content": "added" if ok else "failed"})

        elif name == "describe_board":
            desc = board.describe_board()
            results.append({"role": "tool", "content": desc})

        elif name == "add_image":
            ok = board.add_img(args.get("src", ""), args.get("title", ""))
            results.append({"role": "tool", "content": "staged" if ok else "failed — file not in media/"})

        elif name == "explode_model":
            ok = board.explode()
            results.append({"role": "tool", "content": "exploded" if ok else "no model on board"})

        elif name == "clear_board":
            ok = board.clear()
            results.append({"role": "tool", "content": "cleared" if ok else "failed"})

        else:
            results.append({"role": "tool", "content": f"unknown tool: {name}"})

    return results
```

---

## 6. Notes Orb Integration

If the user has an Obsidian vault or notes folder configured as a barehands orb, the LLM can reference those notes:

### 6.1 Reading Notes for Context

```python
import urllib.request
import urllib.parse

def read_note(orb_index: int, rel_path: str) -> str | None:
    """Read a note from a barehands notes orb."""
    encoded = urllib.parse.quote(f"{orb_index}/{rel_path}")
    try:
        req = urllib.request.Request(f"http://127.0.0.1:8794/note?f={encoded}")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode("utf-8")
    except Exception:
        return None

def get_note_tree(orb_index: int = 0) -> dict | None:
    """Get the folder tree of a notes orb."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:8794/tree?orb={orb_index}")
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None
```

### 6.2 Opening a Note on the Board

```python
def show_note(orb_index: int, rel_path: str, title: str = "") -> bool:
    """Stage a note card and open it as a panel on the board."""
    file_ref = f"{orb_index}/{rel_path}"
    return board.add_card(
        title=title or rel_path.split("/")[-1].replace(".md", ""),
        body="tap to open",
        file=file_ref,
        open=True,  # unfurl the panel immediately
    )
```

---

## 7. Media Airlock — Staging New Content

The AI can only stage files that already exist inside `media/`. To show something new:

```python
import shutil
from pathlib import Path

MEDIA_DIR = BAREHANDS_DIR / "media"

def stage_image(source_path: str, dest_subfolder: str = "misc") -> str | None:
    """Copy an image into the media airlock and return its board src."""
    src = Path(source_path)
    if not src.is_file():
        return None
    dest_dir = MEDIA_DIR / dest_subfolder
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return f"{dest_subfolder}/{src.name}"

# Usage:
src = stage_image("/path/to/screenshot.png")
if src:
    board.add_img(src, title="Your screenshot")
```

---

## 8. The Full Integrated Loop

Putting it all together:

```python
import board
import time

def main():
    # Start barehands server
    bh_proc = start_barehands()

    # Load your models (STT, LLM, TTS)
    stt = load_stt()
    llm = load_llm()
    tts = load_tts()

    # Set initial state
    board.set_state("idle")
    board.set_mood("green")

    # Greeting
    board.set_state("speaking")
    greeting = llm.generate("Greet the user in one short sentence.")
    audio = tts.synthesize(greeting)
    play_with_waveform(audio, tts.sample_rate)
    board.set_state("idle")

    history = []

    try:
        while True:
            # ── LISTEN ──
            board.set_state("listening")
            audio = capture_speech()  # your VAD + mic capture

            # ── TRANSCRIBE ──
            board.set_state("thinking")
            board.set_mood("amber")
            transcript = stt.transcribe(audio)
            print(f"  [{transcript}]")

            # ── BUILD CONTEXT ──
            messages = build_messages(history, transcript)
            # Inject board state so the LLM knows what's up
            board_desc = board.describe_board()
            if board_desc != "The board is empty.":
                messages.insert(1, {
                    "role": "system",
                    "content": f"Current board state:\n{board_desc}"
                })

            # ── GENERATE (with tool calls) ──
            response = llm.generate(messages, tools=board_tools)

            # Handle any tool calls (board commands)
            if response.tool_calls:
                tool_results = handle_tool_calls(response.tool_calls)
                # Re-generate with tool results
                messages.append(response.message)
                messages.extend(tool_results)
                response = llm.generate(messages)

            # ── SPEAK ──
            board.set_state("speaking")
            board.set_mood("green")
            full_text = response.text

            for sentence in split_sentences(full_text):
                audio = tts.synthesize(sentence)
                play_with_waveform(audio, tts.sample_rate)

            # ── IDLE ──
            board.set_state("idle")
            history.append({"user": transcript, "assistant": full_text})

    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        board.set_state("idle")
        if bh_proc:
            bh_proc.terminate()
```

---

## 9. Configuration

### 9.1 `barehands.json` — Point at the User's Notes

```json
{
  "name": "Jarvis",
  "port": 8794,
  "orbs": [
    { "title": "Notes", "path": "/home/user/Documents/MyVault", "kind": "notes" },
    { "title": "Props", "path": "media", "kind": "media" }
  ]
}
```

### 9.2 The Voice Agent Config

Your voice agent needs to know where barehands lives:

```python
# In your agent's config or environment variables:
BAREHANDS_DIR = "/path/to/barehands"
BAREHANDS_PORT = 8794
```

### 9.3 Startup Order

1. Start barehands server first (it's instant — stdlib Python)
2. Start your voice agent (it loads ML models — takes seconds)
3. The voice agent connects to barehands over HTTP on startup

If barehands isn't running, the voice agent should degrade gracefully — skip board operations, continue with voice-only interaction.

---

## 10. Testing the Integration

### 10.1 Manual Tests

```bash
# 1. Start barehands
cd /path/to/barehands
python3 server.py

# 2. Open the tracker in Chrome
open http://127.0.0.1:8794/stage.html

# 3. Test the seams from the command line

# Ring state
echo thinking > state/state

# Board command
curl -X POST http://127.0.0.1:8794/cmd \
  -H "Content-Type: application/json" \
  -d '{"a":"add_card","title":"TEST","body":"hello from curl"}'

# Board state
curl http://127.0.0.1:8794/state | python3 -m json.tool

# Ring state
curl http://127.0.0.1:8794/orb | python3 -m json.tool
```

### 10.2 Integration Tests

```python
def test_ring_state():
    board.set_state("thinking")
    time.sleep(0.6)  # wait for ring poll
    resp = json.loads(urllib.request.urlopen("http://127.0.0.1:8794/orb").read())
    assert resp["state"] == "thinking"

def test_board_command():
    ok = board.add_card("Test Card", "Integration test")
    assert ok
    state = board.get_board_state()
    titles = [i["title"] for i in state["items"] if i["type"] == "card"]
    assert "Test Card" in titles

def test_board_eyes():
    board.clear()
    board.add_card("Visible Card", "should be seen")
    desc = board.describe_board()
    assert "Visible Card" in desc

def test_media_airlock():
    ok = board.add_img("../../etc/passwd")  # should fail
    assert not ok
```

---

## 11. What Changes in Voice Loop Specifically

If you're modifying `voice_loop_mac.py` directly:

### 11.1 Add board.py as a module

Import it at the top of `main()`:
```python
import board
```

### 11.2 Hook the state transitions

In `process_utterance()`, add state writes:
```python
def process_utterance(audio, history):
    board.set_state("thinking")
    # ... existing transcription ...
    heard = transcribe(audio)

    board.set_state("speaking")
    # ... existing TTS ...
    # Add waveform streaming in play_tts_stream()

    board.set_state("idle")
```

### 11.3 Inject board state into the system prompt

In `_sys_messages()`:
```python
def _sys_messages():
    sp = load_system_prompt(include_memory=args.memory)
    board_desc = board.describe_board()
    if board_desc and board_desc != "The board is empty.":
        sp += f"\n\nCurrent board state:\n{board_desc}"
    return [{"role": "system", "content": sp}] if sp else []
```

### 11.4 Add a `--board` CLI flag

```python
ap.add_argument("--board", action=argparse.BooleanOptionalAction,
                default=True, help="barehands glass board integration")
```

### 11.5 Waveform streaming in play_tts_stream()

Inside the `for i in range(0, len(data), 4096)` loop in `play_tts_stream()`:
```python
# Extract waveform samples for the ring
chunk = data[i:i+4096].flatten()
step = max(1, len(chunk) // 64)
samples = [float(abs(chunk[j])) for j in range(0, len(chunk), step)][:64]
board.set_wave(samples)
```

---

## 12. Failure Modes and Graceful Degradation

| Failure | Symptom | Behavior |
|---------|---------|----------|
| barehands not running | HTTP connection refused | Skip all board ops, voice-only mode |
| Server started but no tracker | `/state` returns `{}` | Board commands queue but never deliver |
| Media file missing | `/cmd` returns 400 | `board.add_img()` returns False |
| Path escape attempt | `/cmd` returns 400 | Airlock blocks it |
| State file write fails | Ring stays idle | No crash, ring just doesn't animate |
| Wave file stale (>0.6s) | Ring stops showing waveform | Normal — speaking state ended |

The voice agent should **never crash** because of barehands. Every board operation is wrapped in try/except with a 2-second timeout.

---

## 13. Summary — What Each File Does

| File | In your project | In barehands |
|------|----------------|--------------|
| `board.py` | **New** — bridge module wrapping the three seams | Doesn't exist |
| `main.py` / `voice_loop_mac.py` | **Modified** — state writes + board tool dispatch | Doesn't exist |
| `SOUL.md` | **Modified** — add board awareness to system prompt | Doesn't exist |
| `server.py` | Not touched | Runs as-is |
| `stage.html` | Not touched | Loaded in browser |
| `barehands.json` | Not touched (user configures) | Points at their notes vault |
| `state/` | **Written to** by your agent | **Read from** by the ring |
| `media/` | **Written to** when staging new content | **Read from** by the board |
| `bin/board.sh` | Alternative to `board.py` for shell-based agents | Ships with barehands |
