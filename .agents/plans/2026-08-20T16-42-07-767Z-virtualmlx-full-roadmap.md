# VirtualMlx — Full Roadmap

The whole arc, from where you are now to the distant Rust port. Each phase
has concrete tasks, the files they touch, and a "done when" check.

---

## Status Snapshot — what's verified done

### MVP Voice Loop ✓
- Silero VAD (ONNX) — speech gate, 16 kHz, 32 ms chunks
- Smart Turn v3 (ONNX) — conversation-end estimation, falls back to silence
- Moonshine STT — local CPU transcription
- Kokoro TTS v1.0 (ONNX) — `am_michael` male voice, volume gain 1–20
- Ollama `gemma4:cloud` via LangChain — streaming
- `SentenceAccumulator` — low-latency per-sentence TTS dispatch
- Conversation memory — last 10 turns via `MemorySaver` checkpointer

### LangChain Replacement ✓
- `ChatOllama` (langchain-ollama) instead of raw `ollama` package
- `create_agent` (langgraph) — automatic tool-calling loop
- Per-thread checkpointed memory (`thread_id="virtualmlx"`)
- MCP websearch via `langchain-mcp-adapters` → parallel.ai (streamable-HTTP, anonymous or Bearer)

### Barehands Ring Face ✓
- `board.py` writes `state/state` (idle/listening/thinking/speaking)
- `board.py` writes `state/mood.json` (green/amber/red)
- `board.py` writes `state/wave.json` (64-sample 0..1 waveform during TTS)
- `main.py` drives all three transitions through the loop

### Local File Tools ✓ (just shipped — matches `todo.txt`)
- `UTILITIES/file_tools.py` — sandboxed `create_file` / `edit_file` / `read_file` / `list_output_files` confined to `OUTPUT/`
- Path-traversal + suffix allowlist guards
- Wired into the agent; described in `SYSTEM.md`

### Bridge module ✓
- `board.py` implements all three barehands seams (ring face, board cmds, board eyes) with graceful degradation

**The gap:** `board.py`'s command functions (`present`, `add_card`, `add_img`,
`describe_board`, `clear`, `explode`, `assemble`) are **built but not wired
into the agent**. `model.py` loads websearch + file tools only, and
`SYSTEM.md` has zero board awareness. → **Phase 1.**

---

## Phase 1 — Finish Barehands Integration  ← we are here

Goal: give the agent *hands* and *eyes* on the glass board, so "show me X"
lands a card/panel/image on the board instead of a wall of spoken text.

### 1A. Wire board tools into the agent
- New `UTILITIES/board_tools.py`: wrap `board.present / add_card / add_img /
  describe_board / clear / explode / assemble` as `@tool` functions with
  docstrings written for the LLM (when-to-use guidance, the airlock law).
- In `UTILITIES/model.py` `create()`: `tools = tools + file_tools + board_tools`.
- Keep graceful degradation: if `board.is_connected()` is False, skip board
  tools entirely (don't offer the LLM tools it can't actually execute).
- **Done when:** saying "show me a card called Hello with body World" stages
  a card on the board and the agent *speaks* "I put up a card" instead of
  reciting the text.

### 1B. Give the LLM board awareness in the system prompt
- Add a **Glass Board** block to `SYSTEM.md` (the show-me verb, the airlock
  law, "call `describe_board` before commenting on what's up, the user moves
  things by hand so never trust memory").
- **Done when:** `SYSTEM.md` references the board, and the agent calls
  `describe_board` before answering "what's on the board right now."

### 1C. Hub swiping — gestures feed the AI context
This is the inbound seam barehands doesn't have yet. The agent can *read*
the board, but there's no "the user handed me this note" event. Two designs:
- **Polling approach (fastest):** poll `GET /state` for items that are
  *grabbed into the ring zone* (x≈0.5, y≈0.5, `g` flag + in-hub radius).
  When a note enters that zone, read its content via the `/note` endpoint
  and inject it as a system message on the next turn.
- **Event approach (cleaner, needs barehands change):** add a small
  `POST /agent-event` (or write a `state/inbox.json` file) in barehands
  that the ring's gesture handler writes on a "fling-into-hub" gesture;
  `board.py` drains it each turn.
- Recommend: **start with polling** (no barehands fork needed), graduate to
  the event file if it feels laggy.
- **Done when:** the user flings a note into the ring and the agent's next
  reply references that note's content unprompted.

### 1D. Media airlock for staging new content
- `board.stage_image(src_path)` — copy a generated/downloaded image into
  `barehands/media/misc/` and return its board `src`.
- Wire as a tool so the agent can save an image it found/built and then
  `add_img` it. Pair with the existing image-capable web tools.
- **Done when:** "show me a picture of a redshift shader" downloads, stages,
  and displays the image end-to-end.

---

## Phase 2 — Barehands Augmentation

### 2A. Tool calling for the Jarvis UI (tools TBD)
- The `todo.txt` line: *"Include new tool call abilities e.g. edit and
  powerpoint creation."* File tools cover edit; add:
  - **Document generation**: PowerPoint via `python-pptx`, maybe
    LibreOffice headless for docx/pdf — outputs land in `OUTPUT/` then get
    staged on the board via the airlock.
  - **Calendar / shell** (decide scope — read-only first).
- Each new tool family lives in its own `UTILITIES/*_tools.py` and is
  composed into `model.py`'s tool list.

### 2B. Machine control via the interface
- `UTILITIES/machine_tools.py`: `open_app(name)`, `open_url(url)`,
  `focus_window(title)` — XDG / xdotool on Linux.
- Stage these as tools; guard with an allowlist (no arbitrary `shell=True`).
- **Done when:** "open Steam" or "open the browser to YouTube" works by voice.

### 2C. Barge-in with WebRTC AEC3 echo cancellation
- The current loop can't be interrupted while speaking. Add:
  - AEC3 echo cancellation so the mic ignores the agent's own TTS output.
  - A "wake/interrupt" detector (lower VAD threshold during `speaking`)
    that cancels the in-flight TTS stream and re-enters `listening`.
- This is the hardest single item — needs a duplex stream or a virtual
  loopback. Scope a spike first.
- **Done when:** talking over the agent cuts it off mid-sentence and it
  re-listens.

---

## Phase 3 — Daemonize

- Run the agent as a background service that survives terminal close:
  - systemd user unit (`~/.config/systemd/user/virtualmlx.service`), or
  - a `mise`/`supervisord` wrapper, or
  - a simple `nohup` + PID-file launcher.
- Healthcheck endpoint (reuse barehands-style HTTP) so the board ring can
  show "agent down" in red.
- Auto-restart on model-load failure; keep the board server independent.
- **Done when:** reboot → agent is up and the ring is breathing, no terminal.

---

## Phase 4 — Memory (mem0)

- Add `mem0` for persistent cross-session memory (the roadmap's open item).
  - Extract salient facts/preferences from each turn.
  - Retrieve top-k memories into the system prompt each turn.
- Decide store: local SQLite/Qdrant vs. the existing parallel.ai hook.
- Keep the `MemorySaver` short-term checkpointer for *this* conversation;
  mem0 is the *long-term* layer.
- **Done when:** "my name is X" in one session → "hi X" unprompted next boot.

---

## Phase 5 — Rust Port (distant future)

- Port the pipeline to Rust once the Python design is frozen:
  - Audio I/O → `cpal`
  - VAD / STT / TTS → ONNX Runtime Rust bindings (`ort`)
  - LLM → keep Ollama HTTP client (no port needed) or `reqwest`
  - Agent loop → `tokio` async
- Milestone order: listener → transformer → board → model → orchestrator.
- **Done when:** feature-parity with the Python agent, lower memory + latency,
  single binary.

---

## How to "continue with it"

The natural next move is **Phase 1A + 1B together** (one cohesive change):
wire `board_tools.py` into the agent and add the Glass Board block to
`SYSTEM.md`. That unblocks the whole "agent has hands" experience and is the
single highest-leverage step on the roadmap.

Pick where to start and I'll build it.
