# INSIGHT: barehands — Complete Technical Breakdown

> This document explains exactly how barehands processes hand gestures from webcam to on-screen glass cards, how the server mediates state between the tracker and render pages, and how an AI assistant drives the board. Every endpoint, every gesture gate, every scene object — nothing is abstracted away.

---

## 1. Project Shape

A two-file web application (~3,100 lines of HTML/CSS/JS + ~250 lines of Python) that turns a webcam into a **hand-tracked glass interface**. Notes, images, and 3D models float over the camera feed as translucent cards. The user manipulates them with bare fingers — no headset, no controllers. An AI assistant can drive the board via HTTP, making cards appear, staging images, and reading what's on the board.

```
server.py         ← stdlib-only Python HTTP server (ThreadingHTTPServer)
stage.html        ← the entire frontend: MediaPipe hand tracking + three.js 3D + all gestures
barehands.json    ← config: assistant name, port, orb paths
bin/board.sh      ← AI's hands: POST a command to the board
bin/board-state.sh← AI's eyes: GET what's on the board
state/            ← runtime files the assistant writes (state, mood.json, wave.json)
media/            ← the props airlock: only files here can stage on the board
```

**Zero dependencies.** The server is pure Python stdlib. The frontend loads MediaPipe and three.js from CDNs on first run, then caches.

---

## 2. The Architecture — Two Pages, One Scene

barehands uses a **split-page architecture** connected through the server:

```
┌─────────────────────────────┐     POST /state (~45Hz)     ┌──────────────────────┐
│   TRACKER PAGE              │ ──────────────────────────►  │   server.py          │
│   stage.html (default)      │                              │   (state store +     │
│                             │  ← commands in response ──── │    command queue)    │
│   • Owns the webcam         │                              └──────────┬───────────┘
│   • Runs MediaPipe          │                                         │
│   • Gesture detection       │     GET /state (mirror)                 │
│   • Physics simulation      │ ◄───────────────────────────────────────┘
│   • Scene author of truth   │
└─────────────────────────────┘
                               POST /cmd (AI commands)
┌─────────────────────────────┐     GET /state (mirror)     ┌──────────────────────┐
│   RENDER PAGE               │ ◄─────────────────────────► │   server.py          │
│   stage.html?role=render    │                              │                      │
│                             │                              │   AI assistant       │
│   • No camera               │                              │   (board.sh / curl)  │
│   • Truly transparent       │                              └──────────────────────┘
│   • Mirrors tracker scene   │
│   • Goes into OBS as        │
│     browser source          │
└─────────────────────────────┘
```

### Why two pages?

The tracker page owns the camera (needed for MediaPipe) but can't produce **true transparency** for OBS compositing. The render page has no camera — its `body` is transparent, and OBS composites it as a browser source with real alpha. The tracker streams normalized scene state to the server at ~45Hz, and the render page mirrors it.

---

## 3. The Server — `server.py` (stdlib only)

### 3.1 HTTP Endpoints

| Method | Path | Purpose | Who uses it |
|--------|------|---------|-------------|
| `POST` | `/state` | Tracker's heartbeat: sends full scene state, receives queued commands | Tracker page (~45Hz) |
| `GET` | `/state` | Read the latest scene state | Render page, board-state.sh |
| `POST` | `/cmd` | Queue a board command (AI → board) | board.sh, AI assistants |
| `GET` | `/config` | Returns assistant name + orb definitions | stage.html (builds ring) |
| `GET` | `/tree?orb=N` | A notes orb's folder tree (jailed, .md only) | stage.html (orb bloom) |
| `GET` | `/note?f=N/path` | One note's text content (jailed, .md only) | stage.html (open note) |
| `GET` | `/props` | The media airlock as a browsable tree | stage.html (props orb) |
| `GET` | `/orb` | The assistant's live state (idle/thinking/speaking) | stage.html (ring animation) |
| `GET` | `/media/*` | Static files from the media airlock | stage.html (images/models) |
| `GET` | `/stage.html` | The page itself (with no-store caching) | Browser |

### 3.2 The State Store

The server holds two in-memory variables:

```python
_STATE = b"{}"    # latest scene state: tracker POSTs, render GETs
_CMDS = []        # queued board commands (AI → tracker)
```

**The heartbeat is also the command channel.** When the tracker POSTs to `/state`, the server:
1. Stores the new scene state
2. Returns up to 8 queued commands from `_CMDS`
3. Clears those commands from the queue

```python
def do_POST(self):
    if self.path == "/state":
        _STATE = body              # store the tracker's scene
        out = json.dumps(_CMDS[:8])
        del _CMDS[:8]              # drain what we're sending
        # respond with commands
    if self.path == "/cmd":
        cmd = json.loads(body)
        assert cmd.get("a") in _ALLOWED   # action allowlist
        _CMDS.append(cmd)
```

This means commands queued by the AI arrive at the tracker within one heartbeat cycle (~22ms).

### 3.3 The Action Allowlist

Only these actions are accepted via `/cmd`:

```python
_ALLOWED = ("add_img", "add_card", "clear", "reset", "hand", "give",
            "yank", "hover", "scroll_note", "widget", "explode", "assemble",
            "present")
```

Any command with an action not in this list is rejected with HTTP 400. This is a safety boundary — it prevents the AI from doing anything the board doesn't explicitly support.

### 3.4 The Media Airlock

The most important security feature. For commands that reference files (`add_img`, `hand`, `give`, `present`), the server enforces that the file is **actually inside `./media/`**:

```python
media = (HERE / "media").resolve()
target = (media / rel).resolve()
if media not in target.parents or not target.is_file():
    # path escape → try basename match inside media/
    hits = [p for p in media.rglob("*") if p.is_file() and p.name.lower() == name]
    if len(hits) != 1:
        raise ValueError("not in the media airlock")
    target = hits[0]
cmd["src"] = "/media/" + target.relative_to(media).as_posix()
```

If the exact path misses, a **unique basename match** anywhere inside `media/` self-heals a wrong-folder guess. Zero or multiple matches still 400. This means the AI can say `"src": "misc/logo.png"` even if the file is actually at `media/misc/logo.png`.

### 3.5 The Notes Jail

Notes orbs can point at any folder of markdown. But the `/note` and `/tree` endpoints enforce:
- The resolved path must be **inside the orb's configured root** (path traversal blocked)
- Only `.md` files are served
- `CLAUDE.md` is excluded (it's AI config, not a note)

```python
target = (root / rel).resolve()
if (root not in target.parents) or target.suffix != ".md" or not target.is_file():
    self.send_response(404)
```

### 3.6 The Ring's State Files

The assistant communicates its live state by writing tiny files into `./state/`:

| File | Content | Effect |
|------|---------|--------|
| `state/state` | One word: `idle` / `listening` / `thinking` / `speaking` | Ring animation state |
| `state/mood.json` | `{"mood": "green"\|"amber"\|"red", "ts": <unix>}` | Ring color (expires after 45s) |
| `state/wave.json` | `{"samples": [0..1 × 64], "ts": <unix>}` | Voice waveform (only read during `speaking`, expires after 0.6s) |

The `/orb` endpoint reads these files on every GET and returns a JSON snapshot. Missing files are fine — the ring just idles.

---

## 4. The Frontend — `stage.html` (~3,100 lines)

### 4.1 CDN Dependencies (loaded at runtime)

```javascript
import { HandLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
```

And three.js via import map:
```json
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
} }
```

These load on first visit, then the browser caches them. No build step, no bundler, no npm.

### 4.2 Item Types

Everything on the board is an **item** in the `items` array. Each item has a type:

| Type | What it is | Created by |
|------|-----------|------------|
| `card` | Glass card with title + body text | `add_card` command |
| `panel` | Opened markdown note (scrollable) | Tapping a card with a `file` |
| `img` | Image or video (with optional FX frame) | `add_img` command |
| `model` | 3D GLB/glTF model (solid or hologram) | `add_img` with `.glb` src |
| `orb` | Folder orb around the ring | Built from `/config` |
| `widget` | The assistant ring itself | Always present |
| `browser` | File browser panel | Orb tree navigation |

Item structure:
```javascript
{ el, id, type, def, x, y, scale, vx, vy,
  grabbedBy: [], ox, oy, flying, stretch, scrollY, body }
```

### 4.3 The Scene Objects

```javascript
let items = [];      // all board items (cards, panels, images, models, orbs)
let cursors = {};    // hand cursors keyed by hand index (0, 1, ...)
let landmarker = null; // MediaPipe HandLandmarker instance
```

### 4.4 The Main Loop

```javascript
function frame(now) {
  // 1. Detect hands (only when video frame is new)
  if (cam.currentTime !== lastVideoTs) {
    const res = landmarker.detectForVideo(cam, now);
    // Process each hand's landmarks...
  }

  // 2. Physics (every frame)
  physics(dt);

  // 3. Render 3D models (every frame)
  renderModels();

  // 4. Render the assistant ring (every frame)
  renderRings();

  // 5. Push state to server (~45Hz)
  pushState(now);

  requestAnimationFrame(frame);
}
```

---

## 5. Hand Tracking — MediaPipe

### 5.1 Camera Setup

```javascript
const constraints = {
  video: PORTRAIT
    ? { width: { ideal: RES[1] }, height: { ideal: RES[0] } }
    : { width: { ideal: RES[0] }, height: { ideal: RES[1] } }
};
const stream = await navigator.mediaDevices.getUserMedia(constraints);
cam.srcObject = stream;
```

Default capture: 1920×1080. MediaPipe downscales internally to its model input, so higher capture res only affects self-view sharpness, not tracking accuracy.

### 5.2 Landmark Processing

MediaPipe returns 21 landmarks per hand. The code computes:

- **`tip`** = landmark 8 (index fingertip) — the cursor position
- **`thumb`** = landmark 4 (thumb tip) — pinch partner
- **`span`** = distance from wrist (0) to middle knuckle (9) — the hand's own yardstick
- **`ratio`** = thumb↔index gap ÷ span — the raw pinch signal

The cursor position is smoothed with an EMA:
```javascript
const px = (tip.x + thumb.x) / 2, py = (tip.y + thumb.y) / 2;
cur.x += (px - cur.x) * 0.45;
cur.y += (py - cur.y) * 0.45;
```

### 5.3 Per-Finger Metrics

Every gesture gate uses **shape ratios** measured against the hand's own geometry (the `span`), not screen pixels. This is why gestures work at any camera distance:

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| `r` | thumb↔index gap ÷ span | Pinch openness |
| `c8,c12,c16,c20` | fingertip alignment vs base | Per-finger CURL |
| `f8,f12,f16,f20` | tip-from-wrist ÷ knuckle-from-wrist | Per-finger ARCH |
| `aspect` | span ÷ knuckle-row width | Palm orientation (frontal vs profile) |
| `tRel` | thumb↔ring-knuckle ÷ span | Profile pinch judge |
| `handUp` | (wrist.y - knuckle.y) ÷ span | Palm upness (1 = fingers up) |

---

## 6. Gesture Gates

### 6.1 The Pinch — "THE CONTRAST LAW"

The pinch is the primary interaction. It must distinguish a deliberate OK-sign pinch from a closed fist (which also brings thumb to index).

**The insight:** In a correct pinch, the index finger curls IN to the thumb while the other three fingers ARCH OUT past the knuckle circle. In a fist, all fingers fold in.

```javascript
const f8v = fR(8, 5);                           // index arch
const backMean = (fR(12,9) + fR(16,13) + fR(20,17)) / 3;  // back three mean

// THE CONTRAST LAW: backMean − f8v ≥ 0.18 in every correct sample
// ≤ 0.07 in every impostor. Cut at 0.18 with a 1.30 floor.
const contrast = backMean - f8v;
const isPinch = contrast >= 0.18 && backMean >= 1.30;
```

**Two orientation regimes:**
- **Frontal** (aspect > 2.0): contrast law governs
- **Profile** (aspect < 2.0): thumb distance from knuckle row governs (`tRel > 0.95`)
- **Either** signature admits — every correct pinch passes at least one

### 6.2 The Claw — Force Pull

The claw is a dramatic gesture: flash hand open, form a claw, aim at something across the screen, hold the strain for 2 seconds, then snap shut — the target rips through the air into your hand.

Detection uses:
- **Mouth floor** (`ratio > ~0.80`) — claw must be open enough
- **Per-finger curl ceilings** — fingers must be clawed, not flat
- **Pinky-out** (`c20`) — the characteristic claw splay
- **Aspect rail** — sanity check on palm shape
- **2-second hold** — prevents accidental triggers

### 6.3 The Clap — "THE PRAYER LAW"

Palms together, fingers up. Sweeps the board clean.

Detection:
- **Wrist + knuckle proximity** — both hands' wrists and knuckles must be close (fractions of window width)
- **Vertical fingers** (`handUp > 0.85`) — prayer requires fingers pointing up
- **Was-apart memory** — hands must have been separated before the clap (prevents sustained prayer from re-triggering)

### 6.4 The Throw / Fling

Velocity-based: the item's peak velocity over the last 10 frames must exceed 1300px/s. This is one of the few **screen-pixel-based** gates (not distance-proof ratios).

### 6.5 Tap vs Pinch-Hold

Timing-based: `tMs < 300` distinguishes a quick tap (open/close) from a sustained pinch (grab/drag). The dwell charge glow gives visual feedback during hold-to-rotate.

### 6.6 Two-Hand Scale

When two hands pinch the same item simultaneously, the distance between the two pinch points controls the item's scale. This is the only gesture that requires both hands on one object.

---

## 7. Physics

The `physics(dt)` function runs every frame:

- **Grabbed items** follow the cursor with spring physics
- **Released items** carry velocity from the throw (vx, vy)
- **Friction** decelerates flying items
- **Screen bounds** bounce or clamp items at edges
- **Two-hand stretch** computes the scale factor from inter-hand distance
- **Charge glow** builds when holding still while carrying (signals rotate-to-3D)

---

## 8. 3D Models — three.js

### 8.1 Loading

Models are loaded from GLB/glTF files inside the media airlock:

```javascript
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
```

Each model gets its own `THREE.Scene`, `THREE.WebGLRenderer` (with alpha for transparency), and `THREE.PerspectiveCamera`. The renderer draws into a `<canvas>` element that's positioned as the item's visual.

### 8.2 Two Modes

- **Solid**: standard PBR rendering with environment map, directional light, ambient light
- **Hologram** (`media/holo/`): custom `ShaderMaterial` that renders as a blue wireframe with rim lighting

### 8.3 Exploded View

The `applyExplode()` function separates a model's child meshes along their individual normals, controlled by the `ex` parameter (0 = assembled, 1 = fully exploded). An empty pinch dragged sideways scrubs the explode parameter continuously.

---

## 9. The Ring — Assistant Face

The ring is a `<canvas>` element rendered as a widget item. It reads `/orb` every 500ms:

```javascript
async function pollOrb() {
  const r = await fetch("/orb", { cache: "no-store" });
  const j = await r.json();
  // j.state: "idle" | "listening" | "thinking" | "speaking"
  // j.mood: "green" | "amber" | "red"
  // j.wave: [0..1 × 64] (only during "speaking")
}
```

The ring animation responds to state:
- **idle**: gentle breathing pulse
- **listening**: subtle glow expansion
- **thinking**: spinning arcs
- **speaking**: waveform visualization from `state/wave.json`

---

## 10. Command Processing — `applyCmds()`

When the tracker receives commands in the heartbeat response:

```javascript
function applyCmds(cmds) {
  (cmds || []).forEach(c => {
    if (c.a === "clear")        items.forEach(killItem);
    if (c.a === "reset")        loadTree().then(() => spawnStage(0.5, 0.42));
    if (c.a === "add_card")     { items.push(makeCard({...})); if (c.open) openPanel(it); }
    if (c.a === "add_img")      { items.push(isModelSrc(c.src) ? makeModel(...) : makeImage(...)); }
    if (c.a === "hand")         { deliver(it); }        // materialize at user's reach
    if (c.a === "give")         { give(it); }           // materialize + force-pull into hand
    if (c.a === "yank")         { /* shake + rip away */ }
    if (c.a === "hover")        { /* pulse for attention */ }
    if (c.a === "explode")      { t.ex = 1; }
    if (c.a === "assemble")     { t.ex = 0; }
    if (c.a === "scroll_note")  { /* glide the newest note panel */ }
    if (c.a === "present")      { /* spotlight one item, dim everything else */ }
    if (c.a === "widget")       { summonWidget(c.w); }
  });
}
```

### Delivery Mechanics

- **`add_card`**: materializes center-screen with an `in` animation (back-ease scale + 3D turn-in)
- **`hand`**: materializes at the user's reach (62% × 46% of viewport)
- **`give`**: materializes at reach, then force-pull flight INTO the user's live hand — lands caught-and-held
- **`present`**: the item flies center-stage, enlarged and spotlit, everything else dims (`.dimmed` class)

---

## 11. State Push — `pushState()`

The tracker streams its scene to the server at ~45Hz (throttled to every 22ms):

```javascript
function pushState(now) {
  if (now - lastPush < 22) return;
  const st = {
    cursors: [...],   // hand positions as window fractions
    items: items.map(i => ({
      id, type, title, src, x, y, scale, g, op, ch, sy, rx, ry, rz, ex, ...
    }))
  };
  fetch("/state", { method: "POST", body: JSON.stringify(st) })
    .then(r => r.json())
    .then(applyCmds);   // commands come back in the response
}
```

All coordinates travel as **window fractions** (0.0–1.0), so the tracker window and the render window can be different sizes.

---

## 12. The Render Page — `?role=render`

The render page:
1. Has no camera (no MediaPipe, no gestures)
2. Reads `/state` at ~45Hz and rebuilds the scene
3. Has a transparent `body` for OBS compositing
4. Flips X by default (mirror correction for broadcast)
5. Hides cursor rings with `&cursors=0`

---

## 13. Configuration — `barehands.json`

```json
{
  "name": "Assistant",
  "port": 8794,
  "orbs": [
    { "title": "Notes", "path": "/path/to/vault", "kind": "notes" },
    { "title": "Props", "path": "media",          "kind": "media" }
  ]
}
```

- `name`: displayed on the ring
- `port`: server port (default 8794)
- `orbs`: each becomes a glass orb around the ring. `notes` orbs can point at any markdown folder (including Obsidian vaults). The `media` orb is always the repo's `./media/` — the props airlock.

---

## 14. Audio — Synthesized Foley

Optional WebAudio foley gives glass objects sound:

```javascript
function _tone(freq, dur, gain, type, sweepTo, delay) { /* oscillator */ }
function _noise(dur, fLo, fHi, gain, sweepTo) { /* filtered noise */ }
```

Synthesized sounds: grab thud, release click, throw whoosh, clap sweep, materialize shimmer. Disabled by default (`&sound=1` enables) because they read as noise on camera.

---

## 15. URL Parameters

| Parameter | Effect |
|-----------|--------|
| `?role=render` | Render page mode (no camera, transparent) |
| `?mode=mirror\|overlay\|key` | Background mode (mirror=default, overlay=transparent, key=solid color) |
| `?key=magenta\|green\|blue\|#hex` | Chroma key color (for `mode=key`) |
| `?cam=<label>` | Pin a specific camera |
| `?res=WxH` | Capture resolution (default 1920×1080) |
| `?portrait=1` | Vertical capture (9:16) |
| `?cursors=0` | Hide cursor rings (render page) |
| `?mirror=1` | Disable X-flip on render page |
| `?ss=N` | Super-sample (2× or 3× for sharp cards) |
| `?sound=1` | Enable foley audio |

---

## 16. Complete Dependency Map

### CDN (loaded at runtime)

| Importance | Package | Role |
|:----------:|---------|------|
| **10** | MediaPipe Tasks Vision 0.10.14 | Hand tracking — 21 landmarks per hand at ~30Hz |
| **9** | three.js 0.160.0 | 3D model rendering (GLB/glTF), hologram shaders, bloom post-processing |
| **7** | GLTFLoader / DRACOLoader | 3D model file parsing |

### Server (Python stdlib)

| Importance | Module | Role |
|:----------:|--------|------|
| **10** | `http.server.ThreadingHTTPServer` | The HTTP server itself |
| **10** | `http.server.SimpleHTTPRequestHandler` | Static file serving + custom endpoints |
| **8** | `json` | All state serialization |
| **7** | `pathlib.Path` | File path resolution, jail enforcement |
| **6** | `urllib.parse` | Query string parsing for `/tree`, `/note` |
| **4** | `time` | Mood/wave timestamp validation |

### No Python third-party packages. Zero. The entire server is stdlib.

---

## 17. Data Flow — A Complete Interaction

```
User waves hand at webcam
  ↓ MediaPipe detects 21 landmarks
  ↓ Landmarks processed to cursor position + pinch ratio
  ↓ Pinch detected (CONTRAST LAW passes)
  ↓ Cursor overlaps a card's bounding box
  ↓ Tap detected (tMs < 300)
  ↓ Card has a .file → fetch("/note?f=0/path/to/note.md")
  ↓ Server resolves path against orb jail root
  ↓ Returns markdown text
  ↓ Panel created with scrollable content
  ↓ physics() positions it, renderRings() draws
  ↓ pushState() streams to server at 45Hz
  ↓ Render page mirrors the panel
  ↓ OBS composites it over camera feed
```

```
AI wants to show something
  ↓ AI runs: board.sh '{"a":"present","title":"THE PLAN","body":"..."}'
  ↓ board.sh → curl POST /cmd → server appends to _CMDS
  ↓ Next tracker heartbeat → server returns the command
  ↓ applyCmds() → present action
  ↓ Item materializes center-stage, spotlight applied
  ↓ Everything else gets .dimmed class
  ↓ pushState() streams the new scene
  ↓ board-state.sh → curl GET /state → human-readable board summary
  ↓ AI reads the summary before its next response
```

---

## 18. Latency Budget

| Stage | Typical Latency | Notes |
|-------|:---------------:|-------|
| MediaPipe detection | ~15-30ms | Runs on every new video frame |
| Gesture classification | <1ms | Pure math on landmarks |
| Physics update | <1ms | Per-frame |
| State push (tracker → server) | ~5-15ms | HTTP POST, localhost |
| Command delivery (server → tracker) | ~22ms | One heartbeat cycle |
| AI command (board.sh → board) | ~30-50ms | curl POST + next heartbeat |
| **Total (gesture to visual response)** | **~20-45ms** | ~30fps effective |
| **Total (AI command to board update)** | **~50-70ms** | One heartbeat cycle |
