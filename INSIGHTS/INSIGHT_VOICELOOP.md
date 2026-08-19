# INSIGHT: Voice Loop — Complete Technical Breakdown

> This document explains exactly how `voice_loop_mac.py` processes a user's spoken request from microphone to LLM response and back to audio output. Every package, every function call, every buffer — nothing is abstracted away.

---

## 1. Project Shape

A single Python file (~650 lines) that implements a **fully on-device voice agent** for Mac Apple Silicon. No API keys, no cloud, no server. Everything runs locally using ~3.5 GB of memory.

```
voice_loop_mac.py   ← the entire agent
SOUL.md             ← persona/system prompt (live-reloaded each turn)
MEMORY.md           ← optional persistent facts (--memory flag)
pyproject.toml      ← 12 dependencies, Python ≥3.11
```

---

## 2. The Full Pipeline (One Sentence Summary)

```
Mic → sounddevice → numpy chunks → Silero VAD → Smart Turn → Moonshine STT → Gemma 4 LLM → Kokoro TTS → sounddevice → Speakers
                                                                                                        ↕
                                                                    WebRTC AEC3 + Silero VAD (barge-in)
```

---

## 3. Startup Sequence — What Loads and When

Before the main loop even begins, five models are loaded sequentially:

| Order | What | Package | Time | Memory |
|-------|------|---------|------|--------|
| 1 | Silero VAD (ONNX) | `silero-vad` | ~1s | ~10MB |
| 2 | Moonshine STT | `moonshine-voice` | ~2s | ~250MB |
| 3 | Gemma 4 E4B (4-bit) | `mlx-vlm` | ~15s (first run downloads ~3GB) | ~2.5GB |
| 4 | Smart Turn v3.2 (ONNX) | `onnxruntime` + `transformers` | ~3s (first run downloads ~80MB) | ~80MB |
| 5 | Kokoro TTS v1.0 (ONNX) | `kokoro-onnx` | ~2s (first run downloads ~300MB) | ~300MB |

**Imports that matter:**

```python
from silero_vad import load_silero_vad
from moonshine_voice import Transcriber, get_model_for_language
from mlx_vlm import load, generate, stream_generate
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from kokoro_onnx import Kokoro
```

**Loading code:**

```python
vad = load_silero_vad(onnx=True)

ms_path, ms_arch = get_model_for_language("en")
moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)

model, processor = load("mlx-community/gemma-4-E4B-it-4bit")

# Smart Turn: ONNX model + Whisper feature extractor
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

kokoro = Kokoro(model_file, voices_file)
```

---

## 4. The Main Loop — Mic Capture

### 4.1 sounddevice — Opening the Microphone (Importance: 10/10)

`sounddevice` wraps PortAudio. It opens the mic as a continuous `InputStream` that fires a **callback** every 32 milliseconds, whether anyone is speaking or not. This is the heartbeat of the entire system.

```python
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512   # 32ms at 16kHz — required by Silero VAD

sd.default.latency = 'high'  # larger internal buffer, more robust to CPU saturation

with sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
    blocksize=CHUNK_SAMPLES, callback=callback,
):
```

**Parameters explained:**
- `samplerate=16000` — 16 kHz is the standard for speech models (Moonshine, Silero, Whisper all expect this)
- `channels=1` — mono audio only
- `dtype="float32"` — each sample is a float between -1.0 and 1.0
- `blocksize=512` — the callback receives exactly 512 samples per invocation (512 / 16000 = 0.032 seconds = 32ms)
- `latency='high'` — tells PortAudio to use a larger internal ring buffer so it doesn't drop samples when MLX saturates the CPU during LLM generation

**The callback function** runs in a C-level audio thread (not Python's main thread):

```python
def callback(indata, frames, time, status):
    chunk = indata[:, 0].copy()  # shape: (512,), dtype: float32
    audio_q.put(chunk)           # push to thread-safe queue
```

Key detail: `indata` is a 2D array `(frames, channels)`. We extract channel 0 with `[:, 0]` and `.copy()` because PortAudio reuses the buffer — without `.copy()` the data would be overwritten by the next callback before the main thread reads it.

### 4.2 The Thread-Safe Queue (threading + queue — Importance: 8/10)

The callback runs in a non-Python thread. The main loop runs in the Python main thread. They communicate through `queue.Queue()`:

```python
audio_q: queue.Queue[np.ndarray] = queue.Queue()
```

The main loop pulls chunks one at a time:

```python
while True:
    chunk = audio_q.get()  # blocks until callback delivers a chunk
```

This is the bridge between the audio hardware thread and the Python processing pipeline. The queue is unbounded (no `maxsize`), so if processing is slow, chunks accumulate rather than being dropped.

### 4.3 numpy — The Universal Glue (Importance: 10/10)

Every audio chunk arrives as a `numpy.ndarray` with shape `(512,)` and dtype `float32`. numpy is used for every single data transformation in the system:

- **Accumulation:** chunks are appended to a list, then `np.concatenate(buf)` creates one full utterance
- **Type casting:** `(audio * 32767).astype(np.int16)` converts float32 to int16 for WAV encoding
- **Resampling:** `np.interp(idx, np.arange(len(samples)), samples)` resamples TTS audio from 24kHz to 16kHz for AEC reference
- **Chime synthesis:** sine waves shaped by a Hann-window envelope using `np.sin`, `np.cos`, `np.linspace`
- **Buffer math:** `np.zeros`, `np.concatenate`, `np.pad`, `.clip`, `.reshape`, `.copy`, `.flatten`

numpy is not doing anything clever — it's just the fastest way to move float arrays around in Python.

---

## 5. Speech Detection — Two Gates

Before any audio reaches the STT model, it must pass through **two sequential gates**:

### 5.1 Silero VAD — "Is someone talking?" (Importance: 10/10)

Silero VAD is a tiny neural network that classifies each 32ms chunk as speech or silence. It returns a probability between 0.0 and 1.0.

```python
def _vad_prob(vad, chunk):
    p = vad(torch.from_numpy(chunk), SAMPLE_RATE)
    return p.item() if hasattr(p, "item") else p
```

Notice the `torch.from_numpy(chunk)` — this is the only place `torch` (Importance: 6/10) is used. Silero VAD expects a PyTorch tensor, so we convert the numpy array. This is the entire reason `torch` is a dependency.

**Threshold: 0.5**

```python
speech_prob = _vad_prob(vad, chunk)
if speech_prob > 0.5:
    # accumulate this chunk into the buffer
    buf.append(chunk)
```

**The state machine in the main loop:**

```
IDLE (not speaking):
  → chunk with prob > 0.5 → start accumulating, enter SPEAKING state

SPEAKING:
  → chunk with prob > 0.5 → keep accumulating, reset silence counter
  → chunk with prob ≤ 0.5 → increment silence counter, keep accumulating
  → silence counter reaches limit (default: 700ms = ~22 silent chunks) → check Smart Turn
```

The silence limit is configurable via `--silence-ms` (default 700ms). The math:

```python
silence_limit = int(args.silence_ms / (CHUNK_SAMPLES / SAMPLE_RATE * 1000))
# 700 / (512 / 16000 * 1000) = 700 / 32 = 21.875 → 21 chunks
```

### 5.2 Smart Turn v3 — "Are they actually done?" (Importance: 7/10)

Simple silence detection is too aggressive — people pause mid-sentence. Smart Turn v3 is a classifier that looks at the **entire accumulated buffer** and decides whether the speaker has finished their turn.

```python
def load_smart_turn():
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def predict(audio_float32: np.ndarray) -> float:
        max_samples = 8 * SAMPLE_RATE  # 128000 samples = 8 seconds max
        audio_float32 = audio_float32[-max_samples:]  # truncate to last 8s
        features = extractor(
            audio_float32, sampling_rate=SAMPLE_RATE,
            max_length=max_samples, padding="max_length",
            return_attention_mask=False, return_tensors="np",
        )
        return float(session.run(None, {"input_features": features.input_features.astype(np.float32)})[0].flatten()[0])
    return predict
```

**Two packages work together here:**

1. **transformers** (Importance: 7/10) — provides `WhisperFeatureExtractor`, which converts raw audio into the mel-spectrogram features that the model expects. This is the same feature format used by OpenAI's Whisper model, repurposed here because Smart Turn was trained on Whisper-format features.

2. **onnxruntime** (Importance: 7/10) — runs the actual ONNX neural network inference. `session.run(None, {...})` executes the model and returns the raw output. The `.flatten()[0]` extracts a single scalar probability.

**Threshold: 0.5**

```python
if smart_turn and buf:
    prob = smart_turn(np.concatenate(buf))
    if prob < 0.5:
        silent_chunks = 0  # reset — they're pausing, not done
        continue
# prob >= 0.5 → turn is complete, proceed to STT
```

**Why both gates?** VAD operates per-chunk (32ms) and is fast but noisy. Smart Turn operates on the full utterance (up to 8 seconds) and is slower but much more accurate at distinguishing "pausing to think" from "actually done talking." The two-gate design gives you low-latency detection with high accuracy.

---

## 6. Transcription — Moonshine (Importance: 9/10)

Once the utterance buffer is assembled, it's sent to Moonshine for speech-to-text:

```python
def transcribe(audio_data):
    return " ".join(
        l.text for l in moonshine.transcribe_without_streaming(
            audio_data.tolist(), SAMPLE_RATE
        ).lines if l.text
    ).strip()
```

**Loading:**

```python
ms_path, ms_arch = get_model_for_language("en")
moonshine = Transcriber(model_path=str(ms_path), model_arch=ms_arch)
```

**How it works:**
- `get_model_for_language("en")` returns the path and architecture name for the English model (~250MB, downloaded on first run)
- `Transcriber` loads the model into memory
- `transcribe_without_streaming` takes a Python list of float samples and the sample rate, returns structured output with `.lines`, each containing `.text`
- The `.tolist()` call converts the numpy array to a Python list — Moonshine's API expects a list, not an ndarray
- It runs on **CPU** (not Metal/GPU), which is fast enough for the ~2-5 second utterances it processes

The result is a plain text string like `"What's the weather like today?"` — this becomes the user message for the LLM.

---

## 7. LLM Generation — Gemma 4 via MLX (Importance: 10/10)

This is the brain of the agent. The transcript is formatted into a chat message list and sent to Gemma 4 E4B running on Apple Metal via the MLX framework.

### 7.1 Message Construction

```python
def _sys_messages():
    sp = load_system_prompt(include_memory=args.memory)
    return [{"role": "system", "content": sp}] if sp else []

messages = _sys_messages()
for h in history[-MAX_HISTORY:]:  # last 10 turns
    messages += [
        {"role": "user", "content": h["user"]},
        {"role": "assistant", "content": h["assistant"]},
    ]
messages.append({"role": "user", "content": heard})
```

The system prompt comes from `SOUL.md` (always loaded) and optionally `MEMORY.md`. Both files are **re-read from disk on every turn**, so edits take effect immediately without restarting.

`SOUL.md` contains:
```
You are Voice Loop, an assistant. No lists, no markdown — spoken output.
Use natural contractions (I'll, don't, it's, we're, you're).
```

### 7.2 Token Generation — Two Modes

**Full generation (fallback):**

```python
def llm_generate(messages, max_tokens=200, temperature=0.7, **kwargs):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    r = generate(model, processor, prompt, max_tokens=max_tokens,
                 temperature=temperature, repetition_penalty=1.2, verbose=False, **kwargs)
    return r.text if hasattr(r, "text") else str(r)
```

**Stream generation (preferred — lower latency):**

```python
def stream_sentences(messages, max_tokens=200, temperature=0.7):
    q: queue.Queue[str | None] = queue.Queue()
    cancel = threading.Event()

    def _worker():
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_buf, carry = "", ""
        for result in stream_generate(model, processor, prompt,
                                       max_tokens=max_tokens, temperature=temperature,
                                       repetition_penalty=1.2, verbose=False):
            if cancel.is_set():
                return
            token_buf += result.text
            # Check for sentence-ending punctuation
            while True:
                m = _SENT_END.search(token_buf)  # regex: r'(?<=[.!?])\s+'
                if not m:
                    break
                carry = _merge(carry, token_buf[:m.start() + 1].strip())
                token_buf = token_buf[m.end():]
                if len(carry) >= _SENT_MIN_CHARS:  # 20 chars minimum
                    q.put(carry)
                    carry = ""
        # flush remainder
        if remainder:
            q.put(remainder)
        q.put(None)  # sentinel

    threading.Thread(target=_worker, daemon=True).start()
    # yield sentences as they arrive
    while True:
        s = q.get()
        if s is None:
            return
        yield s
```

**Why this matters for latency:** The LLM might take 3-5 seconds to generate a full response. But the first sentence might be ready in 0.5 seconds. By streaming tokens in a background thread and dispatching complete sentences to TTS immediately, the user hears the response start almost instantly instead of waiting for the entire response to finish generating.

**Sentence splitting logic:**
- Regex `_SENT_END = re.compile(r'(?<=[.!?])\s+')` matches sentence-ending punctuation followed by whitespace
- Short fragments (< 20 chars) are merged with the next sentence to avoid TTS artifacts on abbreviations like "Mr." or "Dr."
- The `cancel` event allows barge-in to stop the LLM mid-generation, saving compute

**mlx-vlm methods used:**
- `load(model_name)` → returns `(model, processor)` tuple
- `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` → formats the message list into the prompt string the model expects
- `generate(model, processor, prompt, max_tokens, temperature, repetition_penalty)` → full response
- `stream_generate(...)` → yields token results one at a time (same params as `generate`)

---

## 8. TTS Synthesis — Kokoro (Importance: 9/10)

Kokoro converts text to speech audio using an ONNX model running on CPU.

```python
from kokoro_onnx import Kokoro
kokoro = Kokoro(model_file, voices_file)

samples, sr = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
```

**Key details:**
- Output is a `numpy.ndarray` of float32 samples at 24 kHz (not 16 kHz like the mic)
- Voices are identified by name: `af_heart` (US female, default), `bf_emma` (UK female), `am_fenrir` (US male), etc.
- Language is inferred from the voice prefix: `a*` → en-us, `b*` → en-gb, `e*` → es, etc.
- Kokoro requires `espeak-ng` as a system dependency for phoneme generation (installed via `brew install espeak-ng`)

### 8.1 Streaming TTS — Synthesis Overlaps Playback

Sentences are synthesised in **pairs** (for natural prosody across sentence boundaries) in a background asyncio task, while the previous pair is playing:

```python
async def _synthesizer():
    GROUP = 2
    buf: list[str] = []
    for sentence in sentence_iter:
        buf.append(sentence)
        if len(buf) == GROUP:
            await synth_q.put(await _synth(" ".join(buf)))
            buf = []
    if buf:
        await synth_q.put(await _synth(" ".join(buf)))
    await synth_q.put(None)  # sentinel

synth_task = asyncio.create_task(_synthesizer())
```

The `_synth` function runs `kokoro.create()` in a thread executor so it doesn't block the asyncio event loop:

```python
async def _synth(text):
    return await loop.run_in_executor(
        None,
        lambda t=text: kokoro.create(t, voice=args.voice, speed=1.0, lang=...)
    )
```

The synthesis queue has `maxsize=1` — the synthesiser stays exactly one sentence-pair ahead of the player. This keeps memory bounded while ensuring there's always audio ready when the current sentence finishes playing.

---

## 9. Audio Output — sounddevice Again (Importance: 10/10)

TTS audio plays through `sd.OutputStream`, not `sd.play`. The output stream writes audio in small frames so barge-in checks can interleave:

```python
out_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
out_stream.start()

data = samples.reshape(-1, 1)  # shape: (N, 1) for mono output
for i in range(0, len(data), 4096):
    # Check for interrupt between every frame
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)
        interrupted = True
    elif check_barge_in():
        interrupted = True
    if interrupted:
        break
    out_stream.write(data[i:i+4096])
```

**Why 4096 samples?** At 24 kHz, 4096 samples ≈ 170ms. This means barge-in is checked roughly 6 times per second — responsive enough to feel instant when the user speaks over the agent.

**Why `select.select`?** The terminal is set to raw mode (`tty.setcbreak`), which means keypresses are available immediately without pressing Enter. `select.select` checks stdin non-blockingly — if a key was pressed, the user wants to interrupt.

---

## 10. Barge-In — WebRTC AEC3 (Importance: 8/10)

This is the most technically complex part of the system. When the agent is speaking through the speakers, the microphone picks up its own voice. Without echo cancellation, the VAD would detect the agent's own speech as the user talking and trigger a false interrupt.

### 10.1 The AEC Pipeline

LiveKit's APM (Audio Processing Module) wraps WebRTC's AEC3 algorithm:

```python
from livekit.rtc import AudioFrame
from livekit.rtc.apm import AudioProcessingModule

apm = AudioProcessingModule(echo_cancellation=True, noise_suppression=True)
```

For each mic chunk received during TTS playback:

```python
def process(mic, ref):
    cleaned = np.zeros_like(mic)
    for i in range(0, len(mic), WF):  # WF = 160 samples = 10ms
        mic_f = _frame(_to_i16(mic[i:i+WF]))
        apm.process_reverse_stream(_frame(_to_i16(ref[i:i+WF])))  # feed TTS reference
        apm.process_stream(mic_f)                                   # process mic
        cleaned[i:i+WF] = (np.frombuffer(bytes(mic_f.data), dtype=np.int16)
                           .astype(np.float32) / 32767)[:len(mic[i:i+WF])]
    return cleaned
```

**How it works:**
1. The TTS audio being played through the speakers is the **reference signal**
2. The mic audio is the **signal to clean**
3. AEC3 subtracts the reference from the mic signal, removing the agent's own voice
4. What remains should be only the user's voice (if they're speaking)
5. This cleaned audio is then fed to Silero VAD

### 10.2 Reference Buffer Alignment

The TTS audio is at 24 kHz (Kokoro's output rate) but the mic is at 16 kHz. The reference signal must be resampled and aligned:

```python
def _append_ref(chunk_samples, sr):
    if sr == SAMPLE_RATE:  # 16kHz
        tts_16k_buf.append(chunk_samples.astype(np.float32))
    else:  # 24kHz → resample to 16kHz
        idx = np.arange(0, len(chunk_samples), sr / SAMPLE_RATE)
        tts_16k_buf.append(
            np.interp(idx, np.arange(len(chunk_samples)), chunk_samples).astype(np.float32)
        )
```

A `mic_pos` counter tracks how far through the reference buffer the mic has consumed, keeping the two signals temporally aligned.

### 10.3 Barge-In Detection

```python
def check_barge_in():
    if not (aec_process and state["play_start"] and
            _time.monotonic() - state["play_start"] >= 0.5):
        return False  # 500ms inhibit window after each sentence starts
    tts_concat = _get_tts_concat()
    while not audio_q.empty():
        mic_chunk = audio_q.get_nowait()
        ref = _get_ref_segment(tts_concat, state["mic_pos"], len(mic_chunk))
        state["mic_pos"] += len(mic_chunk)
        cleaned = aec_process(mic_chunk, ref)
        if _vad_prob(vad, cleaned.astype(np.float32)) > 0.8:  # higher threshold!
            state["consec_speech"] += 1
            if state["consec_speech"] >= 5:  # 5 consecutive speech chunks
                return True
        else:
            state["consec_speech"] = 0
    return False
```

**Three safety mechanisms prevent false interrupts:**
1. **0.5s inhibit window** — barge-in is disabled for the first 500ms after each sentence starts playing, preventing the initial speaker transient from triggering
2. **Higher VAD threshold** — 0.8 instead of 0.5, requiring stronger evidence of speech
3. **5 consecutive chunks** — speech must be detected in 5 consecutive 32ms chunks (160ms total) before interrupt triggers

### 10.4 Inter-Sentence Gap Handling

Between sentences, there's a brief gap where no TTS audio is playing. The room still has reverb from the previous sentence:

```python
_GAP_BLANK_SAMPLES = int(0.15 * 16000)  # 150ms @ 16kHz
```

During the first 150ms after a sentence ends, mic chunks are consumed and `mic_pos` advances (silence is appended to the reference buffer for alignment), but **AEC is not called**. Feeding a zero reference during reverb decay would cause AEC3 to pass residual echo through as speech. After 150ms, the reverb has decayed enough that normal AEC can resume safely.

---

## 11. Memory System (Optional — `--memory` flag)

Two LLM-powered memory operations run after each turn:

**Fact extraction** (after every turn):
```python
def update_memory(heard, response):
    result = llm_generate(
        [{"role": "user", "content": f"Current memory:\n{_read_memory()}\n\n"
          f"User said: {heard}\n\n"
          "Did the user state a new durable fact about themselves? ..."}],
        max_tokens=60, temperature=0.2
    )
    # Append new facts to MEMORY.md
```

**Consolidation** (every 5 turns):
```python
def consolidate_memory():
    result = llm_generate(
        [{"role": "user", "content": f"Here is a memory file:\n\n{_read_memory()}\n\n"
          "Rewrite it: merge duplicates, remove transient items..."}],
        max_tokens=300, temperature=0.2
    )
    # Overwrite MEMORY.md with cleaned version
```

Both use the same LLM (Gemma 4) with low temperature (0.2) for deterministic output. The memory file is re-read from disk every turn, so external edits are respected.

---

## 12. Threading Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ AUDIO THREAD (C, non-Python)                                       │
│   sounddevice callback → audio_q.put(chunk)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ queue.Queue (thread-safe)
┌────────────────────────────▼────────────────────────────────────────┐
│ MAIN THREAD                                                        │
│   audio_q.get() → VAD → Smart Turn → process_utterance()          │
│   → asyncio.run(_play())  [blocks until TTS finishes or interrupt] │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│ LLM WORKER THREAD (threading.Thread, daemon=True)                  │
│   stream_generate() → sentence queue → TTS synthesiser task       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ queue.Queue
┌────────────────────────────▼────────────────────────────────────────┐
│ ASYNCIO EVENT LOOP (inside _play())                                │
│   _synthesizer task: kokoro.create() via run_in_executor           │
│   _play loop: OutputStream.write() + barge-in checks              │
└─────────────────────────────────────────────────────────────────────┘
```

**Four concurrency primitives in play:**
1. `queue.Queue` — between audio thread and main thread (mic chunks)
2. `threading.Thread` — LLM worker generates tokens in background
3. `queue.Queue` — between LLM worker and asyncio loop (sentences)
4. `asyncio` + `ThreadPoolExecutor` — TTS synthesis overlaps with playback

---

## 13. Terminal Handling

```python
import termios, tty, select

old_term = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())    # raw mode: no Enter needed
# ... main loop ...
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)  # restore on exit
```

This puts the terminal into **cbreak mode** so any keypress is immediately available via `sys.stdin.read(1)` without waiting for Enter. During TTS playback, `select.select([sys.stdin], [], [], 0)` polls stdin non-blockingly between every 4096-sample write to detect keypress interrupts.

---

## 14. Complete Dependency Map

### Third-Party (by importance)

| Importance | Package | Role | Key Methods |
|:----------:|---------|------|-------------|
| **10** | `sounddevice` | Audio I/O — the only way sound enters/leaves the system | `InputStream`, `OutputStream`, `play`, `stop`, `wait`, `default.latency` |
| **10** | `numpy` | All audio buffer manipulation | `zeros`, `concatenate`, `interp`, `sin`, `cos`, `linspace`, `frombuffer`, `pad`, `arange`, `.astype`, `.clip`, `.reshape`, `.copy`, `.flatten` |
| **10** | `silero-vad` | Speech detection — gates the entire pipeline | `load_silero_vad(onnx=True)`, `vad(chunk, sr)`, `vad.reset_states()` |
| **10** | `mlx-vlm` | LLM inference on Apple Metal | `load`, `generate`, `stream_generate`, `processor.apply_chat_template` |
| **9** | `moonshine-voice` | Speech-to-text transcription on CPU | `get_model_for_language`, `Transcriber`, `transcribe_without_streaming` |
| **9** | `kokoro-onnx` | Text-to-speech synthesis on CPU | `Kokoro(model_file, voices_file)`, `kokoro.create(text, voice, speed, lang)` |
| **8** | `livekit` | WebRTC AEC3 for barge-in | `AudioProcessingModule`, `apm.process_reverse_stream`, `apm.process_stream`, `AudioFrame` |
| **7** | `onnxruntime` | Runs Smart Turn v3 ONNX model | `InferenceSession`, `session.run` |
| **7** | `transformers` | Whisper feature extraction for Smart Turn | `WhisperFeatureExtractor.from_pretrained`, `extractor(audio, ...)` |
| **6** | `torch` | Tensor bridge for Silero VAD only | `torch.from_numpy(chunk)` |

### Standard Library (by importance)

| Importance | Module | Role |
|:----------:|--------|------|
| **9** | `threading` | LLM background worker, daemon threads |
| **8** | `asyncio` | TTS playback pipeline with concurrent synthesis |
| **8** | `queue` | Thread-safe FIFO for mic chunks and sentences |
| **7** | `argparse` | All CLI flags (`--tts`, `--aec`, `--smart-turn`, `--memory`, `--voice`, `--model`) |
| **7** | `termios` + `tty` | Raw terminal mode for instant keypress interrupt |
| **6** | `select` | Non-blocking stdin poll during TTS playback |
| **5** | `wave` | WAV file writing for debug recording |
| **5** | `re` | Sentence-splitting regex for TTS dispatch |
| **5** | `concurrent.futures` | ThreadPoolExecutor for parallel STT in audio mode |
| **4** | `tempfile` | Model cache directories, temp WAV files |
| **4** | `os` + `pathlib` | File paths, env vars, directory management |
| **4** | `time` | Timing for chime alignment, sleeps |
| **3** | `subprocess` | Auto-detect espeak-ng library path |
| **2** | `urllib.request` | One-time model downloads |

---

## 15. Configuration Constants

```python
SAMPLE_RATE = 16000           # 16 kHz for all speech models
CHUNK_SAMPLES = 512           # 32ms blocks (Silero VAD requirement)
MAX_HISTORY = 10              # conversation turns kept in context
CHIME_SR = 24000              # chime audio sample rate
_SENT_MIN_CHARS = 20          # minimum sentence length for TTS dispatch
_GAP_BLANK_SAMPLES = 2400     # 150ms reverb blanking window @ 16kHz
```

---

## 16. The Data Types Flowing Through the System

```
Microphone (hardware)
  ↓ PortAudio
float32 ndarray, shape (512,), range [-1.0, 1.0]
  ↓ numpy
float32 ndarray, shape (N,), accumulated utterance
  ↓ .tolist()
Python list of floats
  ↓ Moonshine
str (transcript text)
  ↓ apply_chat_template
str (formatted prompt)
  ↓ mlx-vlm tokeniser
int tensor (token IDs)
  ↓ mlx-vlm model
str (generated text, streamed token-by-token)
  ↓ sentence split
list[str] (individual sentences)
  ↓ Kokoro
float32 ndarray, shape (M,), 24 kHz audio samples
  ↓ .reshape(-1, 1)
float32 ndarray, shape (M, 1), for OutputStream
  ↓ PortAudio
Speakers (hardware)
```

---

## 17. Latency Budget — Where Time Goes

| Stage | Typical Latency | Notes |
|-------|:---------------:|-------|
| Mic callback → main thread | <1ms | queue.Queue overhead |
| VAD per chunk | ~2ms | Tiny ONNX model on CPU |
| Smart Turn per utterance | ~50-200ms | Depends on utterance length (up to 8s audio) |
| Moonshine STT | ~500-2000ms | CPU, depends on utterance length |
| LLM first token | ~200-800ms | MLX on Metal, depends on prompt length |
| LLM first sentence | ~500-2000ms | Depends on sentence length |
| Kokoro TTS synthesis | ~300-1000ms | CPU, per sentence-pair |
| **Total (mic to first sound)** | **~2-4 seconds** | Dominated by STT + LLM first sentence |

The streaming architecture (LLM → sentence queue → TTS) overlaps the last three stages, saving 1-3 seconds compared to waiting for the full response.

---

## 18. CLI Flags Reference

```bash
uv run voice_loop_mac.py                    # all features on (default)
uv run voice_loop_mac.py --no-tts           # text output only, no Kokoro
uv run voice_loop_mac.py --no-aec           # no echo cancellation, keypress interrupt only
uv run voice_loop_mac.py --no-smart-turn    # simple silence detection, no turn classifier
uv run voice_loop_mac.py --chime            # play chime on utterance + ticks while generating
uv run voice_loop_mac.py --memory           # enable MEMORY.md read/write
uv run voice_loop_mac.py --voice bf_emma    # UK female voice
uv run voice_loop_mac.py --silence-ms 500   # shorter silence timeout
uv run voice_loop_mac.py --record           # save mic audio to WAV for debugging
uv run voice_loop_mac.py --audio-mode       # send audio directly to Gemma (experimental)
uv run voice_loop_mac.py --model mlx-community/gemma-4-E2B-it-4bit  # smaller model
```

---

> ### 📝 Editor's Note — Microphone Selection
>
> The original `voice_loop_mac.py` opens the **system default** microphone with no device selection. If you're using a specific mic (e.g. a **UGREEN camera/mic**), you need to select it explicitly — otherwise PortAudio may grab the wrong device.
>
> **Finding your device:**
>
> ```python
> import sounddevice as sd
> print(sd.query_devices())
> ```
>
> Look for your mic in the output and note its index number. On our Linux test system the UGREEN appeared as:
>
> ```
>  [ 8] UGREEN Camera: USB Audio (hw:3,0)  (in=2)  ← ALSA, preferred
>  [18] UGREEN Camera Analog Stereo         (in=2)  ← JACK/PipeWire, fallback
> ```
>
> **Selecting it explicitly:**
>
> ```python
> stream = sd.InputStream(
>     samplerate=16000, channels=1, dtype="float32",
>     blocksize=512, callback=callback,
>     device=8,  # ← pass the device index
> )
> ```
>
> **Sample rate gotcha:**
>
> USB microphones don't support every sample rate. If you get:
>
> ```
> sounddevice.PortAudioError: Error opening InputStream: Invalid sample rate [PaErrorCode -9997]
> ```
>
> ...it means either the device doesn't support that rate, or another process is still holding the device. To check which rates your mic supports:
>
> ```python
> for rate in [8000, 16000, 22050, 32000, 44100, 48000, 96000]:
>     try:
>         s = sd.InputStream(device=8, samplerate=rate, channels=1,
>                            dtype="float32", blocksize=512)
>         s.start(); s.stop(); s.close()
>         print(f"  {rate:>6d} Hz  ✓")
>     except:
>         print(f"  {rate:>6d} Hz  ✗")
> ```
>
> Our UGREEN supported **8000, 16000, 22050, 44100, 48000** but not 32000 or 96000.
>
> **Fallback pattern** (try preferred rate, then alternatives):
>
> ```python
> actual_rate = 0
> for try_rate in [16000, 48000, 44100]:
>     try:
>         stream = sd.InputStream(samplerate=try_rate, ...)
>         stream.start()
>         actual_rate = try_rate
>         break
>     except Exception as e:
>         print(f"  {try_rate}Hz failed: {e}")
> ```
>
> If the mic opens at a different rate than 16 kHz, **resample** inside the callback so everything downstream stays at 16 kHz:
>
> ```python
> if actual_rate != 16000:
>     ratio = 16000 / actual_rate
>     new_len = int(len(chunk) * ratio)
>     idx = np.linspace(0, len(chunk) - 1, new_len)
>     chunk = np.interp(idx, np.arange(len(chunk)), chunk).astype(np.float32)
> ```
>
> **Port cleanup:** If you get `Address already in use` or the mic won't open after a crash, kill stale processes:
>
> ```bash
> fuser -k -9 8765/tcp   # if running a WebSocket showcase server
> pkill -f "voice_loop"    # kill any lingering voice-loop processes
> ```
