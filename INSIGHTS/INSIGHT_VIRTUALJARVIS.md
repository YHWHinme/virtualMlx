# INSIGHT: VirtualJarvis — Complete Technical Breakdown

> This document explains exactly how `VirtualJarvis` processes a user's spoken request from microphone to LLM response and back to audio output. Every package, every function call, every thread — nothing is abstracted away.

---

## 1. Project Shape

A multi-file Python project (~500 lines across 8 source files) that implements a **voice assistant** using cloud STT (Google Speech Recognition) or a remote Ollama server for LLM, and Kokoro for TTS. It has two execution paths: a simple blocking mode and a threaded streaming mode.

```
main.py              ← Jarvis class — the orchestrator loop
stt.py               ← Speech-to-Text via Google SpeechRecognition
tts.py               ← Text-to-Speech via Kokoro (KPipeline)
gemini_model.py       ← Google Gemini API (alternative LLM, cloud-based)
ollama_model.py       ← Ollama API (local/remote LLM server)
formatter.py          ← LLM-powered file formatting utility (standalone tool)
whisper.py            ← OpenAI Whisper STT (alternative, file-based)
test_pyaudio.py       ← Streaming TTS test harness with PyAudio + Gemini + Kokoro
pyproject.toml        ← 9 dependencies, Python ≥3.10
requirements.txt      ← Flat dependency list
```

---

## 2. The Full Pipeline (One Sentence Summary)

```
Mic → SpeechRecognition → Google API → text → Ollama/Gemini → response text → Kokoro TTS → ffplay/PyAudio → Speakers
```

There are **two distinct code paths** in this project:

### Path A: `main.py` (the production path)
```
Mic (SpeechRecognition + Google API) → Ollama server → Kokoro TTS → ffplay CLI → Speakers
```

### Path B: `test_pyaudio.py` (the streaming test harness)
```
Hardcoded text → Gemini cloud API (streaming) → Kokoro TTS → PyAudio stream → Speakers
```

Path A is the main entry point. Path B is a test file that demonstrates the streaming architecture the author is moving toward.

---

## 3. Startup Sequence — What Loads and When

Unlike Voice Loop (which loads 5 ML models at startup), VirtualJarvis is lightweight — most work happens over HTTP to remote services.

### main.py startup

| Order | What | Package | Time | Memory |
|-------|------|---------|------|--------|
| 1 | Ollama client | `ollama` | ~instant | ~10MB |
| 2 | SpeechRecognition recognizer | `speechrecognition` | ~instant | ~5MB |
| 3 | Kokoro KPipeline | `kokoro` | ~2s | ~300MB |
| 4 | IPython Audio display | `IPython` | ~instant | ~20MB |

**Imports in main.py:**

```python
import os
from ollama_model import ollama_chat, ollama_chat_stream
from stt import recordAudio
from tts import speakText, speakText_chunk
```

### test_pyaudio.py startup

| Order | What | Package | Time | Memory |
|-------|------|---------|------|--------|
| 1 | Kokoro KPipeline | `kokoro` | ~2s | ~300MB |
| 2 | Gemini client | `google-genai` | ~instant | ~5MB |
| 3 | PyAudio | `pyaudio` | ~instant | ~2MB |

**Imports in test_pyaudio.py:**

```python
import pyaudio
import numpy as np
import threading
import queue
from gemini_model import gemini_chat_stream
from kokoro import KPipeline
import torch
```

### import.py (reference/legacy code — not used by main.py)

This file contains a completely different architecture (MLX-based, fully local) that appears to be a more advanced version or earlier experiment:

```python
from stt.VoiceActivityDetection import VADDetector
from mlx_lm import load, generate
from melo.api import TTS
import librosa
import sounddevice as sd
from playsound import playsound
```

This file uses MLX models, MeloTTS, librosa, and sounddevice — a much heavier stack that's not connected to the main execution path.

---

## 4. The Main Loop — `Jarvis` Class in `main.py`

### 4.1 Class Structure

```python
class Jarvis:
    def __init__(self):
        self.sttData = ""
        self.listening = False
        self.stt = self.transcribeLoop()
        self.Greet()
```

The constructor immediately enters the listen-think-speak loop. There is no separate initialization phase.

### 4.2 The Core Loop — `Main()`

```python
def Main(self):
    streaming_mode = os.getenv("STREAMING_MODE", "false").lower() == "true"
    while True:
        self.transcribeLoop()
        if streaming_mode:
            chunk_gen = self.remoteThink_ollama_stream("gemma3:4b-it-qat")
            self.modelAnswer_stream(chunk_gen)
        else:
            response = self.think()
            self.modelAnswer(response)
        self.listening = True
```

This is an infinite loop: **listen → think → speak → repeat**. The `streaming_mode` flag (set via the `STREAMING_MODE` environment variable in `mise.toml`) determines which path is taken.

### 4.3 The Listening Toggle

```python
def toggleListening(self):
    self.listening = not self.listening
```

This is a simple boolean flip. After STT captures valid speech, `self.listening` is set to `False` (stop listening). After TTS finishes, it's set back to `True`. There is no VAD, no silence detection, no turn classifier — just a toggle.

---

## 5. Speech-to-Text — Google SpeechRecognition (Importance: 10/10)

This is the most critical package — it's the only way user input enters the system.

### 5.1 `stt.py` — The STT Module

```python
import speech_recognition as sr

def recordAudio():
    recorder = sr.Recognizer()
    try:
        with sr.Microphone() as source2:
            recorder.adjust_for_ambient_noise(source2, duration=0.2)
            print("\n\nListening...")
            audio2 = recorder.listen(source2)
            transcribed_text = recorder.recognize_google(audio2)
            print("Processing...")
            return transcribed_text
    except sr.RequestError as e:
        print(f"Request Error {e}")
        return "An error occured"
    except sr.UnknownValueError:
        print("Unknown error occured")
```

**How it works step by step:**

1. `sr.Recognizer()` — creates a recognizer instance
2. `sr.Microphone()` — opens the default system microphone (via PyAudio underneath)
3. `adjust_for_ambient_noise(source2, duration=0.2)` — listens for 200ms to calibrate the noise floor
4. `recorder.listen(source2)` — **blocks** until speech is detected and then silence follows. This is SpeechRecognition's built-in energy-based VAD — not a neural network, just a simple energy threshold
5. `recognize_google(audio2)` — sends the audio to **Google's Speech Recognition API** over HTTP and returns the transcribed text. **This requires internet access.**
6. Returns a `str` or error string

**Key characteristics:**
- **Blocking** — `listen()` blocks the main thread until it detects a complete utterance
- **Cloud-dependent** — `recognize_google()` requires internet; there's no offline fallback in this path
- **No streaming** — the entire utterance is captured and sent as one request
- **No explicit sample rate** — SpeechRecognition handles audio capture internally (typically 16kHz)

### 5.2 The transcribeLoop Wrapper

```python
def transcribeLoop(self):
    while self.listening:
        recorded_data = recordAudio()
        if recorded_data and recorded_data not in ["An error occured", None]:
            self.sttData = recorded_data
            self.toggleListening()
            break
        else:
            print("No valid data, trying again")
```

This loops calling `recordAudio()` until it gets a valid transcript. On success, it stores the text in `self.sttData` and flips listening off. On failure (ambient noise, silence, API error), it retries.

### 5.3 `whisper.py` — Alternative STT (not used by main.py)

```python
import whisper

def whisper_stt(input_file):
    model = whisper.load_model("turbo")
    result = model.transcribe(input_file)
    return result["text"]
```

This is a file-based Whisper implementation — loads the `turbo` model and transcribes an audio file. It's a standalone utility, not integrated into the main loop. No streaming, no mic input — just batch file transcription.

---

## 6. LLM — Two Options (Importance: 9/10 each)

### 6.1 Ollama — Local/Remote LLM Server (Primary — used by main.py)

**`ollama_model.py`:**

```python
from ollama import ChatResponse, Client

client = Client(host="http://172.16.5.39:3000")

def ollama_chat(content, client=client):
    response: ChatResponse = client.chat(
        model='gemma3:4b',
        messages=[{'role': 'user', 'content': content}],
    )
    return response

def ollama_chat_stream(content, client=client):
    response = client.chat(
        model='gemma3:4b',
        messages=[{'role': 'user', 'content': content}],
        stream=True,
    )
    for chunk in response:
        if chunk['message']['content']:
            yield chunk['message']['content']
```

**How it works:**
- Connects to a **remote Ollama server** at `http://172.16.5.39:3000` (a LAN address, not localhost)
- Uses `gemma3:4b` — a 4-billion parameter model running on that remote server
- `ollama_chat()` — blocking: sends the full message, waits for the full response
- `ollama_chat_stream()` — streaming: yields content chunks as they arrive from the server
- The `ollama` package is just an HTTP client — the actual model runs on the remote machine

**Important detail:** The client is recreated inside `ollama_chat` (`client = Client(host=...)`) despite being passed as a default argument. This means every call creates a new connection.

### 6.2 Gemini — Google Cloud LLM (Alternative — used by test_pyaudio.py)

**`gemini_model.py`:**

```python
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

def gemini_chat(input, model="gemini-2.5-flash"):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=input)
    return response

def gemini_chat_stream(input, model="gemini-2.5-flash"):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content_stream(model=model, contents=input)
    for chunk in response:
        if chunk.text:
            yield chunk.text
```

**How it works:**
- Uses **Google's Gemini API** (cloud service) via the `google-genai` SDK
- Requires `GEMINI_API_KEY` environment variable
- Default model: `gemini-2.5-flash`
- Same two modes: `gemini_chat` (blocking) and `gemini_chat_stream` (streaming)
- A new `genai.Client` is created on every call — no connection reuse

### 6.3 How main.py Chooses

```python
# In main.py:
from ollama_model import ollama_chat, ollama_chat_stream
```

main.py **only imports Ollama** — Gemini is not used in the production path. However, `gemini_model.py` is imported by `test_pyaudio.py` for the streaming test harness.

### 6.4 The Think Functions

```python
def think(self):
    model_input = self.sttData
    response = ollama_chat(model_input)["message"]["content"] or "Sorry, I couldn't generate a response."
    return response

def remoteThink_ollama_stream(self, model: str):
    model_input = self.sttData
    for chunk in ollama_chat_stream(model_input):
        yield chunk
```

`think()` is blocking — returns the full response string. `remoteThink_ollama_stream()` is a generator that yields text chunks. Note: the `model` parameter is accepted but never used — the model is hardcoded in `ollama_model.py`.

---

## 7. TTS — Kokoro (Importance: 9/10)

### 7.1 `tts.py` — The TTS Module

Kokoro's `KPipeline` is used for speech synthesis. The pipeline combines text-to-phoneme conversion and phoneme-to-audio generation.

```python
from kokoro import KPipeline
import torch
import soundfile as sf
from IPython.display import display, Audio
import os, random

pipeline = KPipeline(lang_code='a')  # 'a' = US English

def speakText(input_text):
    random_number = random.randint(1, 10)
    all_audio = []

    generator = pipeline(input_text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None and not isinstance(audio, str):
            all_audio.extend(list(audio))

    all_audio = torch.tensor(all_audio)
    display(Audio(data=all_audio, rate=24000, autoplay=True))

    filename = f'combined_{random_number}.wav'
    sf.write(filename, all_audio, 24000)

    try:
        os.system(f"ffplay {filename} -autoexit -nodisp")
    except KeyboardInterrupt:
        print("Process interrupted")

    try:
        os.remove(filename)
    except OSError as e:
        print(f"Error deleting file: {e}")
```

**How it works step by step:**

1. `KPipeline(lang_code='a')` — initializes the Kokoro pipeline for US English at module level (loaded once)
2. `pipeline(input_text, voice='af_heart')` — returns a generator that yields `(graphemes, phonemes, audio)` tuples. The audio is a numpy array of float32 samples at 24kHz
3. All audio chunks are accumulated into `all_audio` list
4. `torch.tensor(all_audio)` — converts the list to a PyTorch tensor (required by IPython's Audio display)
5. `display(Audio(...))` — plays audio inline if running in IPython/Jupyter (no-op in terminal)
6. `sf.write(filename, all_audio, 24000)` — writes to a WAV file using soundfile
7. `os.system(f"ffplay ...")` — **plays the audio via ffplay** (a command-line audio player from FFmpeg). The `-autoexit` flag makes it exit when playback finishes, `-nodisp` suppresses the video window
8. `os.remove(filename)` — deletes the WAV file after playback

**Key characteristics:**
- **File-based playback** — audio is written to disk, then played via an external process (`ffplay`)
- **Blocking** — `os.system()` blocks until ffplay exits
- **No streaming between TTS and playback** — the entire response is synthesised before playback starts (in non-streaming mode)
- **24 kHz output** — Kokoro's native sample rate
- **Voice: `af_heart`** — US English female, hardcoded

### 7.2 `speakText_chunk()` — Per-Sentence TTS

```python
def speakText_chunk(input_text):
    if not input_text.strip():
        return

    all_audio = []
    generator = pipeline(input_text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None and not isinstance(audio, str):
            all_audio.extend(list(audio))

    if not all_audio:
        return

    all_audio = torch.tensor(all_audio)
    display(Audio(data=all_audio, rate=24000, autoplay=True))

    filename = f'chunk_{random_number}.wav'
    sf.write(filename, all_audio, 24000)

    try:
        os.system(f"ffplay {filename} -autoexit -nodisp")
    except KeyboardInterrupt:
        print("Process interrupted")

    try:
        os.remove(filename)
    except OSError as e:
        print(f"Error deleting file: {e}")
```

Same logic as `speakText()` but designed to be called per-sentence. In the streaming path, each completed sentence triggers one `speakText_chunk()` call — synthesize → write WAV → ffplay → delete.

---

## 8. Streaming Mode — Sentence-Level Overlap

### 8.1 Sentence Boundary Detection

```python
def is_sentence_end(text):
    if not text.strip():
        return False
    return text.strip()[-1] in ".!?"
```

Simple heuristic: if the last character of accumulated text is `.`, `!`, or `?`, the sentence is complete.

### 8.2 Streaming Answer Pipeline (in main.py)

```python
def modelAnswer_stream(self, chunk_generator):
    buffer = ""
    for chunk in chunk_generator:
        buffer += chunk
        if is_sentence_end(buffer):
            speakText_chunk(buffer)
            buffer = ""
    if buffer:
        speakText_chunk(buffer)
```

**Flow:**
1. LLM yields text chunks (fractions of a sentence)
2. Chunks are accumulated in `buffer`
3. When a sentence boundary is detected, the full sentence is sent to `speakText_chunk()`
4. `speakText_chunk()` blocks (synthesize → write → ffplay → delete)
5. After playback, the loop continues collecting the next sentence

**Important:** This is **sequential**, not truly streaming. Each sentence must finish playing before the next one starts synthesizing. The "streaming" benefit is only that the LLM generates the next sentence while the current one plays — but actually, since `speakText_chunk()` blocks the main thread and the LLM chunks are yielded by a generator on the same thread, there's no overlap here. The LLM waits for TTS to finish.

### 8.3 True Streaming in test_pyaudio.py

`test_pyaudio.py` implements a **proper three-stage producer-consumer pipeline** with real thread overlap:

```python
def modelAnswer_stream(self, chunk_generator):
    text_queue = queue.Queue()
    audio_queue = queue.Queue()

    # Thread 1: AI producer — feeds sentences to text queue
    def ai_producer():
        buffer = ""
        for chunk in chunk_generator:
            buffer += chunk
            if is_sentence_end(buffer):
                text_queue.put(buffer.strip())
                buffer = ""
        if buffer:
            text_queue.put(buffer.strip())
        text_queue.put(None)  # sentinel

    # Thread 2: TTS consumer — synthesizes text → audio
    def tts_consumer():
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:
                audio_queue.put(None)
                break
            audio_data = generate_audio_from_text_standalone(text_chunk)
            audio_queue.put(audio_data)

    # Thread 3: Playback consumer — streams audio to speakers
    def playback_consumer():
        py_audio = pyaudio.PyAudio()
        stream = py_audio.open(format=pyaudio.paFloat32, channels=1,
                               rate=24000, output=True, frames_per_buffer=1024)
        while True:
            audio_data = audio_queue.get()
            if audio_data is None:
                break
            for i in range(0, len(audio_data), 1024):
                chunk = audio_data[i:i+1024]
                stream.write(chunk.tobytes())
        stream.stop_stream()
        stream.close()
        py_audio.terminate()

    producer_thread = threading.Thread(target=ai_producer)
    tts_thread = threading.Thread(target=tts_consumer)
    playback_thread = threading.Thread(target=playback_consumer)

    producer_thread.start()
    tts_thread.start()
    playback_thread.start()

    producer_thread.join()
    tts_thread.join()
    playback_thread.join()
```

**This is the real streaming architecture:**

```
Thread 1 (AI producer)    → text_queue →
Thread 2 (TTS consumer)   → audio_queue →
Thread 3 (Playback consumer) → PyAudio stream → Speakers
```

- **PyAudio** replaces ffplay — audio is written directly to a hardware stream in 1024-sample frames (≈42ms at 24kHz)
- All three threads run concurrently — while sentence 2 is being synthesized, sentence 1 is still playing
- `queue.Queue()` with `None` sentinels provides clean shutdown

**The standalone TTS function used by test_pyaudio.py:**

```python
kokoro_pipeline = KPipeline(lang_code='a')

def generate_audio_from_text_standalone(text_chunk):
    all_audio = []
    generator = kokoro_pipeline(text_chunk, voice='af_heart')
    for _, _, audio in generator:
        if audio is not None and not isinstance(audio, str):
            all_audio.extend(list(audio))
    if all_audio:
        return torch.tensor(all_audio).numpy().astype(np.float32)
    return np.array([], dtype=np.float32)
```

Same Kokoro pipeline, but returns a numpy float32 array instead of writing to a file. This is then fed directly to PyAudio.

---

## 9. formatter.py — Standalone Utility (Importance: 3/10)

```python
import sys
from ollama_model import ollama_chat

def format_file(input_content: str, switch=True):
    def formatter(unformated_content):
        example = """
        If the file has any occurrence of:
            ```bash
                some command here
            ```
            Ensure this text is embellished and put in its own line
        """
        added_prompt = f"Format this file according to the following example {example}"
        llm_prompt = unformated_content + added_prompt
        llm_content = ollama_chat(llm_prompt)
        return llm_content
    # ... reads file, passes to formatter
```

This is a standalone CLI tool (`python formatter.py <filename>`) that sends file content to the Ollama LLM with a formatting prompt. It's not connected to the voice assistant pipeline — it's a separate utility.

---

## 10. import.py — Legacy/Reference Architecture (Importance: 2/10 for active code, but architecturally interesting)

This file contains a **completely different and much more sophisticated architecture** that is not used by the main execution path. It appears to be either an earlier experiment or a more advanced version:

```python
from stt.VoiceActivityDetection import VADDetector
from mlx_lm import load, generate
from melo.api import TTS
import librosa
import sounddevice as sd
from playsound import playsound
from pydantic import BaseModel
```

**Key differences from main.py:**

| Feature | main.py | import.py |
|---------|---------|-----------|
| STT | Google SpeechRecognition (cloud) | VADDetector + FastTranscriber (local MLX Whisper) |
| VAD | Energy-based (SpeechRecognition built-in) | Neural VADDetector |
| LLM | Ollama server (remote) | MLX Llama 3 8B (local Apple Silicon) |
| TTS | Kokoro + ffplay (file-based) | MeloTTS + sounddevice (direct stream) |
| Audio I/O | PyAudio (implicit via SpeechRec) | sounddevice (explicit) |
| History | None | ChatML-formatted conversation history |
| Trimming | None | librosa.effects.trim (silence removal) |

The `Client` class in `import.py` is architecturally closer to Voice Loop — it has VAD, local models, direct audio streaming, and conversation history. But it depends on packages not in `pyproject.toml` (mlx_lm, melo, playsound, pydantic) and references local modules (`stt.VoiceActivityDetection`, `stt.whisper.transcribe`) that don't exist in the repository.

---

## 11. Threading Model

### main.py — Single Thread

```
MAIN THREAD
  transcribeLoop() [blocks on mic]
  → think() [blocks on HTTP to Ollama]
  → modelAnswer() [blocks on Kokoro + ffplay]
  → repeat
```

In non-streaming mode, everything runs sequentially on the main thread. No concurrency.

### main.py streaming mode — Still Single Thread

```
MAIN THREAD
  transcribeLoop() [blocks]
  → remoteThink_ollama_stream() [generator, same thread]
  → modelAnswer_stream() [same thread, blocks on each sentence]
  → repeat
```

Even in "streaming mode," the generator and consumer run on the same thread. The LLM chunk generator yields on the main thread, and `speakText_chunk()` blocks the main thread. There is no real overlap.

### test_pyaudio.py — Three Threads

```
Thread 1: ai_producer      → text_queue  →
Thread 2: tts_consumer     → audio_queue →
Thread 3: playback_consumer → PyAudio    → Speakers
```

This is the only file with real concurrency. Three threads connected by two `queue.Queue` instances, forming a producer-consumer chain.

---

## 12. Audio Output — Two Approaches

### Approach A: ffplay (used by main.py / tts.py)

```python
sf.write(filename, all_audio, 24000)       # write WAV to disk
os.system(f"ffplay {filename} -autoexit -nodisp")  # shell out to ffplay
os.remove(filename)                         # delete file
```

- Requires **FFmpeg** (`ffplay`) installed on the system
- File I/O overhead: write to disk, read from disk, delete
- Blocking via `os.system()` — no way to interrupt mid-playback
- No programmatic control over the audio stream

### Approach B: PyAudio (used by test_pyaudio.py)

```python
py_audio = pyaudio.PyAudio()
stream = py_audio.open(format=pyaudio.paFloat32, channels=1,
                       rate=24000, output=True, frames_per_buffer=1024)
for i in range(0, len(audio_data), 1024):
    chunk = audio_data[i:i+1024]
    stream.write(chunk.tobytes())
stream.stop_stream()
stream.close()
py_audio.terminate()
```

- Direct hardware access via PortAudio (same backend as sounddevice)
- 1024-sample frames ≈ 42ms at 24kHz
- `paFloat32` format — matches Kokoro's output natively
- Programmatic control — can stop/interrupt between frames
- No file I/O — audio goes directly from memory to speakers

---

## 13. Complete Dependency Map

### Third-Party (by importance)

| Importance | Package | Role | Key Methods | Where Used |
|:----------:|---------|------|-------------|------------|
| **10** | `speechrecognition` | STT — the only input path in main.py | `Recognizer()`, `Microphone()`, `listen()`, `recognize_google()`, `adjust_for_ambient_noise()` | stt.py → main.py |
| **9** | `kokoro` | TTS synthesis | `KPipeline(lang_code='a')`, `pipeline(text, voice='af_heart')` → yields `(gs, ps, audio)` | tts.py → main.py, test_pyaudio.py |
| **9** | `ollama` | Primary LLM (remote server) | `Client(host=...)`, `client.chat(model, messages)`, `client.chat(..., stream=True)` | ollama_model.py → main.py |
| **8** | `google-genai` | Alternative LLM (cloud API) | `genai.Client(api_key=...)`, `client.models.generate_content()`, `client.models.generate_content_stream()` | gemini_model.py → test_pyaudio.py |
| **7** | `pyaudio` | Direct audio output (streaming path) | `PyAudio()`, `.open(format, channels, rate, output, frames_per_buffer)`, `stream.write()`, `stream.stop_stream()` | test_pyaudio.py |
| **7** | `torch` | Tensor conversion for Kokoro audio | `torch.tensor(all_audio)`, `.numpy()`, `.astype()` | tts.py, test_pyaudio.py |
| **6** | `numpy` | Audio buffer manipulation in streaming path | `.astype(np.float32)`, array slicing | test_pyaudio.py |
| **5** | `soundfile` | WAV file I/O | `sf.write(filename, audio, samplerate)` | tts.py |
| **5** | `IPython` | Inline audio playback in notebooks | `display(Audio(data, rate, autoplay))` | tts.py |
| **4** | `librosa` | Audio trimming (silence removal) | `librosa.effects.trim(data, top_db=20)` | import.py (unused) |
| **3** | `whisper` | File-based STT alternative | `whisper.load_model("turbo")`, `model.transcribe(file)` | whisper.py (standalone) |

### Standard Library (by importance)

| Importance | Module | Role | Where Used |
|:----------:|--------|------|------------|
| **8** | `threading` | Producer-consumer threads in streaming mode | test_pyaudio.py |
| **8** | `queue` | Thread-safe queues connecting pipeline stages | test_pyaudio.py |
| **7** | `os` | Environment variables, system calls (ffplay), file deletion | main.py, tts.py, gemini_model.py |
| **5** | `sys` | CLI argument parsing | formatter.py, whisper.py, tts.py, gemini_model.py, ollama_model.py |
| **4** | `random` | Random filename generation for WAV files | tts.py |
| **3** | `time` | Sleep between TTS playback and listening restart | import.py (unused) |

---

## 14. Configuration & Environment

### Environment Variables

```bash
GEMINI_API_KEY    # Required for Google Gemini API (gemini_model.py)
STREAMING_MODE    # "true" or "false" — toggles streaming in main.py (set in mise.toml)
```

### mise.toml

```toml
[tools]
uv = "latest"

[tasks.go]
run = "uv run main.py"

[env]
STREAMING_MODE = "true"
```

Running `mise run go` executes `main.py` with `STREAMING_MODE=true`.

### Hardcoded Configuration

```python
# Ollama server address (ollama_model.py)
host = "http://172.16.5.39:3000"
model = "gemma3:4b"

# Kokoro voice (tts.py)
voice = "af_heart"
lang_code = "a"  # US English

# Gemini model (gemini_model.py)
model = "gemini-2.5-flash"

# Audio sample rate (tts.py, test_pyaudio.py)
rate = 24000  # Kokoro's native output rate
```

---

## 15. The Data Flow — Types at Each Stage

### main.py (non-streaming)

```
Microphone (hardware)
  ↓ sr.Microphone() → sr.Recognizer().listen()
sr.AudioData object (internal binary format)
  ↓ recognizer.recognize_google()
str (transcribed text, via Google Cloud API)
  ↓ stored in self.sttData
str
  ↓ ollama_chat(content)
ChatResponse object
  ↓ response["message"]["content"]
str (generated text)
  ↓ speakText(text)
list of numpy float32 arrays (from Kokoro generator)
  ↓ torch.tensor(all_audio)
torch.Tensor
  ↓ sf.write(filename, all_audio, 24000)
WAV file on disk (24kHz, 16-bit)
  ↓ os.system("ffplay ...")
Speakers (hardware, via ffplay process)
```

### test_pyaudio.py (streaming)

```
str (hardcoded test input)
  ↓ gemini_chat_stream(input)
generator yielding str chunks
  ↓ ai_producer thread: accumulate → sentence boundary → text_queue
str (complete sentences)
  ↓ tts_consumer thread: Kokoro pipeline → torch.tensor → .numpy()
numpy float32 array (24kHz audio samples)
  ↓ audio_queue
numpy float32 array
  ↓ playback_consumer thread: .tobytes() → stream.write()
Raw bytes (float32, 1024 samples per frame)
  ↓ PyAudio stream
Speakers (hardware, direct)
```

---

## 16. Latency Budget — Where Time Goes

### main.py (non-streaming)

| Stage | Typical Latency | Notes |
|-------|:---------------:|-------|
| Ambient noise calibration | 200ms | `adjust_for_ambient_noise(duration=0.2)` |
| Speech capture | 1-5s | Blocks until user stops speaking |
| Google STT API | 500-2000ms | Network round-trip to Google's servers |
| Ollama LLM response | 2-10s | Depends on server load, model size, network to LAN server |
| Kokoro TTS synthesis | 300-1000ms | CPU, full response at once |
| File write + ffplay startup | 100-300ms | Disk I/O + process spawn overhead |
| **Total (speech end to first sound)** | **~4-14 seconds** | Dominated by LLM + sequential TTS |

### test_pyaudio.py (streaming)

| Stage | Typical Latency | Notes |
|-------|:---------------:|-------|
| Gemini first chunk | 300-1000ms | Cloud API, first token |
| First sentence complete | 500-2000ms | Depends on sentence length |
| Kokoro TTS (first sentence) | 300-800ms | CPU |
| PyAudio playback start | <10ms | Direct hardware stream |
| **Total (input to first sound)** | **~1-3 seconds** | Overlap between stages saves 2-5s |

---

## 17. Architectural Gaps and Observations

1. **No VAD** — SpeechRecognition's `listen()` uses energy-based detection, not a neural VAD. It's prone to false triggers from background noise and can't distinguish mid-sentence pauses from end-of-turn.

2. **No echo cancellation** — There's no AEC. If the speakers are loud enough, the mic will pick up the TTS output and try to transcribe it as user speech.

3. **No conversation history** — main.py sends only the current utterance to the LLM. Each turn is independent — the model has no memory of previous exchanges.

4. **No barge-in / interrupt** — Once TTS starts playing (via `os.system("ffplay ...")`), there's no way to stop it except killing the process.

5. **Cloud-dependent STT** — `recognize_google()` requires internet. The project has `whisper.py` as a local alternative but it's not integrated.

6. **File-based TTS playback** — Writing to disk, spawning ffplay, and deleting is high-latency compared to direct audio streaming (which test_pyaudio.py demonstrates with PyAudio).

7. **No streaming overlap in main.py** — Despite having a "streaming mode," the generator and consumer run on the same thread. test_pyaudio.py shows the correct three-thread architecture.

8. **import.py is orphaned** — Contains the most sophisticated architecture (VAD, MLX, MeloTTS, conversation history) but depends on packages not in pyproject.toml and local modules that don't exist.

---

## 18. Comparison: VirtualJarvis vs Voice Loop

| Aspect | VirtualJarvis (main.py) | Voice Loop |
|--------|------------------------|------------|
| STT | Google Cloud API (internet required) | Moonshine (local, CPU) |
| VAD | Energy threshold (SpeechRecognition) | Silero VAD (neural network) |
| Turn detection | None (waits for silence timeout) | Smart Turn v3 (neural classifier) |
| LLM | Ollama remote server (LAN) | Gemma 4 E4B (local, Apple Metal via MLX) |
| TTS | Kokoro → WAV file → ffplay | Kokoro → direct OutputStream |
| Audio I/O | PyAudio (implicit) / ffplay | sounddevice (explicit, InputStream + OutputStream) |
| Echo cancellation | None | WebRTC AEC3 (LiveKit APM) |
| Barge-in | None | VAD on AEC-cleaned audio |
| Conversation history | None | Last 10 turns |
| Memory system | None | LLM-powered MEMORY.md |
| Streaming overlap | Sequential (main.py) / 3-thread (test_pyaudio.py) | LLM thread → sentence queue → asyncio TTS |
| Latency (mic to sound) | ~4-14s | ~2-4s |
| Runs offline | No (Google STT requires internet) | Yes (fully local) |
| File count | 8 source files | 1 file |

---

## 19. How to Run

```bash
# Install dependencies
cd /home/oj2/projects/clones/VirtualJarvis
uv sync

# Set environment
export GEMINI_API_KEY="your-key"    # only if using Gemini
export STREAMING_MODE="true"        # enable streaming (already set in mise.toml)

# Run via mise
mise run go

# Or directly
uv run main.py

# Run the streaming test harness
uv run test_pyaudio.py
```

**Prerequisites:**
- `ffplay` (from FFmpeg) installed on the system for audio playback in main.py
- Network access to `http://172.16.5.39:3000` for Ollama (or update the host in `ollama_model.py`)
- Internet access for Google SpeechRecognition STT
- `GEMINI_API_KEY` if using the Gemini path (test_pyaudio.py)
