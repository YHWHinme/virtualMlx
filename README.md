# VirtualMlx

Local voice agent combining proven on-device speech tech with a cloud LLM.

## Requirements

- **Python** ≥ 3.11
- **PortAudio** (for sounddevice)
  - macOS: `brew install portaudio`
  - Ubuntu: `sudo apt install portaudio19-dev python3-dev`
- **espeak-ng** (for Kokoro TTS phonemes)
  - macOS: `brew install espeak-ng`
  - Ubuntu: `sudo apt install espeak-ng`
- **Ollama** running with the `gemma4:cloud` model pulled
- **barehands** server (for the glass board)
  - `cd /home/oj2/projects/clones/barehands && python3 server.py`
  - open `http://127.0.0.1:8794/stage.html` in Chrome for the tracker

## Quick Start

```bash
uv sync

# Terminal 1 — the glass board (blocking, foreground)
mise run barehands    # or: uv run board.py
# then open http://127.0.0.1:8794/stage.html in Chrome

# Terminal 2 — the voice agent
mise run go           # or: uv run main.py
```

## Tech Stack

| Layer | Technology | Runs On |
|-------|-----------|---------|
| Speech detection | Silero VAD (ONNX) | CPU, ~10 MB |
| Turn detection | Smart Turn v3 (ONNX) | CPU, ~8 MB |
| Speech-to-text | Moonshine | CPU, ~250 MB |
| Text-to-speech | Kokoro v1.0 (ONNX) | CPU, ~300 MB |
| LLM | Ollama (`gemma4:cloud`) | Cloud |
| Glass board | barehands | Localhost |

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | `VirtualMlx` class — the listen → think → speak loop |
| `board.py` | barehands bridge + run as blocking server (`uv run board.py`) |
| `UTILITIES/listener.py` | Mic capture via sounddevice, Silero VAD, Smart Turn |
| `UTILITIES/transformer.py` | Moonshine STT + Kokoro TTS + volume/loudness indicator |
| `UTILITIES/model.py` | Ollama client with streaming, sentence splitting, history |
| `config.py` | All constants: sample rates, thresholds, model names, paths |
| `SYSTEM.md` | System prompt — live-reloaded each turn |

## Configuration

Environment variables (override defaults in `config.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `gemma4:cloud` | Ollama model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `BAREHANDS_HOST` | `127.0.0.1` | barehands server host |
| `BAREHANDS_PORT` | `8794` | barehands server port |
| `TTS_VOLUME` | `12` | Agent response loudness (1-20, 10 = normal) |
