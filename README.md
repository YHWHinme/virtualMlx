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

## Quick Start

```bash
uv sync
mise run go        # or: uv run main.py
```

## Tech Stack

| Layer | Technology | Runs On |
|-------|-----------|---------|
| Speech detection | Silero VAD (ONNX) | CPU, ~10 MB |
| Turn detection | Smart Turn v3 (ONNX) | CPU, ~8 MB |
| Speech-to-text | Moonshine | CPU, ~250 MB |
| Text-to-speech | Kokoro v1.0 (ONNX) | CPU, ~300 MB |
| LLM | Ollama (`gemma4:cloud`) | Cloud |

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | `VirtualMlx` class — the listen → think → speak loop |
| `listener.py` | Mic capture via sounddevice, Silero VAD, Smart Turn |
| `transformer.py` | Moonshine STT + Kokoro TTS synthesis + audio playback |
| `model.py` | Ollama client with streaming, sentence splitting, history |
| `config.py` | All constants: sample rates, thresholds, model names, paths |
| `SOUL.md` | System prompt — live-reloaded each turn |

## Configuration

Environment variables (override defaults in `config.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `gemma4:cloud` | Ollama model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
