# VirtualMlx MVP Build

## VirtualMlx MVP — Amalgamation Build Plan

### Goal
Build the MVP voice agent combining Voice Loop's proven local tech stack with VirtualJarvis's code style, using Ollama (gemma4:cloud) as the LLM.

### Architecture
```
Mic → sounddevice → Silero VAD → Smart Turn → Moonshine STT → Ollama (gemma4:cloud) → Kokoro TTS (am_michael) → sounddevice → Speakers
```

### File Structure
| File | Purpose |
|------|---------|
| `config.py` | All constants: sample rates, thresholds, model names, paths |
| `listener.py` | Mic capture via sounddevice + Silero VAD + Smart Turn v3 (ONNX) |
| `transformer.py` | Moonshine STT + Kokoro TTS synthesis + audio playback |
| `model.py` | Ollama client with streaming, sentence splitting, conversation history |
| `main.py` | `VirtualMlx` orchestrator class (listen → transcribe → think → speak) |
| `SOUL.md` | System prompt (live-reloaded each turn) |
| `pyproject.toml` | Dependencies |
| `mise.toml` | Tasks + env vars |
| `ROADMAP.md` | MVP → Barehands → LangChain → Augmentation |
| `README.md` | Python version, tech stack, file descriptions |

### Key Decisions
- **VAD**: Silero (ONNX) — proven, fast, ~10MB
- **Turn Detection**: Smart Turn v3 (ONNX via onnx-community/smart-turn-v3-ONNX) with WhisperFeatureExtractor
- **STT**: Moonshine (local, CPU) — no cloud dependency
- **TTS**: Kokoro v1.0 (ONNX) with `am_michael` (US male, C+ grade) — replaces `af_heart`
- **LLM**: Ollama `gemma4:cloud` via `ollama` Python package (streaming)
- **Audio I/O**: sounddevice (InputStream for mic, OutputStream for playback — no ffplay)
- **Voice**: Changed from female `af_heart` → male `am_michael`

### Streaming Approach
- LLM streams tokens via `ollama.chat(stream=True)`
- `SentenceAccumulator` detects sentence boundaries and dispatches to TTS
- Each sentence is synthesized and played immediately (sequential but first-sentence-fast)
