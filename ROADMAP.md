# Roadmap

## MVP
- [x] Silero VAD for speech detection (local, ONNX)
- [x] ONNXRuntime Smart Turn v3 for conversation end estimation
- [x] Moonshine for local speech-to-text
- [x] Kokoro TTS with male voice (`am_michael`)
- [x] Ollama LLM (`gemma4:cloud`) with streaming
- [x] Conversation history (last 10 turns)
- [x] Sentence-level streaming TTS dispatch

## LangChain Replacement ← we are here
- [ ] Replace raw `ollama` package with LangChain
- [ ] Structured prompt management
- [ ] Better conversation memory (summarization, retrieval)
- [ ] Websearch capabilities (via tool calling)

## Barehands Integration
- [x] Wire barehands glass board as the visual interface
- [x] Ring face: reflect agent state (idle / listening / thinking / speaking)
- [ ] Hub swiping: feed notes and 3D models into the AI context via gestures
- [ ] Board commands: agent stages cards, images, and models on the glass

## Barehands Augmentation
- [ ] Tool calling for the Jarvis UI (tools TBD)
- [ ] Machine control via the interface (open Steam, browser, etc.)
- [ ] Barge-in with WebRTC AEC3 echo cancellation

## Daemonize
- [ ] Run the agent as a background daemon on the machine

## Memory
- [ ] Add mem0 (or similar memory tech) for better context management and chat history

## Rust Port (distant future)
- [ ] Port the full system to Rust
