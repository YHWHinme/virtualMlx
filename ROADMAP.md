# Roadmap

## MVP ← we are here
- [x] Silero VAD for speech detection (local, ONNX)
- [x] ONNXRuntime Smart Turn v3 for conversation end estimation
- [x] Moonshine for local speech-to-text
- [x] Kokoro TTS with male voice (`am_michael`)
- [x] Ollama LLM (`gemma4:cloud`) with streaming
- [x] Conversation history (last 10 turns)
- [x] Sentence-level streaming TTS dispatch

## Barehands Integration
- [ ] Wire barehands glass board as the visual interface
- [ ] Hub swiping: feed notes and 3D models into the AI context via gestures
- [ ] Ring face: reflect agent state (idle / listening / thinking / speaking)
- [ ] Board commands: agent stages cards, images, and models on the glass

## LangChain Replacement
- [ ] Replace raw `ollama` package with LangChain
- [ ] Structured prompt management
- [ ] Better conversation memory (summarization, retrieval)

## Barehands Augmentation
- [ ] Tool calling for the Jarvis UI (tools TBD)
- [ ] Machine control via the interface (open Steam, browser, etc.)
- [ ] Barge-in with WebRTC AEC3 echo cancellation
