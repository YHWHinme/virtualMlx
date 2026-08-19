# Graph Report - VirtualMlx  (2026-08-19)

## Corpus Check
- Corpus is ~39,351 words - fits in a single context window. You may not need a graph.

## Summary
- 149 nodes · 210 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Voice Agent Core
- Board Bridge (board.py)
- Reference Systems & Concepts
- LLM Client (model.py)
- Mic Listener (listener.py)
- barehands Animations & Gestures
- barehands Integration Seams
- UTILITIES Package
- VirtualMlx Singleton
- Future Memory (mem0)

## God Nodes (most connected - your core abstractions)
1. `Voice Loop on-device agent` - 13 edges
2. `Model` - 10 edges
3. `Listener` - 9 edges
4. `Transformer` - 9 edges
5. `barehands glass interface system` - 8 edges
6. `cmd()` - 7 edges
7. `VirtualMlx` - 7 edges
8. `VirtualJarvis voice agent` - 7 edges
9. `SentenceAccumulator` - 6 edges
10. `VirtualMlx project` - 6 edges

## Surprising Connections (you probably didn't know these)
- `VirtualMlx project` --semantically_similar_to--> `VirtualJarvis voice agent`  [INFERRED] [semantically similar]
  README.md → INSIGHTS/INSIGHT_VIRTUALJARVIS.md
- `VirtualMlx project` --semantically_similar_to--> `Voice Loop on-device agent`  [INFERRED] [semantically similar]
  README.md → INSIGHTS/INSIGHT_VOICELOOP.md
- `Gemma 4 E4B via MLX (local)` --semantically_similar_to--> `Ollama LLM (gemma4:cloud)`  [INFERRED] [semantically similar]
  INSIGHTS/INSIGHT_VOICELOOP.md → README.md
- `SentenceAccumulator streaming` --semantically_similar_to--> `Three-thread PyAudio streaming pipeline`  [INFERRED] [semantically similar]
  .agents/plans/2026-08-19T01-27-39-919Z-virtualmlx-mvp-build.md → INSIGHTS/INSIGHT_VIRTUALJARVIS.md
- `SentenceAccumulator streaming` --semantically_similar_to--> `Sentence-streaming TTS overlap`  [INFERRED] [semantically similar]
  .agents/plans/2026-08-19T01-27-39-919Z-virtualmlx-mvp-build.md → INSIGHTS/INSIGHT_VOICELOOP.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **On-device voice pipeline stages** — readme_silero_vad, readme_smart_turn, readme_moonshine, readme_kokoro, insights_insight_voiceloop_sounddevice [EXTRACTED 0.95]
- **Three reference voice-agent systems** — readme_virtualmlx, insights_insight_voiceloop_voiceloop, insights_insight_virtualjarvis_jarvis [INFERRED 0.85]
- **barehands three integration seams** — insights_implementation_barehands_ring_seam, insights_implementation_barehands_cmd_seam, insights_implementation_barehands_eyes_seam [EXTRACTED 1.00]

## Communities (10 total, 3 thin omitted)

### Community 0 - "Voice Agent Core"
Cohesion: 0.11
Nodes (18): _extract_waveform(), main.py — VirtualMlx orchestrator. The listen → transcribe → think → speak…, Convert a float32 audio chunk into 64 amplitude values 0..1., The main voice agent. Boots all subsystems, then enters the main loop., # NOTE: Main loop, listen → transcribe → think → speak, forever., VirtualMlx, _ensure_kokoro_files() (+10 more)

### Community 1 - "Board Bridge (board.py)"
Cohesion: 0.12
Nodes (24): add_card(), add_img(), clear(), cmd(), describe_board(), _ensure_state_dir(), get_board_state(), is_connected() (+16 more)

### Community 2 - "Reference Systems & Concepts"
Cohesion: 0.14
Nodes (24): VirtualMlx MVP amalgamation plan, SentenceAccumulator streaming, ffplay file-based TTS playback, Google SpeechRecognition STT (cloud), import.py orphaned MLX architecture, VirtualJarvis voice agent, Three-thread PyAudio streaming pipeline, WebRTC AEC3 barge-in (+16 more)

### Community 3 - "LLM Client (model.py)"
Cohesion: 0.10
Nodes (14): Configuration constants for VirtualMlx., _load_soul(), Model, model.py — Ollama LLM client with streaming, sentence splitting, and history.…, Build the chat message list: system prompt + history + current input., Blocking chat — sends the full message, waits for the full response., Streaming chat — yields raw text chunks as they arrive from Ollama., Stream the LLM response as complete sentences. Each yielded string is a full… (+6 more)

### Community 4 - "Mic Listener (listener.py)"
Cohesion: 0.11
Nodes (18): Text, _find_microphone(), Listener, _load_smart_turn(), _print_mic_details(), ndarray, listener.py — Microphone capture with Silero VAD and Smart Turn v3. Handles the…, Captures complete utterances from the microphone. Uses Silero VAD as the first… (+10 more)

### Community 5 - "barehands Animations & Gestures"
Cohesion: 0.15
Nodes (13): barehands pipeline animation, VirtualJarvis pipeline animation, Voice Loop pipeline animation, motion.dev scroll animation library, barehands glass interface system, THE CONTRAST LAW (pinch), Heartbeat-as-command-channel, Media airlock security boundary (+5 more)

### Community 6 - "barehands Integration Seams"
Cohesion: 0.20
Nodes (10): board.py three-seam bridge, Board-command seam (POST /cmd), Board-state seam (GET /state), Ring-state seam (write files), LLM tool-calling board dispatch, barehands glass board, Barehands Integration milestone, Hub swiping gesture input (+2 more)

## Knowledge Gaps
- **12 isolated node(s):** `VirtualMlx`, `am_michael male voice`, `Hub swiping gesture input`, `LangChain replacement`, `Barge-in with WebRTC AEC3` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Model` connect `LLM Client (model.py)` to `Voice Agent Core`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `Listener` connect `Mic Listener (listener.py)` to `Voice Agent Core`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Voice Loop on-device agent` (e.g. with `import.py orphaned MLX architecture` and `VirtualMlx project`) actually correct?**
  _`Voice Loop on-device agent` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `VirtualMlx`, `am_michael male voice`, `Hub swiping gesture input` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Voice Agent Core` be split into smaller, more focused modules?**
  _Cohesion score 0.1076923076923077 - nodes in this community are weakly interconnected._
- **Should `Board Bridge (board.py)` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
- **Should `Reference Systems & Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.14492753623188406 - nodes in this community are weakly interconnected._