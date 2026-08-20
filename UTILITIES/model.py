"""model.py — LangChain + Ollama agent with MCP tools and conversation memory.

Replaces the raw ollama package with:
  - ChatOllama (langchain-ollama) as the LLM
  - create_agent (langgraph) for automatic tool-calling + checkpointer memory
  - MultiServerMCPClient (langchain-mcp-adapters) for websearch via MCP
  - SentenceAccumulator for low-latency TTS dispatch

Conversation history is checkpointed per thread so multi-turn context
persists across turns automatically (LangGraph short-term memory).
"""

import re
import urllib.request

# AI Imports
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from config import (
    MAX_TOKENS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    PARALLEL_API_KEY,
    PARALLEL_SEARCH_URL,
    SENTENCE_MIN_CHARS,
    SYSTEM_PATH,
    TEMPERATURE,
)

console = Console()

_SENT_END = re.compile(r"(?<=[.!?])\s+")


# ── System prompt ────────────────────────────────────────────────


def _load_soul() -> str:
    """Read the system prompt from SYSTEM.md."""
    if SYSTEM_PATH.exists():
        return SYSTEM_PATH.read_text().strip()
    return "You are a helpful voice assistant. Speak naturally and concisely."


# ── Sentence accumulator ────────────────────────────────────────


class SentenceAccumulator:
    """Accumulates streaming LLM text and yields complete sentences.

    Short fragments (< *min_chars*) are merged with the next sentence to
    avoid TTS artifacts on abbreviations like "Mr." or "Dr.".
    """

    def __init__(self, min_chars: int = 20):
        self.min_chars = min_chars
        self._buffer = ""
        self._carry = ""

    def feed(self, text: str) -> list[str]:
        """Feed a new text chunk. Returns zero or more complete sentences."""
        self._buffer += text
        sentences: list[str] = []

        while True:
            m = _SENT_END.search(self._buffer)
            if not m:
                break
            fragment = self._buffer[: m.start() + 1].strip()
            self._buffer = self._buffer[m.end() :]

            self._carry = (
                (self._carry + " " + fragment).strip() if self._carry else fragment
            )
            if len(self._carry) >= self.min_chars:
                sentences.append(self._carry)
                self._carry = ""

        return sentences

    def flush(self) -> str | None:
        """Flush any remaining text as a final sentence."""
        remainder = self._buffer.strip()
        if self._carry:
            remainder = (self._carry + " " + remainder).strip()
        self._buffer = ""
        self._carry = ""
        return remainder if remainder else None


# ── Model class ──────────────────────────────────────────────────


class Model:
    """LangChain agent: ChatOllama + MCP tools + checkpointed memory."""

    @classmethod
    async def create(cls) -> "Model":
        """Async factory — sets up the LLM, MCP tools, and the agent."""
        self = cls()


        # create_agent moved between packages across versions — try both
        try:
            from langchain.agents import create_agent
        except ImportError:
            from langgraph.prebuilt import create_agent

        console.print(f"Connecting to Ollama at {OLLAMA_HOST}…")
        self._llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
        )

        # Lightweight connectivity check (no model call — just /api/tags)
        try:
            urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2)  # noqa: ASYNC210
            console.print(f"  Ollama connected [green]✓[/] (model: {OLLAMA_MODEL})")
        except Exception:  # noqa: BLE001
            console.print(f"  [yellow]⚠ Ollama unreachable at {OLLAMA_HOST}[/]")

        # MCP tools (parallel.ai websearch) — checks API key + connection
        tools = await self._load_mcp_tools()

        # Local sandboxed file tools (create/edit/read in OUTPUT/) — always
        # available, no external service needed.
        from UTILITIES.file_tools import file_tools
        tools = tools + file_tools

        soul = _load_soul()
        self._checkpointer = MemorySaver()
        self._thread_id = "virtualmlx"

        self._agent = create_agent(
            self._llm,
            tools=tools,
            system_prompt=soul,
            checkpointer=self._checkpointer,
        )
        console.print(f"  Agent ready [green]✓[/] (tools: {len(tools)})")
        return self

    # ── MCP tool loading ─────────────────────────────────────────

    async def _load_mcp_tools(self) -> list:
        """Load websearch tools from the parallel.ai remote MCP server.

        Uses the streamable-HTTP transport (no local Node.js or playwright).
        basic mode is free & anonymous — an API key is optional and only
        raises rate limits.  Returns an empty list on connection failure.
        """
        console.print("  Connecting to parallel.ai MCP…")
        console.print(f"    url: [dim]{PARALLEL_SEARCH_URL}[/]")

        # basic mode = anonymous; Bearer key optional for higher limits
        headers = (
            {"Authorization": f"Bearer {PARALLEL_API_KEY}"}
            if PARALLEL_API_KEY
            else {}
        )
        if PARALLEL_API_KEY:
            console.print("    auth: [green]Bearer API key[/] (higher limits)")
        else:
            console.print("    auth: [dim]anonymous (basic mode, free)[/]")

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    "parallel-web-search": {
                        "url": PARALLEL_SEARCH_URL,
                        "transport": "streamable_http",
                        "headers": headers or None,
                    }
                }
            )
            tools = await client.get_tools()
            if tools:
                names = ", ".join(t.name for t in tools)
                console.print(f"  parallel.ai connected [green]✓[/] ({len(tools)} tools)")
                console.print(f"    tools: [dim]{names}[/]")
                return tools
            console.print("  [yellow]⚠ connected but no tools returned[/]")
            return []
        except Exception as e:  # noqa: BLE001
            console.print(f"  [yellow]⚠ parallel.ai connection failed: {e}[/]")
            return []

    # ── streaming response ───────────────────────────────────────

    async def stream_sentences(self, user_text: str, break_check=None):
        """Stream the agent's text response as complete sentences.

        Prints debug messages when a websearch tool is called and when it
        returns.  Only the final assistant *text* tokens go to TTS.

        If ``break_check`` (a zero-arg callable returning bool) is supplied,
        the stream stops early when it returns truthy (Ctrl-B barge-in).
        """
        from langchain_core.messages import AIMessageChunk, ToolMessage

        config = {"configurable": {"thread_id": self._thread_id}}
        acc = SentenceAccumulator(min_chars=SENTENCE_MIN_CHARS)
        tool_active = False  # track so we print one debug line per call

        aiter = self._agent.astream(
            {"messages": [{"role": "user", "content": user_text}]},
            config,
            stream_mode="messages",
        )
        # ``completed`` is set True only when the stream exhausts without a
        # break. It gates the post-loop remainder yield so that we NEVER yield
        # out of ``finally`` — yielding during an ``aclose()`` (GeneratorExit)
        # is what raised "async generator ignored GeneratorExit".
        completed = False
        try:
            async for chunk, _metadata in aiter:
                if break_check is not None and break_check():
                    break
                if isinstance(chunk, AIMessageChunk):
                    # Debug: the model just decided to call a tool
                    if chunk.tool_call_chunks and not tool_active:
                        for tc in chunk.tool_call_chunks:
                            name = tc.get("name") if isinstance(tc, dict) else None
                            if name:
                                console.print(f"  [bold yellow]🔍 {name}[/] …")
                                tool_active = True
                                break
                    # Text content → sentence accumulator for TTS
                    content = chunk.content
                    if isinstance(content, str) and content:
                        for sentence in acc.feed(content):
                            yield sentence

                elif isinstance(chunk, ToolMessage):
                    # Debug: the tool returned its result
                    tname = getattr(chunk, "name", "tool") or "tool"
                    console.print(f"  [green]✓ {tname}[/] returned")
                    tool_active = False
            else:
                completed = True
        finally:
            # Cleanup only — NEVER yield here. If the caller abandoned us
            # (barge-in, exception, or aclose) the partial remainder is
            # discarded so we don't speak a trailing fragment.
            if not completed:
                acc.flush()

        # Reached only on normal completion (no break, no GeneratorExit).
        # During an aclose(), control never gets here, so this yield is safe.
        remainder = acc.flush()
        if remainder:
            yield remainder
