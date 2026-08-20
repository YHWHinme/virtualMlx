"""Sandboxed file tools for the VirtualMlx agent.

Every create/edit/read is confined to OUTPUT_DIR (see config.py). A path
that would resolve outside it — via `..`, an absolute path, a drive
letter, or a symlink — is rejected before any disk write, so the agent
cannot touch files outside the sandbox.

Runnable standalone:  uv run python UTILITIES/file_tools.py
"""

import sys
from pathlib import Path

# Allow running this file directly as a script so `config` is importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool

from config import OUTPUT_ALLOWED_SUFFIXES, OUTPUT_DIR


class SandboxError(ValueError):
    """Raised when a path would escape the OUTPUT sandbox."""


def _resolve(path: str) -> Path:
    """Resolve *path* under OUTPUT_DIR and enforce the sandbox."""
    if not path or not isinstance(path, str):
        raise SandboxError("path must be a non-empty string")

    root = OUTPUT_DIR.resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Only relative paths allowed — no absolute paths, no drive letters.
    if path.startswith("/") or Path(path).is_absolute() or ":" in path[:3]:
        raise SandboxError(
            f"only relative paths are allowed (got: {path!r}); "
            "name the file relative to the OUTPUT folder"
        )

    # Markdown by default when no extension is given.
    if not Path(path).suffix:
        path = path + ".md"

    candidate = (root / path).resolve()

    # Traversal guard: the resolved path must be OUTPUT_DIR or sit beneath it.
    if root != candidate and root not in candidate.parents:
        raise SandboxError(f"path escapes the OUTPUT sandbox: {path!r} -> {candidate}")

    return candidate


def _check_suffix(path: Path) -> None:
    """Allow only text-friendly extensions (no binaries or executables)."""
    if path.suffix.lower() not in OUTPUT_ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(OUTPUT_ALLOWED_SUFFIXES))
        raise SandboxError(f"extension {path.suffix!r} not allowed; permitted: {allowed}")


@tool
def create_file(path: str, content: str | bytes) -> str:
    """Create or overwrite a file inside the OUTPUT folder.

    Files are Markdown by default: a path with no extension gets `.md`
    appended. Content may be a string or bytes. Parent folders are
    created automatically. Only text file extensions are permitted.

    Args:
        path: Relative file path within the OUTPUT folder (no leading /).
        content: The full text or bytes to write — the whole block at once.

    Returns:
        A short confirmation with the byte count written.
    """
    target = _resolve(path)
    _check_suffix(target)
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    rel = target.relative_to(OUTPUT_DIR.resolve())
    return f"Created OUTPUT/{rel} ({len(data)} bytes)."


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact, unique occurrence of text in an OUTPUT file.

    `old_text` must match exactly one region (case- and whitespace-
    sensitive). Call `read_file` first to get the exact current text.

    Args:
        path: Relative file path within the OUTPUT folder (no leading /).
        old_text: The exact text to find — must be unique in the file.
        new_text: The replacement text.

    Returns:
        A short confirmation, or an error if old_text was not found /
        was ambiguous.
    """
    target = _resolve(path)
    _check_suffix(target)
    if not target.exists():
        return f"Error: OUTPUT file not found: {path!r}"
    if old_text == "":
        return "Error: old_text is empty — nothing to match."
    if old_text == new_text:
        return "No change: old_text and new_text are identical."

    body = target.read_text(encoding="utf-8")
    occurrences = body.count(old_text)
    if occurrences == 0:
        return "Error: old_text not found in the file; call read_file for the exact text."
    if occurrences > 1:
        return f"Error: old_text matches {occurrences} times — add context so it matches once."

    target.write_text(body.replace(old_text, new_text, 1), encoding="utf-8")
    rel = target.relative_to(OUTPUT_DIR.resolve())
    return f"Edited OUTPUT/{rel} (1 replacement)."


@tool
def read_file(path: str) -> str:
    """Read the full text of a file inside the OUTPUT folder.

    Args:
        path: Relative file path within the OUTPUT folder (no leading /).

    Returns:
        The file's full text, or an error if it does not exist.
    """
    target = _resolve(path)
    _check_suffix(target)
    if not target.exists():
        return f"Error: OUTPUT file not found: {path!r}"
    return target.read_text(encoding="utf-8")


# The list handed to create_agent().
file_tools = [create_file, edit_file, read_file]


if __name__ == "__main__":
    # Simple standalone check — one call, string content. The sandbox
    # confines it to OUTPUT/; nothing outside that can be written.
    print(create_file.invoke({
        "path": "check.md",
        "content": "# Hello\n\nWritten to the OUTPUT sandbox in one call.\n",
    }))
