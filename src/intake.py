"""Diff intake: the front door of the review desk (FR-1).

Every diff — read from a path by the CLI or pasted into the Chainlit UI —
flows through this module and is split into per-file chunks *before* any
model call. Nothing here talks to a model or the network.

House rules honoured here (NFR-4): failures become friendly, actionable
sentences (a ``DiffError.message`` or a returned ``Error: ...`` string),
never tracebacks.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DiffError",
    "ReviewContext",
    "intake",
    "load_ruleset",
    "read_diff",
    "split_diff",
]

_GIT_HEADER = "diff --git "
_HUNK_HEADER = "@@"
_RULESETS_SUBDIR = Path("data") / "rulesets"


@dataclass
class ReviewContext:
    """Per-review context handed to every agent run (FR-2).

    Exactly the PDF shape. Tools read it through ``RunContextWrapper``; its
    content never appears in prompt text.
    """

    repo: str
    language: str
    ruleset_id: str
    strictness: str = "normal"


class DiffError(Exception):
    """A diff problem a human can act on: empty input, malformed text, bad path.

    Callers catch this and render ``message`` to the user — one or two plain
    sentences, never a traceback.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


async def read_diff(path: str | Path) -> str:
    """Read a unified diff from *path* (async entry, FR-1).

    Raises:
        DiffError: with a friendly sentence when the file is missing, is a
            directory, is unreadable, or is not UTF-8 text. Nothing else
            escapes; callers render ``DiffError.message`` directly.
    """
    diff_path = Path(path)
    try:
        # Real file I/O stays off the event loop; the read itself is tiny.
        return await _read_text_off_loop(diff_path)
    except FileNotFoundError:
        raise DiffError(
            f"I couldn't find a diff at '{diff_path}' — check the path and try again."
        ) from None
    except IsADirectoryError:
        raise DiffError(
            f"'{diff_path}' is a directory, not a diff file — point me at a file."
        ) from None
    except PermissionError:
        raise DiffError(
            f"I don't have permission to read '{diff_path}' — check the file's permissions."
        ) from None
    except UnicodeDecodeError:
        raise DiffError(
            f"'{diff_path}' is not a UTF-8 text file, so I couldn't read it as a diff."
        ) from None
    except OSError as exc:
        reason = exc.strerror or "unexpected OS error"
        raise DiffError(f"I couldn't read the diff at '{diff_path}' ({reason}).") from None


def split_diff(text: str) -> list[str]:
    """Split a git-style unified diff into per-file chunks (pure, no model).

    A chunk starts at a ``diff --git a/x b/y`` line and holds everything for
    that file: ``index``, ``new file mode`` / ``deleted file mode``,
    ``similarity index`` / ``rename from`` / ``rename to``, ``---``/``+++``
    headers and all ``@@`` hunks, up to (not including) the next
    ``diff --git`` line. Preamble before the first header is dropped.

    Safe against hunks that *add* a line looking like a git header: every
    hunk-body line starts with ``+``, ``-``, a space, or ``\\``, never with
    ``diff --git``.

    Returns ``[]`` when the text has no ``diff --git`` header at all (plain
    ``---``/``+++`` diffs without git headers are treated as malformed, per
    the intake contract) — the validating wrapper ``intake`` turns that into
    a ``DiffError``.
    """
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith(_GIT_HEADER):
            if current:
                chunks.append("".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        chunks.append("".join(current))
    return chunks


async def intake(diff_text: str) -> list[str]:
    """Validate *diff_text* and split it into per-file chunks (FR-1).

    The async front door of the desk: combines validation and the pure
    :func:`split_diff` step, and is awaited before any model call.

    Raises:
        DiffError: empty/whitespace-only input, text with no ``diff --git``
            sections, or sections with no ``@@`` hunks — with a friendly,
            actionable message. Callers catch it and render ``.message``;
            nothing else is ever raised, so no traceback can reach the user.

    A section without hunks (e.g. a pure rename) is kept when the diff has
    real changes elsewhere; only a diff where *no* section has hunks is
    rejected as having nothing to review.
    """
    if not diff_text or not diff_text.strip():
        raise DiffError(
            "Nothing to review yet — paste or point me at a unified diff "
            "and I'll take it from there."
        )
    chunks = split_diff(diff_text)
    if not chunks:
        raise DiffError(
            "That text doesn't look like a unified diff — I need at least one "
            "'diff --git' file section. Run `git diff` (or `git diff --staged`) "
            "and paste or point me at the result."
        )
    if not any(_has_hunks(chunk) for chunk in chunks):
        raise DiffError(
            "I found 'diff --git' headers but no change hunks ('@@' lines) in "
            "that text — there's nothing to review yet. Export the change with "
            "`git diff` and try again."
        )
    return chunks


def load_ruleset(ruleset_id: str, base_dir: Path | None = None) -> str:
    """Load ``data/rulesets/<ruleset_id>.json`` formatted for prompt injection.

    The file must be a JSON object ``{"ruleset_id": str, "name": str,
    "rules": [{"id": str, "area": str, "rule": str}, ...]}``; ``id`` and
    ``area`` are optional per rule. The return value is readable text the
    caller can drop into reviewer instructions (FR-4).

    Never raises (NFR-4 — this is later wired into a tool): any failure —
    unsafe id, missing file, unreadable file, invalid JSON, wrong shape —
    comes back as one ``Error: ruleset '<id>' could not be loaded (<reason>)
    sentence telling the model to continue on built-in judgement.
    """
    root = Path(base_dir).resolve() if base_dir is not None else _project_root()

    def fail(reason: str) -> str:
        return (
            f"Error: ruleset '{ruleset_id}' could not be loaded ({reason}). "
            "Continue using built-in judgement and say so."
        )

    if not ruleset_id or Path(ruleset_id).name != ruleset_id:
        return fail("the ruleset id must be a plain name like 'default'")

    ruleset_path = root / _RULESETS_SUBDIR / f"{ruleset_id}.json"
    try:
        raw = ruleset_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return fail(f"no ruleset file at data/rulesets/{ruleset_id}.json")
    except UnicodeDecodeError:
        return fail(f"data/rulesets/{ruleset_id}.json is not UTF-8 text")
    except OSError as exc:
        reason = exc.strerror or "unexpected OS error"
        return fail(f"data/rulesets/{ruleset_id}.json could not be read ({reason})")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return fail(f"data/rulesets/{ruleset_id}.json is not valid JSON ({exc.msg} at line {exc.lineno})")

    return _format_ruleset(ruleset_id=ruleset_id, data=data, fail=fail)


def _project_root() -> Path:
    """Project root, resolved from this module file (src/intake.py)."""
    return Path(__file__).resolve().parent.parent


async def _read_text_off_loop(diff_path: Path) -> str:
    return await asyncio.to_thread(diff_path.read_text, encoding="utf-8")


def _has_hunks(chunk: str) -> bool:
    return any(line.startswith(_HUNK_HEADER) for line in chunk.splitlines())


def _format_ruleset(
    ruleset_id: str, data: object, fail: Callable[[str], str]
) -> str:
    """Render a parsed ruleset as readable text; shape problems -> fail()."""
    if not isinstance(data, dict):
        return fail("the ruleset file must contain a JSON object")
    name = data.get("name")
    rules = data.get("rules")
    if not isinstance(name, str) or not name.strip():
        return fail("the ruleset file has no usable 'name' field")
    if not isinstance(rules, list) or not rules:
        return fail("the ruleset file contains no rules")

    lines = [f'Ruleset "{name.strip()}" (id: {ruleset_id}, rules below):']
    for position, entry in enumerate(rules, start=1):
        if not isinstance(entry, dict):
            return fail(f"rule {position} is not an object")
        text = entry.get("rule")
        if not isinstance(text, str) or not text.strip():
            return fail(f"rule {position} has no rule text")
        tag = entry.get("id")
        tag_text = f"[{tag}]" if isinstance(tag, str) and tag.strip() else f"[R{position}]"
        area = entry.get("area")
        area_text = f" ({area.strip()})" if isinstance(area, str) and area.strip() else ""
        lines.append(f"- {tag_text}{area_text} {text.strip()}")
    return "\n".join(lines)
