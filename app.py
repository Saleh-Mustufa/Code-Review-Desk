"""Code Review Desk — Chainlit UI entry (FR-12, AD-10).

One chat turn is one review: paste a unified diff and the desk streams
progressive status updates plus a severity-styled findings card per reviewer
as it lands, then the merged report, the optional remediation proposal, the
FR-8 refusal when the secret guardrail trips, and the FR-10 measurements
footer (per-reviewer latency and REAL token usage) with the full report
attached as a text element.

Session state (``cl.user_session``) carries the review context, the review
count and the last report (FR-12): a second diff pasted in the same session
REUSES the first review's repo/language/ruleset/strictness unless the user
explicitly overrides them with a ``desk:`` directive line.

House rules honoured here (NFR-4): ANY exception becomes ONE friendly
sentence in the chat — never a traceback; keys and server internals stay in
the server log. The pipeline's async generator is always driven to
exhaustion (``async for`` with no early exit): the generator's ``finally``
resets the planted-secret contextvars, so the stream is never abandoned.
"""

from __future__ import annotations

import asyncio
import os

import chainlit as cl

from src.intake import DiffError, ReviewContext
from src.observe import setup_tracing
from src.pipeline import (
    FindingsLanded,
    GuardrailRefused,
    MergedReport,
    PartialReview,
    RemediationOffered,
    ReviewComplete,
    ReviewReport,
    ReviewerStarted,
    ReviewStarted,
    render_report_markdown,
    run_review,
)
from src.review import Finding

__all__ = [
    "format_findings_card",
    "format_footer_message",
    "format_merged_card",
    "format_refusal_message",
    "format_status",
    "on_chat_start",
    "on_message",
    "parse_desk_directive",
    "resolve_context",
]

# --- Desk defaults (first diff of a session; later diffs reuse them, FR-12) ---

DEFAULT_REPO = "pasted-diff"
DEFAULT_LANGUAGE = "python"
DEFAULT_RULESET = "default"
DEFAULT_STRICTNESS = "normal"

# --- Severity badges (Task 6 styling contract) ---

SEVERITY_BADGES: dict[str, str] = {
    "critical": "CRITICAL",
    "major": "MAJOR",
    "minor": "MINOR",
}

# --- Session-state keys (FR-12) ---

SESSION_CONTEXT = "context"
SESSION_LAST_REPORT = "last_report"
SESSION_REVIEW_COUNT = "review_count"
SESSION_LOCK = "review_lock"


# --- Pure formatting helpers (unit-tested without the browser) ---


def severity_badge(severity: str) -> str:
    """The uppercase severity label; unknown severities degrade gracefully."""
    return SEVERITY_BADGES.get(severity, severity.upper())


def severity_counts(findings: list[Finding]) -> dict[str, int]:
    """Findings per severity: ``{"critical": n, "major": n, "minor": n, ...}``."""
    counts: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def format_findings_card(
    reviewer: str,
    findings: list[Finding],
    *,
    partial: bool = False,
    partial_reason: str | None = None,
    latency_ms: float | None = None,
) -> str:
    """One severity-styled markdown card for a reviewer's landed findings.

    Severity labels are plain uppercase text; the CSS theme in
    ``public/styles.css`` styles the surrounding message card generically.
    A partial review is flagged in the header and its reason footnoted.
    """
    counts = severity_counts(findings)
    state = "partial review" if partial else "findings"
    header = (
        f"**{reviewer} — {state}: {len(findings)} finding(s) "
        f"({counts['critical']} critical / {counts['major']} major / "
        f"{counts['minor']} minor)"
    )
    if latency_ms is not None:
        header += f" · {latency_ms:.0f} ms"
    header += "**"
    lines = [header]
    if not findings:
        lines.append("")
        lines.append("_No findings from this reviewer._")
    for finding in findings:
        lines.append(
            f"- **{severity_badge(finding.severity)}** "
            f"`{finding.file}:{finding.line}` — {finding.message}"
        )
    if partial and partial_reason:
        lines.append("")
        lines.append(f"> Partial: {partial_reason}")
    return "\n".join(lines)


def format_merged_card(findings: list[Finding]) -> str:
    """The deduplicated, severity-ordered merged findings as one card."""
    counts = severity_counts(findings)
    lines = [
        f"## Merged findings — {len(findings)} unique",
        "",
        f"_{counts['critical']} critical · {counts['major']} major · "
        f"{counts['minor']} minor_",
        "",
    ]
    if not findings:
        lines.append("No findings — the diff looks clean to the desk.")
    for finding in findings:
        sources = getattr(finding, "sources", None)
        suffix = f" *(sources: {', '.join(sources)})*" if sources else ""
        lines.append(
            f"- **{severity_badge(finding.severity)}** "
            f"`{finding.file}:{finding.line}` — {finding.message}{suffix}"
        )
    return "\n".join(lines)


def format_refusal_message(reason: str, masked: str | None) -> str:
    """The visible FR-8 refusal: nothing is echoed, nothing secret leaks.

    The guardrail masked everything it put into ``reason``/``masked`` before
    this renderer ever saw it; the rendered text interpolates EXACTLY those
    two strings and nothing else, so no raw diff or credential material can
    enter the chat from here.
    """
    lines = [
        "**The report was refused by the secret guardrail.**",
        "",
        reason,
    ]
    if masked:
        lines.extend(["", f"Matched credential patterns (masked): `{masked}`"])
    lines.extend(
        [
            "",
            "Nothing is echoed — remove the credential from the diff and paste it again.",
        ]
    )
    return "\n".join(lines)


def format_footer_message(report: ReviewReport) -> str:
    """The FR-10 measurements footer as a chat message body."""
    return f"## Measurements\n\n{report.footer}"


def format_status(event: object, state: dict | None = None) -> str | None:
    """The progressive status line for one pipeline event (FR-12).

    Returns ``None`` when the event should not touch the status message.
    ``state`` is a dict the caller keeps for the whole review: the
    ``ReviewerStarted`` branch counts launched reviewers in it so the desk
    can say "3 reviewers running…", and the ``ReviewStarted`` branch reads
    the optional ``review_number`` key to say "Review #N for repo …".
    """
    if isinstance(event, ReviewStarted):
        body = (
            f"Splitting diff… {event.n_chunks} file chunk(s) · "
            f"repo `{event.context.repo}` · {event.context.strictness} mode."
        )
        if state is not None and state.get("review_number"):
            return f"Review #{state['review_number']} for repo `{event.context.repo}` — {body}"
        return body
    if isinstance(event, ReviewerStarted):
        running = 0
        if state is not None:
            state["reviewers_started"] = state.get("reviewers_started", 0) + 1
            running = state["reviewers_started"]
        return f"{event.reviewer} started — {running} reviewer(s) running…"
    if isinstance(event, FindingsLanded):
        counts = severity_counts(event.findings)
        partial = " · PARTIAL" if event.partial else ""
        return (
            f"{event.reviewer} landed: {len(event.findings)} finding(s) "
            f"({counts['critical']} critical) in {event.latency_ms or 0:.0f} ms"
            f"{partial}"
        )
    if isinstance(event, PartialReview):
        return f"Partial review — {event.reason}"
    if isinstance(event, MergedReport):
        counts = severity_counts(event.findings)
        return (
            f"Merging… {len(event.findings)} unique finding(s) "
            f"({counts['critical']} critical)."
        )
    if isinstance(event, RemediationOffered):
        if event.escalation is not None:
            return "A critical security finding was escalated — writing the remediation proposal…"
        return "No remediation needed — no critical security finding required a patch."
    if isinstance(event, GuardrailRefused):
        return "Refused — the secret guardrail stopped the report; nothing is echoed."
    if isinstance(event, ReviewComplete):
        return "Review complete — measurements below."
    return None


def format_agent_board(state: dict) -> str | None:
    """The live agent board: one line per reviewer, updated in place.

    This is the "watch the agents work" surface: each reviewer's line moves
    from *running* to its landed state (findings, latency) or *partial*.
    Returns ``None`` until the first reviewer is launched, so the board
    message is only created when there is something to show.
    """
    agents = state.get("agents") or {}
    if not agents:
        return None
    lines = ["**Agents**", ""]
    for name, line in agents.items():
        lines.append(f"- **{name}** — {line}")
    return "\n".join(lines)


def _agent_line(event: object) -> str:
    """The board line for a landed reviewer (kept in sync with its card)."""
    counts = severity_counts(event.findings)
    landed_state = "partial" if getattr(event, "partial", False) else "done"
    base = (
        f"{landed_state}: {len(event.findings)} finding(s) "
        f"({counts['critical']} critical)"
    )
    latency = getattr(event, "latency_ms", None)
    if latency is not None:
        base += f" · {latency:.0f} ms"
    return base


# --- Session-state helpers (FR-12) ---


def resolve_context(
    session_context: ReviewContext | None, overrides: dict[str, str]
) -> ReviewContext:
    """The context for THIS diff, honouring session-state reuse (FR-12).

    - No stored context: build one from the desk defaults plus any
      ``desk:`` directive overrides (first diff of the session).
    - Stored context without overrides: REUSE it untouched — repo, language,
      ruleset and strictness stay from the first review of the session.
    - Stored context with explicit overrides: an updated copy — the user
      explicitly asked to change a setting for this diff.
    """
    if session_context is None:
        values: dict[str, str] = {
            "repo": DEFAULT_REPO,
            "language": DEFAULT_LANGUAGE,
            "ruleset_id": DEFAULT_RULESET,
            "strictness": DEFAULT_STRICTNESS,
        }
        values.update(overrides)
        return ReviewContext(**values)
    if not overrides:
        return session_context
    values = {
        "repo": session_context.repo,
        "language": session_context.language,
        "ruleset_id": session_context.ruleset_id,
        "strictness": session_context.strictness,
    }
    values.update(overrides)
    return ReviewContext(**values)


_DIRECTIVE_PREFIX = "desk:"


def parse_desk_directive(text: str) -> tuple[str, dict[str, str]]:
    """Split an optional first-line ``desk:`` directive off the pasted text.

    ``desk: repo=foo lang=python ruleset=strict-mode strict=strict`` on the
    FIRST line only; any other first line means the whole message is the
    diff, so a plain diff can never break (the directive is opt-in).
    Returns ``(diff_text, overrides)``; ``overrides`` holds only the
    recognised keys (``repo``, ``language``, ``ruleset_id``, ``strictness``).
    """
    first_line, _, rest = text.partition("\n")
    if not first_line.strip().lower().startswith(_DIRECTIVE_PREFIX):
        return text, {}
    overrides: dict[str, str] = {}
    for token in first_line.strip()[len(_DIRECTIVE_PREFIX):].split():
        key, sep, value = token.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not sep or not value:
            continue
        if key == "repo":
            overrides["repo"] = value
        elif key in ("lang", "language"):
            overrides["language"] = value
        elif key in ("ruleset", "ruleset_id"):
            overrides["ruleset_id"] = value
        elif key in ("strict", "strictness"):
            if value.lower() in ("normal", "strict"):
                overrides["strictness"] = value.lower()
    return rest, overrides


# --- Welcome text ---

WELCOME = """**ReviewDesk** — paste a diff, get a structured review.

Three agents review your change in parallel: **security**, **tests** and **style**. Their findings stream in as each agent finishes, then get merged and severity-ordered, with per-agent latency and token usage at the end.

**How to use**

1. Run `git diff` and paste the whole output (it starts with a `diff --git` line).
2. Watch each agent report, then read the merged report.

Settings from your first diff (repo, language, ruleset, strictness) are reused for the session. To change them, put a directive on the first line:

```
desk: repo=my-repo lang=python ruleset=default strict=strict
diff --git a/app.py b/app.py
...
```

A diff containing a credential is refused rather than echoed."""


# --- Chainlit handlers ---


@cl.on_chat_start
async def on_chat_start() -> None:
    """Seed the session state and greet the user (FR-12).

    ``setup_tracing`` never raises and never echoes the key; its sentence (if
    any) goes to the SERVER LOG only — never to the chat.
    """
    tracing_note = setup_tracing()
    if tracing_note:
        print(f"[code-review-desk] {tracing_note}")
    cl.user_session.set(SESSION_REVIEW_COUNT, 0)
    cl.user_session.set(SESSION_CONTEXT, None)
    cl.user_session.set(SESSION_LAST_REPORT, None)
    cl.user_session.set(SESSION_LOCK, asyncio.Lock())
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print(
            "[code-review-desk] GEMINI_API_KEY is not set — reviews will fail "
            "until it is added to the server environment (.env, never committed)."
        )
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """One chat turn: paste a diff → progressive findings → footer (FR-12).

    The whole body is guarded so ANY exception becomes ONE friendly sentence
    in the chat (NFR-4) — the real error goes to the server log instead.
    """
    try:
        await _handle_message(message)
    except Exception as exc:  # noqa: BLE001 - never a traceback in the chat
        print(
            f"[code-review-desk] review failed: {type(exc).__name__}: {exc}"
        )
        await cl.Message(
            content=(
                "The review could not be completed "
                f"({type(exc).__name__}) — please try again. If it keeps "
                "failing, check the server log for the cause."
            )
        ).send()


async def _handle_message(message: cl.Message) -> None:
    """The review flow for one pasted message, behind the session lock.

    The pipeline generator is driven with a single ``async for`` to
    EXHAUSTION on the success path (never ``break``, never abandoned); the
    ``finally`` closes it deterministically on EVERY path — DiffError, a
    loop-body raise, or normal exhaustion — so its contextvar resets always
    run at the end of this handler, never "eventually" at asyncgen GC.
    """
    text = message.content or ""
    if not text.strip():
        await cl.Message(
            content=(
                "That message is empty — paste a unified diff (start it with "
                "a `diff --git` line; `git diff` produces one) and I'll review it."
            )
        ).send()
        return

    # Serialize reviews per session (session-state race protection). If two
    # messages race before on_chat_start's seed task ran, the FIRST one to
    # get here creates the lock AND stores it — a lock created without being
    # stored would not serialize anything.
    lock: asyncio.Lock = cl.user_session.get(SESSION_LOCK)
    if lock is None:
        lock = asyncio.Lock()
        cl.user_session.set(SESSION_LOCK, lock)
    async with lock:
        diff_text, overrides = parse_desk_directive(text)
        if not diff_text.strip():
            await cl.Message(
                content=(
                    "I see the `desk:` settings directive, but no diff below "
                    "it — put the unified diff on the lines after the directive."
                )
            ).send()
            return

        # FR-12 session state: first diff creates the context, later diffs
        # reuse it (explicit directive overrides are honoured).
        ctx = resolve_context(cl.user_session.get(SESSION_CONTEXT), overrides)
        cl.user_session.set(SESSION_CONTEXT, ctx)
        review_count = (cl.user_session.get(SESSION_REVIEW_COUNT) or 0) + 1
        cl.user_session.set(SESSION_REVIEW_COUNT, review_count)

        status = cl.Message(content="Starting review…")
        await status.send()
        state: dict = {"reviewers_started": 0, "review_number": review_count}
        gen = run_review(diff_text, ctx)
        try:
            # ONE async-for, driven to exhaustion on the success path — always.
            async for event in gen:
                await _handle_event(event, status, state)
        except DiffError as exc:
            status.content = "Stopped — the diff needs a fix before a review can run."
            await status.update()
            await cl.Message(content=exc.message).send()
            return
        finally:
            # The drain contract, made deterministic: a generator abandoned
            # SUSPENDED (the loop body raised — e.g. a card send failed on a
            # client disconnect) would otherwise wait for asyncgen GC
            # finalization before its finally could reset the planted-secret
            # contextvars. aclose() is a no-op on an exhausted/closed
            # generator and closes a suspended one NOW, so the resets always
            # run at the end of this handler, never "eventually".
            await gen.aclose()

        # A fully drained stream always ended with ReviewComplete, so the
        # report is stored; if one somehow ended without it, say so instead
        # of going silent (the DiffError path returned above).
        if cl.user_session.get(SESSION_LAST_REPORT) is None:
            status.content = "The review ended without a report — please try again."
            await status.update()


async def _handle_event(event: object, status: cl.Message, state: dict) -> None:
    """Render one pipeline event into the chat (progressive, FR-12)."""
    status_text = format_status(event, state)
    if status_text is not None:
        status.content = status_text
        await status.update()

    # Live agent board: update the agent's line FIRST, then render, so the
    # board shows this event's effect immediately (never one event behind).
    if isinstance(event, ReviewerStarted):
        state.setdefault("agents", {})[event.reviewer] = "running"
    elif isinstance(event, FindingsLanded):
        state.setdefault("agents", {})[event.reviewer] = _agent_line(event)

    board_text = format_agent_board(state)
    if board_text is not None:
        board: cl.Message | None = state.get("board")
        if board is None:
            board = cl.Message(content=board_text)
            state["board"] = board
            await board.send()
        else:
            board.content = board_text
            await board.update()

    if isinstance(event, FindingsLanded):
        await cl.Message(
            content=format_findings_card(
                event.reviewer,
                event.findings,
                partial=event.partial,
                partial_reason=event.partial_reason,
                latency_ms=event.latency_ms,
            )
        ).send()
    elif isinstance(event, PartialReview):
        # The reason is already in the status and the card footnote.
        pass
    elif isinstance(event, MergedReport):
        await cl.Message(content=format_merged_card(event.findings)).send()
    elif isinstance(event, RemediationOffered):
        if event.escalation is not None:
            await cl.Message(
                content=f"## Remediation proposal\n\n{event.text}"
            ).send()
    elif isinstance(event, GuardrailRefused):
        await cl.Message(
            content=format_refusal_message(event.reason, event.masked)
        ).send()
    elif isinstance(event, ReviewComplete):
        report = event.report
        cl.user_session.set(SESSION_LAST_REPORT, report)
        full_report = cl.Text(
            content=render_report_markdown(report),
            name=f"review-{report.request_id}.md",
            display="inline",
        )
        await cl.Message(
            content=format_footer_message(report),
            elements=[full_report],
        ).send()
