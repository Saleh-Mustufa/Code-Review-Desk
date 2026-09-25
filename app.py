"""Code Review Desk — Chainlit UI entry (FR-12, AD-10).

One chat turn is one review: paste a unified diff and the desk streams
progressive status updates plus a severity-styled findings card per reviewer
as it lands, then the merged report, the optional remediation proposal, the
FR-8 refusal when the secret guardrail trips, and the FR-10 measurements
footer (per-reviewer latency and REAL token usage) with the full report
attached as a text element.

Session state (``cl.user_session``) carries the review context, the review
count, the last report and the pending settings-panel overrides (FR-12): a
second diff pasted in the same session REUSES the first review's
repo/language/ruleset/strictness unless overridden. Override precedence,
most to least explicit (per key): a ``desk:`` directive typed on THIS
message > the ChatSettings panel (until the next review absorbs it) > the
stored context from the first review > the desk defaults.

House rules honoured here (NFR-4): ANY exception becomes ONE friendly
sentence in the chat — never a traceback; keys and server internals stay in
the server log. The pipeline's async generator is always closed
deterministically (``finally: await gen.aclose()``): its ``finally`` resets
the planted-secret contextvars whether the stream drained, raised, or the
loop body failed mid-review.

Presentation is pure markdown styled by ``public/styles.css`` (a dark design
system): severity pills and file chips are inline-code chips — a
``**`pill`**`` renders the CRITICAL red pill, a bare ``code`` chip the MAJOR
amber chip, an ``*`chip`*`` the MINOR sky pill (see the CSS comments).
"""

from __future__ import annotations

import asyncio
import os

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput

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
    "format_review_header",
    "format_settings_confirmation",
    "format_status",
    "normalize_settings",
    "on_chat_start",
    "on_message",
    "on_settings_update",
    "parse_desk_directive",
    "resolve_context",
    "severity_badge",
    "severity_counts",
]

# --- Desk defaults (first diff of a session; later diffs reuse them, FR-12) ---

DEFAULT_REPO = "pasted-diff"
DEFAULT_LANGUAGE = "python"
DEFAULT_RULESET = "default"
DEFAULT_STRICTNESS = "normal"

KNOWN_RULESETS = ["default", "strict"]
"""Ruleset ids offered in the settings panel (files under data/rulesets/)."""

# --- Severity badges (rendered as inline-code pills; see module docstring) ---

SEVERITY_BADGES: dict[str, str] = {
    "critical": "🔴 CRITICAL",
    "major": "🟠 MAJOR",
    "minor": "🔵 MINOR",
}

# --- Session-state keys (FR-12) ---

SESSION_CONTEXT = "context"
SESSION_LAST_REPORT = "last_report"
SESSION_REVIEW_COUNT = "review_count"
SESSION_LOCK = "review_lock"
SESSION_SETTINGS_OVERRIDES = "settings_overrides"


# --- Pure formatting helpers (unit-tested without the browser) ---


def severity_badge(severity: str) -> str:
    """The emoji badge label for one severity; unknown severities degrade."""
    return SEVERITY_BADGES.get(severity, severity.upper())


def severity_pill(severity: str) -> str:
    """A severity label as an inline-code pill, wrapped by severity.

    The markdown wrapper is the CSS hook (no raw HTML allowed):
    ``**`pill`**`` → CRITICAL red pill, bare ``code`` → MAJOR amber pill,
    ``*`pill`*`` → MINOR sky pill (public/styles.css).
    """
    label = severity_badge(severity)
    if severity == "critical":
        return f"**`{label}`**"
    if severity == "minor":
        return f"*`{label}`*"
    return f"`{label}`"


def severity_counts(findings: list[Finding]) -> dict[str, int]:
    """Findings per severity: ``{"critical": n, "major": n, "minor": n, ...}``."""
    counts: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def counts_summary(counts: dict[str, int]) -> str:
    """A compact '1 critical · 2 major · 0 minor' summary line."""
    return (
        f"{counts.get('critical', 0)} critical · "
        f"{counts.get('major', 0)} major · "
        f"{counts.get('minor', 0)} minor"
    )


def format_finding_line(finding: Finding) -> str:
    """One finding as a styled list row: pill, file chip, message, sources."""
    sources = getattr(finding, "sources", None)
    via = f" *(via {', '.join(sources)})*" if sources else ""
    return (
        f"{severity_pill(finding.severity)} · *`{finding.file}:{finding.line}`* "
        f"— {finding.message}{via}"
    )


def format_findings_card(
    reviewer: str,
    findings: list[Finding],
    *,
    partial: bool = False,
    partial_reason: str | None = None,
    latency_ms: float | None = None,
) -> str:
    """One severity-styled markdown card for a reviewer's landed findings.

    A reviewer header with per-severity counts, one styled row per finding,
    and a warning blockquote footnote for turn-ceiling partial reviews.
    """
    counts = severity_counts(findings)
    state = "partial review" if partial else "findings"
    lines = [
        f"### 🧐 {reviewer} — {state}",
        "",
        f"**{len(findings)}** finding(s) · {counts_summary(counts)}",
    ]
    if latency_ms is not None:
        lines[2] += f" · ⏱ {latency_ms:.0f} ms"
    lines.append("")
    if not findings:
        lines.append("_Nothing to flag from this reviewer._")
        lines.append("")
    for finding in findings:
        lines.append(f"- {format_finding_line(finding)}")
    if partial and partial_reason:
        lines.append("")
        lines.append(f"> ⚠️ Partial — {partial_reason}")
    return "\n".join(lines).rstrip()


def format_merged_card(findings: list[Finding]) -> str:
    """The deduplicated, severity-ordered merged findings as one card."""
    counts = severity_counts(findings)
    lines = [
        "## 🧾 Merged report",
        "",
        f"**{len(findings)}** unique finding(s) after dedupe — "
        f"{counts_summary(counts)}",
        "",
    ]
    if not findings:
        lines.append("No findings — the diff looks clean to the desk. 🎉")
    for position, finding in enumerate(findings, start=1):
        lines.append(f"{position}. {format_finding_line(finding)}")
    return "\n".join(lines).rstrip()


def format_review_header(event: ReviewStarted, review_number: int) -> str:
    """The review header card: settings strip for THIS review (FR-2)."""
    ctx = event.context
    lines = [
        f"## 🎛 Review #{review_number} — settings",
        "",
        (
            f"**Repo** *`{ctx.repo}`* · **Language** *`{ctx.language}`* "
            f"· **Ruleset** *`{ctx.ruleset_id}`* · **Strictness** *`{ctx.strictness}`*"
        ),
        "",
        f"Reviewing **{event.n_chunks}** file chunk(s) as request *`{event.request_id}`*.",
    ]
    return "\n".join(lines)


def format_refusal_message(reason: str, masked: str | None) -> str:
    """The visible FR-8 refusal: nothing is echoed, nothing secret leaks.

    The guardrail masked everything it put into ``reason``/``masked`` before
    this renderer ever saw it; the rendered text interpolates EXACTLY those
    two strings and nothing else, so no raw diff or credential material can
    enter the chat from here.
    """
    lines = [
        "## 🚫 Report refused",
        "",
        "**The secret guardrail stopped this report.**",
        "",
        reason,
    ]
    if masked:
        lines.extend(["", f"Matched credential patterns (masked): *`{masked}`*"])
    lines.extend(
        [
            "",
            "Nothing is echoed — remove the credential from the diff and paste it again.",
        ]
    )
    return "\n".join(lines)


def format_footer_message(report: ReviewReport) -> str:
    """The FR-10 measurements footer: table, wall clock, attachment hint."""
    duration = report.duration_ms / 1000.0
    return (
        "## ⏱ Measurements\n\n"
        f"{report.footer}\n\n"
        f"Wall clock: **{duration:.1f} s** · "
        "_the full report is attached to this message — expand it for the "
        "complete write-up._"
    )


def format_settings_confirmation(overrides: dict[str, str]) -> str:
    """One-line confirmation after a settings-panel update."""
    if not overrides:
        return "🎛 Settings saved — no changes detected, the desk keeps the current setup."
    parts = [
        f"repo *`{overrides['repo']}`*" if "repo" in overrides else None,
        f"language *`{overrides['language']}`*" if "language" in overrides else None,
        f"ruleset *`{overrides['ruleset_id']}`*" if "ruleset_id" in overrides else None,
        f"strictness *`{overrides['strictness']}`*" if "strictness" in overrides else None,
    ]
    return (
        "🎛 Settings saved — the next review runs with "
        + " · ".join(part for part in parts if part)
        + "."
    )


def format_status(event: object, state: dict | None = None) -> str | None:
    """The progressive status line for one pipeline event (FR-12).

    Returns ``None`` when the event should not touch the status message.
    ``state`` is a dict the caller keeps for the whole review: the
    ``ReviewerStarted`` branch counts launched reviewers in it so the desk
    can say "3 reviewers running…", and the ``ReviewStarted`` branch reads
    the optional ``review_number`` key to say "Review #N for repo …".
    Status glyphs: ⏳ running → ✅ done → ⚠️ partial → 🚫 refused → 🛠 remediation.
    """
    if isinstance(event, ReviewStarted):
        body = (
            f"Splitting diff… {event.n_chunks} file chunk(s) · "
            f"repo `{event.context.repo}` · {event.context.strictness} mode."
        )
        if state is not None and state.get("review_number"):
            return f"⏳ Review #{state['review_number']} for repo `{event.context.repo}` — {body}"
        return f"⏳ {body}"
    if isinstance(event, ReviewerStarted):
        running = 0
        if state is not None:
            state["reviewers_started"] = state.get("reviewers_started", 0) + 1
            running = state["reviewers_started"]
        return f"⏳ {event.reviewer} started — {running} reviewer(s) running…"
    if isinstance(event, FindingsLanded):
        counts = severity_counts(event.findings)
        glyph = "⚠️" if event.partial else "✅"
        partial = " · PARTIAL" if event.partial else ""
        return (
            f"{glyph} {event.reviewer} landed: {len(event.findings)} finding(s) "
            f"({counts['critical']} critical) in {event.latency_ms or 0:.0f} ms"
            f"{partial}"
        )
    if isinstance(event, PartialReview):
        return f"⚠️ Partial review — {event.reason}"
    if isinstance(event, MergedReport):
        counts = severity_counts(event.findings)
        return (
            f"🧮 Merging… {len(event.findings)} unique finding(s) "
            f"({counts['critical']} critical)."
        )
    if isinstance(event, RemediationOffered):
        if event.escalation is not None:
            return "🛠 A critical security finding was escalated — writing the remediation proposal…"
        return "✅ No remediation needed — no critical security finding required a patch."
    if isinstance(event, GuardrailRefused):
        return "🚫 Refused — the secret guardrail stopped the report; nothing is echoed."
    if isinstance(event, ReviewComplete):
        return "✅ Review complete — measurements below."
    return None


# --- Session-state helpers (FR-12) ---


def resolve_context(
    session_context: ReviewContext | None, overrides: dict[str, str]
) -> ReviewContext:
    """The context for THIS diff, honouring session-state reuse (FR-12).

    - No stored context: build one from the desk defaults plus any
      overrides (first diff of the session).
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


def normalize_settings(settings: dict) -> dict[str, str]:
    """Settings-panel values → the same override shape as a ``desk:`` directive.

    Recognised ids: ``repo``, ``language``, ``ruleset_id`` (non-empty
    strings) and ``strict_mode`` (a Switch boolean mapped to the
    ``strictness`` string). Unknown keys and blank values are ignored, so a
    partial update only changes what the user actually set.
    """
    overrides: dict[str, str] = {}
    for key in ("repo", "language", "ruleset_id"):
        value = str(settings.get(key, "") or "").strip()
        if value:
            overrides[key] = value
    strict_mode = settings.get("strict_mode")
    if isinstance(strict_mode, bool):
        overrides["strictness"] = "strict" if strict_mode else "normal"
    return overrides


# --- Settings panel (cl.ChatSettings) ---


def build_settings_inputs() -> list:
    """The settings-panel inputs, initialised from the desk defaults."""
    return [
        TextInput(
            id="repo",
            label="Repo name",
            initial=DEFAULT_REPO,
            description="Label used in the report header (informational).",
        ),
        TextInput(
            id="language",
            label="Language",
            initial=DEFAULT_LANGUAGE,
            description="Primary language of the diff under review.",
        ),
        Select(
            id="ruleset_id",
            label="Ruleset",
            values=list(KNOWN_RULESETS),
            # Select.__post_init__ OVERWRITES `initial` with `initial_value`
            # when `values` is given — passing initial= would be ignored.
            initial_value=DEFAULT_RULESET,
            description="Which ruleset file (data/rulesets/) the reviewers load.",
        ),
        Switch(
            id="strict_mode",
            label="Strict mode",
            initial=DEFAULT_STRICTNESS == "strict",
            description="Terser, harsher reviewer prompts (strictness).",
        ),
    ]


# --- Welcome text ---

WELCOME = """# 🪑 Code Review Desk

*Professional code review for pasted diffs — three concurrent reviewers,
one merged report, real measurements.*

**How to use**

1. Paste a **unified diff** straight into the chat (run `git diff`, copy the
   whole output including the `diff --git` lines, paste).
2. Watch each reviewer's findings land live — severity pills, file chips,
   sources — followed by the merged report and the measurements footer.
3. A **settings panel** (gear icon) lets you set repo, language, ruleset and
   strict mode; a second diff in the same session **reuses those settings**.

**Optional inline override** — prefix the FIRST line of a message with a
directive (it beats the settings panel for that one review):

```
desk: repo=my-repo lang=python ruleset=strict strict=strict
diff --git a/app.py b/app.py
...
```

Every failure is one friendly sentence — a malformed diff never crashes the
desk, and if a credential shape appears in the output, the secret guardrail
refuses the report instead of echoing it. 🛡️"""


# --- Chainlit handlers ---


async def _send_settings_panel() -> None:
    """Push the settings panel to the UI with the desk defaults."""
    panel = cl.ChatSettings(inputs=build_settings_inputs())
    await panel.send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """Seed the session state, greet the user, offer the settings panel (FR-12).

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
    cl.user_session.set(SESSION_SETTINGS_OVERRIDES, {})
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print(
            "[code-review-desk] GEMINI_API_KEY is not set — reviews will fail "
            "until it is added to the server environment (.env, never committed)."
        )
    await cl.Message(content=WELCOME).send()
    await _send_settings_panel()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """Store settings-panel values as pending overrides for the next review.

    Precedence, most to least explicit, per key: a ``desk:`` directive typed
    on a message > these pending overrides > the stored session context >
    the desk defaults. FR-12 reuse is untouched: with no overrides pending,
    a second diff reuses the first review's context.
    """
    overrides = normalize_settings(settings)
    cl.user_session.set(SESSION_SETTINGS_OVERRIDES, overrides)
    await cl.Message(content=format_settings_confirmation(overrides)).send()


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
        diff_text, directive_overrides = parse_desk_directive(text)
        if not diff_text.strip():
            await cl.Message(
                content=(
                    "I see the `desk:` settings directive, but no diff below "
                    "it — put the unified diff on the lines after the directive."
                )
            ).send()
            return

        # FR-12 override precedence, per key: a desk: directive on THIS
        # message wins over the pending settings-panel overrides; both beat
        # the stored context, which beats the desk defaults.
        pending = cl.user_session.get(SESSION_SETTINGS_OVERRIDES) or {}
        used_pending = bool(pending)
        overrides = {**pending, **directive_overrides}

        # FR-12 session state: first diff creates the context, later diffs
        # reuse it (explicit overrides are honoured).
        ctx = resolve_context(cl.user_session.get(SESSION_CONTEXT), overrides)
        cl.user_session.set(SESSION_CONTEXT, ctx)
        if used_pending:
            # The panel overrides are now absorbed into the stored context;
            # later diffs take the pure FR-12 reuse path again until the
            # panel is changed.
            cl.user_session.set(SESSION_SETTINGS_OVERRIDES, {})
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
            status.content = "⚠️ Stopped — the diff needs a fix before a review can run."
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
            status.content = "⚠️ The review ended without a report — please try again."
            await status.update()


async def _handle_event(event: object, status: cl.Message, state: dict) -> None:
    """Render one pipeline event into the chat (progressive, FR-12)."""
    status_text = format_status(event, state)
    if status_text is not None:
        status.content = status_text
        await status.update()

    if isinstance(event, ReviewStarted):
        review_number = state.get("review_number", 1)
        await cl.Message(content=format_review_header(event, review_number)).send()
    elif isinstance(event, FindingsLanded):
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
                content=f"## 🛠 Remediation proposal\n\n{event.text}"
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
