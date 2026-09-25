"""Unit tests for app.py — the Chainlit UI logic, WITHOUT the browser.

Coverage (Task 6):

- pure formatting: severity badges, findings cards (counts, partial flag,
  latency), merged card, status lines, refusal text, footer message;
- session-state helpers: the optional ``desk:`` directive parser and the
  FR-12 context-reuse rules;
- handler wiring WITHOUT a live server: ``import app`` registers the
  Chainlit handlers, and ``_handle_message`` is driven against a fake
  ``cl`` module and a canned pipeline event stream (monkeypatched
  ``app.run_review``), proving progressive messages, session reuse, the
  friendly DiffError path, and that a crashed stream still CLOSES its
  generator (the drain contract) while the user sees ONE friendly sentence.

No live chainlit server is ever started; the browser pass is Task 7's
black-box gate.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from chainlit.config import config as chainlit_config

import app as app_module
from src.intake import DiffError, ReviewContext
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
    render_footer,
)
from src.review import Finding

# --- Fixtures and fakes --------------------------------------------------------

THE_SECRET = 'API_KEY = "gsk_aaaaaaaaaabbbbbbbbbb"'
"""A raw credential shape; renderers must never echo it."""

MASKED_PATTERN = "gsk_****aaaa****"
"""What the guardrail hands downstream: masked, never the raw secret."""

VALID_DIFF = (
    "diff --git a/app.py b/app.py\n"
    "--- a/app.py\n"
    "+++ b/app.py\n"
    "@@ -1,3 +1,5 @@\n"
    "+import os\n"
    " def main():\n"
    "     pass\n"
)


def make_finding(
    severity: str = "major", file: str = "app.py", line: int = 3, message: str = "Watch out."
) -> Finding:
    return Finding(file=file, line=line, severity=severity, message=message)  # type: ignore[arg-type]


class FakeMessage:
    """A cl.Message stand-in: records sends, updates and attached elements."""

    sent: list["FakeMessage"] = []

    def __init__(self, content: str = "", elements: list | None = None, **kwargs):
        self.content = content
        self.elements = list(elements or [])
        self.send_count = 0
        self.update_count = 0
        # Every content the message ever had (updates mutate in place).
        self.history: list[str] = [content]

    async def send(self):
        self.send_count += 1
        FakeMessage.sent.append(self)
        return self

    async def update(self):
        self.update_count += 1
        self.history.append(self.content)
        return True


class FakeUserSession:
    """A cl.user_session stand-in backed by a plain dict."""

    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def set(self, key: str, value) -> None:
        self.store[key] = value


class FakeText:
    """A cl.Text element stand-in."""

    def __init__(self, content: str = "", name: str = "", display: str = "inline", **kwargs):
        self.content = content
        self.name = name
        self.display = display


def install_fake_cl(monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, FakeUserSession]:
    """Swap app.cl for a recording fake; return (fake_cl, fake_session)."""
    FakeMessage.sent = []
    session = FakeUserSession()
    fake_cl = SimpleNamespace(Message=FakeMessage, Text=FakeText, user_session=session)
    monkeypatch.setattr(app_module, "cl", fake_cl)
    return fake_cl, session


class FakeStream:
    """A canned replacement for pipeline.run_review (event-driven).

    Yields the given events, then optionally raises. Records whether it was
    fully exhausted and whether its ``finally`` ran — the drain contract.
    """

    def __init__(self, events: list, error: Exception | None = None):
        self.events = list(events)
        self.error = error
        self.calls: list = []
        self.exhausted = False
        self.finally_ran = False

    def __call__(self, diff_text: str, ctx, **kwargs):
        self.calls.append((diff_text, ctx, kwargs))
        return self._gen()

    async def _gen(self):
        try:
            for event in self.events:
                yield event
            if self.error is not None:
                raise self.error
            self.exhausted = True
        finally:
            self.finally_ran = True


def make_report(
    request_id: str = "rev_deadbeef",
    findings: list | None = None,
    refused: bool = False,
) -> ReviewReport:
    """A final report; merged findings use MergedFinding, as the pipeline does."""
    from src.specialists import MergedFinding

    merged = findings if findings is not None else [make_finding()]
    merged = [
        f if isinstance(f, MergedFinding)
        else MergedFinding(
            file=f.file,
            line=f.line,
            severity=f.severity,
            message=f.message,
            sources=["TestsReviewer"],
        )
        for f in merged
    ]
    report = ReviewReport(
        request_id=request_id,
        repo="pasted-diff",
        language="python",
        ruleset_id="default",
        strictness="normal",
        findings=merged,
        per_reviewer=[],
        refused=refused,
        refusal_reason="Output quoted credential-shaped text." if refused else None,
    )
    report.footer = render_footer(report)
    return report


# --- Pure formatting: severity badges and findings cards ------------------------


def test_severity_badge_mapping():
    assert app_module.severity_badge("critical") == "🔴 CRITICAL"
    assert app_module.severity_badge("major") == "🟠 MAJOR"
    assert app_module.severity_badge("minor") == "🟡 MINOR"
    # Unknown severities degrade to the raw label, never crash.
    assert app_module.severity_badge("blocker") == "BLOCKER"


def test_findings_card_counts_and_badges():
    findings = [
        make_finding("critical", line=1),
        make_finding("major", line=2),
        make_finding("minor", line=3),
    ]
    card = app_module.format_findings_card("SecurityReviewer", findings)
    assert "SecurityReviewer" in card
    assert "3 finding(s)" in card
    assert "1 critical" in card and "1 major" in card and "1 minor" in card
    assert "🔴 CRITICAL" in card
    assert "🟠 MAJOR" in card
    assert "🟡 MINOR" in card
    assert "`app.py:1`" in card and "Watch out." in card
    assert "partial" not in card.lower()


def test_findings_card_empty():
    card = app_module.format_findings_card("TestsReviewer", [])
    assert "0 finding(s)" in card
    assert "No findings from this reviewer." in card


def test_findings_card_partial_flag_and_footnote():
    findings = [make_finding("critical")]
    card = app_module.format_findings_card(
        "QualityReviewer",
        findings,
        partial=True,
        partial_reason="The reviewer hit its turn ceiling of 4 turns.",
        latency_ms=1234.7,
    )
    assert "partial review" in card
    assert "1 finding(s)" in card
    assert "1235 ms" in card
    assert "The reviewer hit its turn ceiling of 4 turns." in card


def test_findings_card_latency_shown_when_present():
    card = app_module.format_findings_card("TestsReviewer", [], latency_ms=98.4)
    assert "98 ms" in card
    plain = app_module.format_findings_card("TestsReviewer", [])
    assert "ms" not in plain


# --- Pure formatting: merged card, refusal, footer ------------------------------


def test_merged_card_deduped_count_sources_and_order():
    from src.specialists import MergedFinding

    def make_merged(severity: str, file: str = "app.py", message: str = "Watch out.") -> MergedFinding:
        return MergedFinding(file=file, line=3, severity=severity, message=message)  # type: ignore[arg-type]

    merged = [
        make_merged("critical", message="RCE risk."),
        make_merged("critical", file="b.py", message="Second one."),
        make_merged("minor", message="Style nit."),
    ]
    merged[0].sources = ["SecurityReviewer", "QualityReviewer"]
    card = app_module.format_merged_card(merged)
    assert "3 unique" in card
    assert "2 critical · 0 major · 1 minor" in card
    assert card.index("🔴 CRITICAL") < card.index("🟡 MINOR")
    assert "sources: SecurityReviewer, QualityReviewer" in card
    assert "Style nit." in card


def test_merged_card_clean_diff():
    card = app_module.format_merged_card([])
    assert "0 unique" in card
    assert "No findings" in card


def test_refusal_message_masks_and_never_echoes_secrets():
    reason = (
        f"The desk output quoted credential-shaped text matching {MASKED_PATTERN}; "
        "it was refused rather than echoed."
    )
    text = app_module.format_refusal_message(reason, MASKED_PATTERN)
    assert "refused by the secret guardrail" in text
    assert reason in text
    assert MASKED_PATTERN in text
    assert "Nothing is echoed" in text
    # The renderer interpolates ONLY its (already masked) inputs: the raw
    # secret never appears, and nothing unmasked is added around them.
    assert THE_SECRET not in text


def test_refusal_message_without_matches():
    text = app_module.format_refusal_message("Output quoted credential-shaped text.", None)
    assert "Output quoted credential-shaped text." in text
    assert "Matched credential patterns" not in text


def test_footer_message_carries_measurements_table():
    report = make_report()
    text = app_module.format_footer_message(report)
    assert "Measurements" in text
    assert "| Reviewer | Latency (ms) | Tokens in | Tokens out |" in text
    assert "**Total**" in text


# --- Pure helpers: status lines -------------------------------------------------


def test_status_review_started():
    event = ReviewStarted(
        request_id="rev_x",
        context=ReviewContext(repo="demo", language="python", ruleset_id="default"),
        n_chunks=2,
    )
    status = app_module.format_status(event)
    assert status is not None
    assert "2 file chunk(s)" in status
    assert "`demo`" in status
    assert "normal mode" in status


def test_status_review_started_surfaces_review_number():
    event = ReviewStarted(
        request_id="rev_x",
        context=ReviewContext(repo="demo", language="python", ruleset_id="default"),
        n_chunks=1,
    )
    status = app_module.format_status(event, {"review_number": 2})
    assert status is not None
    assert "Review #2 for repo `demo`" in status
    assert "1 file chunk(s)" in status
    # Without the state dict the line stays graceful (no "#None").
    assert "#" not in app_module.format_status(event)


def test_status_reviewer_started_counts_running():
    state: dict = {}
    first = app_module.format_status(ReviewerStarted(reviewer="SecurityReviewer"), state)
    second = app_module.format_status(ReviewerStarted(reviewer="TestsReviewer"), state)
    third = app_module.format_status(ReviewerStarted(reviewer="QualityReviewer"), state)
    assert "SecurityReviewer" in first and "1 reviewer(s) running" in first
    assert "2 reviewer(s) running" in second
    assert "3 reviewer(s) running" in third


def test_status_findings_landed_with_critical_count_and_partial():
    event = FindingsLanded(
        reviewer="SecurityReviewer",
        findings=[make_finding("critical"), make_finding("major"), make_finding("minor")],
        partial=True,
        partial_reason="ceiling",
        latency_ms=250.0,
    )
    status = app_module.format_status(event)
    assert status is not None
    assert "SecurityReviewer landed: 3 finding(s) (1 critical)" in status
    assert "250 ms" in status
    assert "PARTIAL" in status


def test_status_merge_remediation_refusal_and_complete():
    from src.specialists import Escalation

    merged = app_module.format_status(
        MergedReport(findings=[make_finding("critical"), make_finding("minor")])
    )
    assert "2 unique finding(s) (1 critical)" in merged

    no_remediation = app_module.format_status(
        RemediationOffered(escalation=None, text="No remediation needed.")
    )
    assert "No remediation needed" in no_remediation

    escalated_event = RemediationOffered(
        escalation=Escalation(
            finding=make_finding("critical"),
            reviewer="SecurityReviewer",
            reason="Critical security finding needs a patch.",
        ),
        text="patch",
    )
    escalated = app_module.format_status(escalated_event)
    assert escalated is not None and "remediation" in escalated.lower()

    refused = app_module.format_status(GuardrailRefused(reason="r", masked=None))
    assert "Refused" in refused

    complete = app_module.format_status(ReviewComplete(report=make_report()))
    assert "Review complete" in complete


def test_status_unknown_event_is_none():
    assert app_module.format_status(object()) is None


# --- desk: directive parsing ----------------------------------------------------


def test_parse_plain_diff_has_no_directive():
    text = "desk chair\ndiff --git a/x b/x\n"  # does not start with 'desk:'
    diff_text, overrides = app_module.parse_desk_directive(VALID_DIFF)
    assert diff_text == VALID_DIFF
    assert overrides == {}
    diff_text, overrides = app_module.parse_desk_directive(text)
    assert diff_text == text  # kept whole; intake will reject it as malformed
    assert overrides == {}


def test_parse_directive_full_and_aliases():
    text = (
        "desk: repo=my-repo lang=go ruleset=strict-mode strict=strict\n"
        f"{VALID_DIFF}"
    )
    diff_text, overrides = app_module.parse_desk_directive(text)
    assert diff_text == VALID_DIFF
    assert overrides == {
        "repo": "my-repo",
        "language": "go",
        "ruleset_id": "strict-mode",
        "strictness": "strict",
    }


def test_parse_directive_ignores_malformed_tokens():
    text = "desk: lang= (not-a-pair) strict=loose ruleset=\n" + VALID_DIFF
    diff_text, overrides = app_module.parse_desk_directive(text)
    assert diff_text == VALID_DIFF
    assert overrides == {}  # 'strict=loose' is not a valid value; junk is dropped


def test_parse_directive_is_first_line_only():
    text = f"{VALID_DIFF}\ndesk: repo=late"
    diff_text, overrides = app_module.parse_desk_directive(text)
    assert overrides == {}


# --- FR-12 session-state context reuse ------------------------------------------


def test_resolve_context_defaults_on_first_diff():
    ctx = app_module.resolve_context(None, {})
    assert (ctx.repo, ctx.language, ctx.ruleset_id, ctx.strictness) == (
        "pasted-diff",
        "python",
        "default",
        "normal",
    )


def test_resolve_context_reuses_stored_context():
    stored = ReviewContext(repo="demo", language="go", ruleset_id="strict", strictness="strict")
    assert app_module.resolve_context(stored, {}) is stored  # identity: reused


def test_resolve_context_explicit_override_updates_copy():
    stored = ReviewContext(repo="demo", language="go", ruleset_id="strict")
    ctx = app_module.resolve_context(stored, {"repo": "other", "strictness": "strict"})
    assert ctx is not stored
    assert ctx.repo == "other"
    assert ctx.strictness == "strict"
    assert ctx.language == "go"  # untouched fields carry over
    assert stored.repo == "demo"  # the stored context is never mutated


# --- Handler wiring (import-level) -----------------------------------------------


def test_import_app_registers_chainlit_handlers():
    # `import app` at module top must not crash without a server, and the
    # decorators must have registered the handlers with chainlit.
    assert chainlit_config.code.on_chat_start is not None
    assert chainlit_config.code.on_message is not None


# --- Handler flow against a fake cl (no browser, no server) ----------------------


def _events_for_review(report: ReviewReport, findings_1: list, findings_2: list) -> list:
    ctx = ReviewContext(repo="pasted-diff", language="python", ruleset_id="default")
    return [
        ReviewStarted(request_id=report.request_id, context=ctx, n_chunks=1),
        ReviewerStarted(reviewer="SecurityReviewer"),
        ReviewerStarted(reviewer="TestsReviewer"),
        ReviewerStarted(reviewer="QualityReviewer"),
        FindingsLanded(
            reviewer="SecurityReviewer",
            findings=findings_1,
            partial=False,
            partial_reason=None,
            latency_ms=100.0,
        ),
        FindingsLanded(
            reviewer="TestsReviewer",
            findings=findings_2,
            partial=False,
            partial_reason=None,
            latency_ms=120.0,
        ),
        MergedReport(findings=list(report.findings)),
        RemediationOffered(escalation=None, text="No remediation needed."),
        ReviewComplete(report=report),
    ]


@pytest.mark.asyncio
async def test_handle_message_progressive_messages_and_session_state(monkeypatch):
    _, session = install_fake_cl(monkeypatch)
    findings_1 = [make_finding("critical"), make_finding("minor")]
    findings_2 = [make_finding("major")]
    report = make_report(findings=[make_finding("critical"), make_finding("minor"), make_finding("major")])
    stream = FakeStream(_events_for_review(report, findings_1, findings_2))
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))

    # The generator was driven to exhaustion (the drain contract).
    assert stream.exhausted and stream.finally_ran
    assert len(stream.calls) == 1
    diff_used, ctx_used, kwargs = stream.calls[0]
    assert diff_used == VALID_DIFF
    assert (ctx_used.repo, ctx_used.language, ctx_used.ruleset_id) == (
        "pasted-diff",
        "python",
        "default",
    )

    sent = FakeMessage.sent
    # ONE status message, updated in place as events arrive (progressive).
    assert sent[0].send_count == 1
    # One card per FindingsLanded, separate messages (not one lump).
    cards = [m for m in sent if "— findings:" in m.content]
    assert len(cards) == 2
    assert "SecurityReviewer" in cards[0].content
    assert "TestsReviewer" in cards[1].content
    assert "2 finding(s)" in cards[0].content and "1 finding(s)" in cards[1].content
    # Merged card, no remediation message (escalation is None), then the footer.
    assert any("Merged findings — 3 unique" in m.content for m in sent)
    assert not any("Remediation proposal" in m.content for m in sent)
    footer = sent[-1]
    assert "Measurements" in footer.content
    assert len(footer.elements) == 1
    assert report.request_id in footer.elements[0].name
    assert "Code Review Desk" in footer.elements[0].content  # report header inside the element
    # Status was updated along the way; its final content says complete.
    assert sent[0].update_count >= 4
    assert "Review complete" in sent[0].content
    # review_count is surfaced: the first status line said "Review #1".
    assert any("Review #1" in c for c in sent[0].history)
    # FR-12 session state: context stored, count incremented, report remembered.
    assert session.store["review_count"] == 1
    assert session.store["context"] is ctx_used
    assert session.store["last_report"] is report


@pytest.mark.asyncio
async def test_handle_message_reuses_context_on_second_diff(monkeypatch):
    _, session = install_fake_cl(monkeypatch)
    report = make_report()
    stream = FakeStream(_events_for_review(report, [], []))
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))
    first_ctx = stream.calls[0][1]
    assert session.store["review_count"] == 1

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))
    second_ctx = stream.calls[1][1]
    assert session.store["review_count"] == 2
    assert second_ctx is first_ctx  # FR-12: second diff reuses the first context
    assert first_ctx.repo == "pasted-diff"  # repo/language/ruleset stay


@pytest.mark.asyncio
async def test_handle_message_directive_on_second_diff_updates_context(monkeypatch):
    _, session = install_fake_cl(monkeypatch)
    stream = FakeStream(_events_for_review(make_report(), [], []))
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))
    message = f"desk: repo=real-repo strict=strict\n{VALID_DIFF}"
    await app_module._handle_message(SimpleNamespace(content=message))

    updated = stream.calls[1][1]
    assert (updated.repo, updated.strictness) == ("real-repo", "strict")
    assert updated.language == "python"  # untouched fields carry over


@pytest.mark.asyncio
async def test_handle_message_empty_input_never_runs_pipeline(monkeypatch):
    _, _ = install_fake_cl(monkeypatch)
    stream = FakeStream([])
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content="   \n  "))

    assert stream.calls == []  # nothing ran
    assert len(FakeMessage.sent) == 1
    assert "paste a unified diff" in FakeMessage.sent[0].content.lower()


@pytest.mark.asyncio
async def test_handle_message_directive_without_diff(monkeypatch):
    _, _ = install_fake_cl(monkeypatch)
    stream = FakeStream([])
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content="desk: repo=x strict=strict"))

    assert stream.calls == []
    assert len(FakeMessage.sent) == 1
    assert "directive" in FakeMessage.sent[0].content.lower()


@pytest.mark.asyncio
async def test_handle_message_diff_error_is_friendly(monkeypatch):
    _, session = install_fake_cl(monkeypatch)
    stream = FakeStream(
        [], error=DiffError("That text doesn't look like a unified diff — try again.")
    )
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content="not a diff at all"))

    assert stream.finally_ran  # the generator closed; contextvar resets ran
    sent = FakeMessage.sent
    assert any("unified diff — try again" in m.content for m in sent)
    assert not any("Traceback" in m.content for m in sent)
    assert session.store.get("last_report") is None


@pytest.mark.asyncio
async def test_handle_message_remediation_message_when_escalated(monkeypatch):
    install_fake_cl(monkeypatch)
    from src.specialists import Escalation

    escalation = Escalation(
        finding=make_finding("critical", file="auth.py", line=7, message="SQL injection."),
        reviewer="SecurityReviewer",
        reason="Critical security finding needs a patch.",
    )
    report = make_report()
    events = _events_for_review(report, [], [])
    events[-2] = RemediationOffered(escalation=escalation, text="Use parameterised queries.")
    stream = FakeStream(events)
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))

    remediation = [m for m in FakeMessage.sent if "Remediation proposal" in m.content]
    assert len(remediation) == 1
    assert "Use parameterised queries." in remediation[0].content


@pytest.mark.asyncio
async def test_handle_message_guardrail_refusal(monkeypatch):
    install_fake_cl(monkeypatch)
    report = make_report(findings=[], refused=True)
    events = [
        ReviewStarted(
            request_id=report.request_id,
            context=ReviewContext(repo="pasted-diff", language="python", ruleset_id="default"),
            n_chunks=1,
        ),
        GuardrailRefused(
            reason=f"Output matched {MASKED_PATTERN}.", masked=MASKED_PATTERN
        ),
        ReviewComplete(report=report),
    ]
    stream = FakeStream(events)
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))

    refusals = [m for m in FakeMessage.sent if "refused by the secret guardrail" in m.content]
    assert len(refusals) == 1
    assert MASKED_PATTERN in refusals[0].content
    assert THE_SECRET not in refusals[0].content
    # A refused review still completes with its measurements footer.
    assert "Measurements" in FakeMessage.sent[-1].content


@pytest.mark.asyncio
async def test_on_message_crash_renders_one_friendly_sentence(monkeypatch, capsys):
    _, session = install_fake_cl(monkeypatch)
    boom = RuntimeError("model exploded")
    stream = FakeStream(
        [
            ReviewStarted(
                request_id="rev_x",
                context=ReviewContext(repo="pasted-diff", language="python", ruleset_id="default"),
                n_chunks=1,
            )
        ],
        error=boom,
    )
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module.on_message(SimpleNamespace(content=VALID_DIFF))

    # The generator's finally ran: the stream was closed, never abandoned.
    assert stream.finally_ran
    friendly = [m for m in FakeMessage.sent if "could not be completed" in m.content]
    assert len(friendly) == 1
    assert "RuntimeError" in friendly[0].content
    assert "model exploded" not in friendly[0].content  # details stay in the log
    log = capsys.readouterr().out
    assert "model exploded" in log  # the server log holds the cause


@pytest.mark.asyncio
async def test_session_lock_serialises_reviews(monkeypatch):
    _, session = install_fake_cl(monkeypatch)
    overlap: list[str] = []

    class SlowStream(FakeStream):
        async def _gen(self):
            try:
                overlap.append("start")
                await asyncio.sleep(0.01)
                for event in self.events:
                    yield event
                overlap.append("end")
            finally:
                self.finally_ran = True

    stream = SlowStream(_events_for_review(make_report(), [], []))
    monkeypatch.setattr(app_module, "run_review", stream)
    session.set(app_module.SESSION_LOCK, asyncio.Lock())

    await asyncio.gather(
        app_module._handle_message(SimpleNamespace(content=VALID_DIFF)),
        app_module._handle_message(SimpleNamespace(content=VALID_DIFF)),
    )

    # Each review ran start→end with no interleave: the lock serialised them.
    assert overlap == ["start", "end", "start", "end"]
    assert stream.calls and stream.finally_ran


@pytest.mark.asyncio
async def test_session_lock_fallback_is_created_and_stored(monkeypatch):
    _, session = install_fake_cl(monkeypatch)
    stream = FakeStream(_events_for_review(make_report(), [], []))
    monkeypatch.setattr(app_module, "run_review", stream)
    assert app_module.SESSION_LOCK not in session.store  # no seed ran

    # Two messages racing before on_chat_start: the first stores ONE lock...
    await asyncio.gather(
        app_module._handle_message(SimpleNamespace(content=VALID_DIFF)),
        app_module._handle_message(SimpleNamespace(content=VALID_DIFF)),
    )
    stored = session.store[app_module.SESSION_LOCK]
    assert isinstance(stored, asyncio.Lock)

    # ...and a later message REUSES that same stored lock (never a new one).
    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))
    assert session.store[app_module.SESSION_LOCK] is stored


@pytest.mark.asyncio
async def test_body_raise_closes_generator_deterministically(monkeypatch, capsys):
    """The drain contract when the LOOP BODY raises, not the generator.

    A card .send() failing on a client disconnect must not leave the
    run_review generator suspended until asyncgen GC: aclose() in the
    handler's finally runs the generator's cleanup (contextvar resets) NOW.
    """
    install_fake_cl(monkeypatch)
    stream = FakeStream(
        [
            ReviewStarted(
                request_id="rev_x",
                context=ReviewContext(repo="pasted-diff", language="python", ruleset_id="default"),
                n_chunks=1,
            ),
            ReviewerStarted(reviewer="SecurityReviewer"),
        ]
    )
    monkeypatch.setattr(app_module, "run_review", stream)

    async def disconnect(event, status, state):
        raise RuntimeError("client disconnected mid-review")

    monkeypatch.setattr(app_module, "_handle_event", disconnect)

    await app_module.on_message(SimpleNamespace(content=VALID_DIFF))

    # The generator was suspended at a yield when the body raised; the
    # handler's finally closed it, so its cleanup ALREADY ran.
    assert stream.finally_ran
    # The user still sees exactly ONE friendly sentence (no traceback).
    friendly = [m for m in FakeMessage.sent if "could not be completed" in m.content]
    assert len(friendly) == 1
    assert "RuntimeError" in friendly[0].content
    assert "client disconnected" not in friendly[0].content
    assert "client disconnected" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_on_chat_start_seeds_state_and_sends_welcome(monkeypatch, capsys):
    _, session = install_fake_cl(monkeypatch)

    await app_module.on_chat_start()

    assert session.store[app_module.SESSION_REVIEW_COUNT] == 0
    assert session.store[app_module.SESSION_CONTEXT] is None
    assert session.store[app_module.SESSION_LAST_REPORT] is None
    assert isinstance(session.store[app_module.SESSION_LOCK], asyncio.Lock)
    # Exactly one welcome message, explaining the desk and the diff paste.
    assert len(FakeMessage.sent) == 1
    welcome = FakeMessage.sent[0].content
    assert "Code Review Desk" in welcome
    assert "unified diff" in welcome
    assert "reuses those settings" in welcome


@pytest.mark.asyncio
async def test_partial_review_reaches_status_and_card(monkeypatch):
    _, _ = install_fake_cl(monkeypatch)
    reason = "The reviewer hit its turn ceiling of 4 turns; findings may be incomplete."
    events = [
        ReviewStarted(
            request_id="rev_x",
            context=ReviewContext(repo="pasted-diff", language="python", ruleset_id="default"),
            n_chunks=1,
        ),
        ReviewerStarted(reviewer="QualityReviewer"),
        FindingsLanded(
            reviewer="QualityReviewer",
            findings=[make_finding("major")],
            partial=True,
            partial_reason=reason,
            latency_ms=400.0,
        ),
        PartialReview(reviewer="QualityReviewer", reason=reason),
        MergedReport(findings=[make_finding("major")]),
        RemediationOffered(escalation=None, text="No remediation needed."),
        ReviewComplete(report=make_report(findings=[make_finding("major")])),
    ]
    stream = FakeStream(events)
    monkeypatch.setattr(app_module, "run_review", stream)

    await app_module._handle_message(SimpleNamespace(content=VALID_DIFF))

    cards = [m for m in FakeMessage.sent if "— findings:" in m.content or "partial review" in m.content]
    assert len(cards) == 1
    assert "partial review" in cards[0].content
    assert reason in cards[0].content  # the card footnote carries the reason
    # The status message mentioned it too, along the way.
    status_history = "\n".join(FakeMessage.sent[0].history)
    assert "PARTIAL" in status_history
    assert "Partial review" in status_history
