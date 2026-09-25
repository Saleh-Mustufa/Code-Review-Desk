"""Unit and integration tests for src/pipeline.py and cli.py — fully offline.

The pipeline is exercised end-to-end with a stub runner that mirrors
LedgerRunner's hook injection (so run-level hooks fire with REAL canned usage
numbers) and a stub reviewers factory; the Desk run is canned or made to
raise the guardrail/ceiling exceptions the real runner would raise. No
network, no API keys, no real model calls.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from agents import MaxTurnsExceeded, RunConfig
from agents.guardrail import GuardrailFunctionOutput, OutputGuardrailResult
from agents.items import ToolCallOutputItem
from agents.run_context import RunContextWrapper
from agents.usage import Usage

import src.pipeline as pipeline_module
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
    render_report_markdown,
    run_review,
)
from src.review import Finding
from src.specialists import (
    Escalation,
    OutputGuardrailTripwireTriggered,
    current_secrets,
    last_escalation,
    secret_guardrail,
    set_current_secrets,
)

# --- Fakes ---------------------------------------------------------------------


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = None
        self.hooks = None


DIFF = (
    "diff --git a/app.py b/app.py\n"
    "--- a/app.py\n"
    "+++ b/app.py\n"
    "@@ -1,3 +1,5 @@\n"
    "+import os\n"
    "+API_KEY = \"gsk_aaaaaaaaaabbbbbbbbbb\"\n"
    " def main():\n"
    "     pass\n"
)
"""A small valid diff planting one credential-shaped assignment (FR-8)."""

THE_SECRET = 'API_KEY = "gsk_aaaaaaaaaabbbbbbbbbb"'


def make_ctx(**overrides: object) -> ReviewContext:
    fields = {
        "repo": "demo",
        "language": "python",
        "ruleset_id": "default",
        "strictness": "normal",
    }
    fields.update(overrides)
    return ReviewContext(**fields)  # type: ignore[arg-type]


def finding(file: str, line: int, severity: str, message: str) -> Finding:
    return Finding(file=file, line=line, severity=severity, message=message)  # type: ignore[arg-type]


def make_usage(requests: int, tokens_in: int, tokens_out: int) -> Usage:
    return Usage(
        requests=requests,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        total_tokens=tokens_in + tokens_out,
    )


class StubRunner:
    """Offline runner: mirrors LedgerRunner (hook injection), returns canned
    results, and can raise the exceptions a real runner would raise."""

    def __init__(
        self,
        reviewer_results: dict[str, "ReviewerStub"],
        desk_behaviour=None,
    ) -> None:
        self.reviewer_results = reviewer_results
        self.desk_behaviour = desk_behaviour
        self.calls: list[tuple[str, dict]] = []
        self.desk_runs = 0
        self.secrets_seen_at_desk: list[list[str] | None] = []

    async def run(self, agent: FakeAgent, input: object, **kwargs: object):
        hooks = pipeline_module.observe.current_run_hooks.get()
        if hooks is not None:
            kwargs.setdefault("hooks", hooks)
        self.calls.append((agent.name, dict(kwargs)))

        stub = self.reviewer_results.get(agent.name)
        if stub is not None:
            wrapper = RunContextWrapper(context=kwargs.get("context"))
            wrapper.usage = stub.usage
            if hooks is not None:
                await hooks.on_agent_start(wrapper, agent)
                await asyncio.sleep(stub.delay)
                await hooks.on_agent_end(wrapper, agent, stub.final_output)
            if stub.raise_max_turns:
                raise MaxTurnsExceeded(f"max turns exceeded for {agent.name}")
            return SimpleNamespace(
                final_output=stub.final_output,
                last_agent=agent,
                context_wrapper=wrapper,
                new_items=[],
            )

        # The Desk run: record what the pipeline had planted by now (FR-8 layer b).
        self.desk_runs += 1
        registry = current_secrets.get()
        self.secrets_seen_at_desk.append(list(registry.patterns) if registry else None)
        if self.desk_behaviour is not None:
            outcome = self.desk_behaviour(agent, input, kwargs)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            return outcome
        raise AssertionError("no desk behaviour configured")


class ReviewerStub:
    def __init__(
        self,
        findings: list[Finding],
        usage: Usage,
        *,
        delay: float = 0.0,
        raise_max_turns: bool = False,
    ) -> None:
        self.final_output = list(findings)
        self.usage = usage
        self.delay = delay
        self.raise_max_turns = raise_max_turns


def stub_reviewers() -> dict[str, FakeAgent]:
    return {
        name: FakeAgent(name)
        for name in ("SecurityReviewer", "TestsReviewer", "StyleReviewer")
    }


def default_reviewer_results() -> dict[str, ReviewerStub]:
    """Security and Tests report the SAME critical finding (merge/dedupe check)."""
    return {
        "SecurityReviewer": ReviewerStub(
            [
                finding("app.py", 42, "critical", "SQL injection risk"),
                finding("app.py", 7, "minor", "Unused import"),
            ],
            make_usage(2, 540, 210),
            delay=0.003,
        ),
        "TestsReviewer": ReviewerStub(
            [finding("app.py", 42, "critical", "SQL injection risk")],
            make_usage(1, 300, 100),
            delay=0.002,
        ),
        "StyleReviewer": ReviewerStub(
            [finding("util.py", 3, "minor", "Misleading name")],
            make_usage(1, 120, 40),
            delay=0.001,
        ),
    }


def desk_result(
    *,
    summary: str = "Desk: 3 findings (1 critical).",
    last_agent_name: str = "ReviewDesk",
    new_items: list | None = None,
) -> SimpleNamespace:
    agent = SimpleNamespace(name=last_agent_name)
    return SimpleNamespace(
        final_output=summary,
        last_agent=agent,
        context_wrapper=SimpleNamespace(usage=make_usage(4, 800, 300)),
        new_items=new_items or [],
    )


def merge_tool_item(merged: list[dict]) -> ToolCallOutputItem:
    return ToolCallOutputItem(
        raw_item={"type": "function_call_output", "name": "merge_findings"},
        agent=FakeAgent("MergeSpecialist"),
        output=json.dumps(merged),
    )


async def collect(gen: AsyncIterator[object]) -> list[object]:
    return [event async for event in gen]


# --- Happy path: event sequence, merge, footer ----------------------------------


@pytest.mark.asyncio
async def test_run_review_event_sequence_and_merged_findings() -> None:
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    ctx = make_ctx()

    events = await collect(
        run_review(
            DIFF,
            ctx,
            runner=runner,
            reviewers_factory=stub_reviewers,
            secret_planter=set_current_secrets,
        )
    )

    types = [type(event).__name__ for event in events]
    assert types[0] == "ReviewStarted"
    assert types[1:4] == ["ReviewerStarted"] * 3
    assert sorted(types[4:7]) == ["FindingsLanded"] * 3
    assert types[-3:] == ["MergedReport", "RemediationOffered", "ReviewComplete"]
    assert "PartialReview" not in types and "GuardrailRefused" not in types

    started = events[0]
    assert isinstance(started, ReviewStarted)
    assert started.request_id.startswith("rev_")
    assert started.context is ctx
    assert started.n_chunks == 1  # intake split the diff before any model call

    landed = [event for event in events if isinstance(event, FindingsLanded)]
    assert {event.reviewer for event in landed} == {
        "SecurityReviewer",
        "TestsReviewer",
        "StyleReviewer",
    }
    security = next(event for event in landed if event.reviewer == "SecurityReviewer")
    assert [f.message for f in security.findings] == [
        "SQL injection risk",
        "Unused import",
    ]
    assert security.partial is False

    merged = next(event for event in events if isinstance(event, MergedReport))
    # 3 merged rows: the duplicated critical collapses with both sources.
    assert [(f.severity, f.file, f.line) for f in merged.findings] == [
        ("critical", "app.py", 42),
        ("minor", "app.py", 7),
        ("minor", "util.py", 3),
    ]
    assert merged.findings[0].sources == ["SecurityReviewer", "TestsReviewer"]


@pytest.mark.asyncio
async def test_footer_carries_real_token_counts_from_run_context() -> None:
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    complete = events[-1]
    assert isinstance(complete, ReviewComplete)
    report = complete.report

    rows = {stats.reviewer_name: stats for stats in report.per_reviewer}
    assert rows["SecurityReviewer"].tokens_in == 540  # REAL canned run-context usage
    assert rows["SecurityReviewer"].tokens_out == 210
    assert rows["SecurityReviewer"].requests == 2
    assert rows["TestsReviewer"].tokens_in == 300
    assert rows["StyleReviewer"].tokens_out == 40

    footer = report.footer
    assert "| SecurityReviewer | " in footer
    assert "| TestsReviewer | " in footer
    assert "| StyleReviewer | " in footer
    assert "| **Total** |" in footer
    assert "540" in footer and "210" in footer  # per-row numbers present
    assert "960" in footer and "350" in footer  # totals 540+300+120 / 210+100+40
    # The footer is a table of numbers and names only — never diff or secret text.
    assert "gsk_" not in footer and "API_KEY" not in footer


@pytest.mark.asyncio
async def test_report_markdown_renders_findings_and_context() -> None:
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    events = await collect(
        run_review(DIFF, make_ctx(strictness="strict"), runner=runner, reviewers_factory=stub_reviewers)
    )
    report = events[-1].report
    markdown = render_report_markdown(report)
    assert f"#{report.request_id}" in markdown or report.request_id in markdown
    assert "1 critical" in markdown
    assert "SQL injection risk" in markdown
    assert "sources: SecurityReviewer, TestsReviewer" in markdown
    assert "## Measurements" in markdown
    assert report.trace_group_id.startswith("rev_")
    assert report.duration_ms >= 0


@pytest.mark.asyncio
async def test_desk_merge_tool_output_is_used_when_parseable() -> None:
    runner = StubRunner(default_reviewer_results())
    tool_merged = [
        {
            "file": "app.py",
            "line": 42,
            "severity": "critical",
            "message": "SQL injection risk",
            "sources": ["SecurityReviewer", "TestsReviewer"],
        }
    ]
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result(
        new_items=[merge_tool_item(tool_merged)]
    )
    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    merged = next(event for event in events if isinstance(event, MergedReport))
    assert [f.message for f in merged.findings] == ["SQL injection risk"]
    assert merged.findings[0].sources == ["SecurityReviewer", "TestsReviewer"]


# --- FR-7: run-level model override, zero agent edits ----------------------------


@pytest.mark.asyncio
async def test_model_override_rides_on_run_config_only() -> None:
    from src.model_config import model_for_run

    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    same_objects = stub_reviewers()
    events = await collect(
        run_review(
            DIFF,
            make_ctx(),
            runner=runner,
            reviewers_factory=lambda: same_objects,
            model_override="gemini-3.5-flash-lite",
        )
    )
    request_id = events[0].request_id
    reviewer_calls = [(name, kwargs) for name, kwargs in runner.calls if name in same_objects]
    assert len(reviewer_calls) == 3
    for _name, kwargs in reviewer_calls:
        run_config = kwargs["run_config"]
        assert isinstance(run_config, RunConfig)
        assert run_config.model is model_for_run("gemini-3.5-flash-lite")
        assert run_config.group_id == request_id
        assert run_config.workflow_name == "Code Review Desk"
    # The very same agent objects ran — no clones, no edits.
    ran_names = {name for name, _ in reviewer_calls}
    assert ran_names == set(same_objects)
    for agent in same_objects.values():
        assert agent.model is None  # untouched by the override


@pytest.mark.asyncio
async def test_no_override_leaves_run_config_model_none() -> None:
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    await collect(run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers))
    reviewer_calls = [kwargs for name, kwargs in runner.calls if name != "ReviewDesk"]
    assert all(kwargs["run_config"].model is None for kwargs in reviewer_calls)


# --- FR-8 layer (b): planting, refusal, no echo ----------------------------------


@pytest.mark.asyncio
async def test_pipeline_plants_diff_patterns_before_desk_run() -> None:
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    planted: list[list[str]] = []

    def planter(patterns: list[str]):
        planted.append(list(patterns))
        return set_current_secrets(patterns)

    await collect(
        run_review(
            DIFF,
            make_ctx(),
            runner=runner,
            reviewers_factory=stub_reviewers,
            secret_planter=planter,
        )
    )
    assert planted and planted[0], "the diff's credential-shaped text must be planted"
    assert any("API_KEY" in pattern for pattern in planted[0])
    assert runner.secrets_seen_at_desk and runner.secrets_seen_at_desk[0] == planted[0]
    assert current_secrets.get() is None  # planting is cleared after the desk run


@pytest.mark.asyncio
async def test_guardrail_trip_is_caught_as_refusal_never_an_echo() -> None:
    def tripping_desk(agent, input, kwargs):
        registry = current_secrets.get()
        planted = registry.patterns if registry else []
        text = f"Desk summary quoting {planted[0]}" if planted else "clean summary"
        if planted and planted[0] in text:
            info = {
                "guardrail": "secret_guardrail",
                "clean": False,
                "matches": ["API_***"],
                "reason": "Output quotes credential-shaped text; refused, not echoed.",
            }
            raise OutputGuardrailTripwireTriggered(
                OutputGuardrailResult(
                    guardrail=secret_guardrail,
                    agent_output=text,
                    agent=agent,
                    output=GuardrailFunctionOutput(
                        output_info=info, tripwire_triggered=True
                    ),
                )
            )
        return desk_result()

    runner = StubRunner(default_reviewer_results(), desk_behaviour=tripping_desk)
    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    types = [type(event).__name__ for event in events]
    assert "GuardrailRefused" in types
    assert types[-2:] == ["GuardrailRefused", "ReviewComplete"]
    assert "MergedReport" not in types  # no report content when refused

    refusal = next(event for event in events if isinstance(event, GuardrailRefused))
    assert "refused" in refusal.reason
    assert refusal.masked == "API_***"
    assert THE_SECRET not in (refusal.masked or "")

    report = events[-1].report
    assert report.refused is True
    assert report.findings == []  # nothing is echoed, ever
    markdown = render_report_markdown(report)
    assert THE_SECRET not in markdown and "gsk_" not in markdown


# --- FR-9: ceilings become partial reviews, never crashes -------------------------


@pytest.mark.asyncio
async def test_reviewer_ceiling_yields_partial_review_event() -> None:
    results = default_reviewer_results()
    results["StyleReviewer"] = ReviewerStub([], make_usage(1, 5, 5), raise_max_turns=True)
    runner = StubRunner(results)
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()

    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    partials = [event for event in events if isinstance(event, PartialReview)]
    assert len(partials) == 1
    assert partials[0].reviewer == "StyleReviewer"
    assert "turn ceiling" in partials[0].reason

    report = events[-1].report
    assert any("StyleReviewer" in note for note in report.footnotes)
    landed = next(
        event for event in events if isinstance(event, FindingsLanded) and event.reviewer == "StyleReviewer"
    )
    assert landed.partial is True and landed.findings == []


@pytest.mark.asyncio
async def test_desk_ceiling_falls_back_to_deterministic_merge() -> None:
    def ceiling_desk(agent, input, kwargs):
        raise MaxTurnsExceeded("desk ran out of turns")

    runner = StubRunner(default_reviewer_results(), desk_behaviour=ceiling_desk)
    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    types = [type(event).__name__ for event in events]
    assert types[-3:] == ["MergedReport", "RemediationOffered", "ReviewComplete"]
    report = events[-1].report
    assert any("turn ceiling" in note for note in report.footnotes)
    assert len(report.findings) == 3  # deterministic merge of the tagged findings
    assert report.remediation is None
    offered = next(event for event in events if isinstance(event, RemediationOffered))
    assert "No remediation needed" in offered.text


# --- NFR-4: a reviewer's model error becomes a partial outcome, not a crash ----


@pytest.mark.asyncio
async def test_reviewer_model_error_becomes_partial_outcome_concurrent(
    monkeypatch,
) -> None:
    """One reviewer dying on an unexpected model error must not crash the
    review: the others' findings land, the failure is surfaced as a partial
    outcome with the containment sentence, and the report still completes."""
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    real_run_reviewer = pipeline_module.run_reviewer

    async def flaky_run_reviewer(agent, diff, ctx, **kwargs):
        if agent.name == "StyleReviewer":
            raise RuntimeError("provider 400: forced function calling unsupported")
        return await real_run_reviewer(agent, diff, ctx, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_reviewer", flaky_run_reviewer)

    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    types = [type(event).__name__ for event in events]
    assert types[-3:] == ["MergedReport", "RemediationOffered", "ReviewComplete"]
    assert "GuardrailRefused" not in types

    landed = {
        event.reviewer: event for event in events if isinstance(event, FindingsLanded)
    }
    assert [f.message for f in landed["SecurityReviewer"].findings] != []  # healthy
    assert landed["StyleReviewer"].partial is True
    assert landed["StyleReviewer"].findings == []
    assert "could not complete its review (RuntimeError)" in (
        landed["StyleReviewer"].partial_reason or ""
    )
    assert "the review continues without it" in (
        landed["StyleReviewer"].partial_reason or ""
    )

    partials = [event for event in events if isinstance(event, PartialReview)]
    assert len(partials) == 1 and partials[0].reviewer == "StyleReviewer"
    report = events[-1].report
    assert any("StyleReviewer" in note for note in report.footnotes)
    # The healthy reviewers' findings still merged (Style contributed nothing).
    assert [(f.severity, f.file, f.line) for f in report.findings] == [
        ("critical", "app.py", 42),
        ("minor", "app.py", 7),
    ]


@pytest.mark.asyncio
async def test_reviewer_model_error_becomes_partial_outcome_sequential(
    monkeypatch,
) -> None:
    """Same containment in the sequential branch: the failed reviewer becomes
    a partial outcome and the loop continues to the next reviewer."""
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    real_run_reviewer = pipeline_module.run_reviewer

    async def flaky_run_reviewer(agent, diff, ctx, **kwargs):
        if agent.name == "TestsReviewer":
            raise RuntimeError("provider 400")
        return await real_run_reviewer(agent, diff, ctx, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_reviewer", flaky_run_reviewer)

    events = await collect(
        run_review(
            DIFF,
            make_ctx(),
            runner=runner,
            reviewers_factory=stub_reviewers,
            mode="sequential",
        )
    )
    types = [type(event).__name__ for event in events]
    assert types[-3:] == ["MergedReport", "RemediationOffered", "ReviewComplete"]

    landed = [event for event in events if isinstance(event, FindingsLanded)]
    assert [event.reviewer for event in landed] == [
        "SecurityReviewer",
        "TestsReviewer",
        "StyleReviewer",
    ]
    assert landed[1].partial is True
    assert "could not complete its review (RuntimeError)" in (
        landed[1].partial_reason or ""
    )
    # The loop continued: Style landed normally after the mid-roster failure.
    assert landed[2].partial is False and landed[2].findings

    report = events[-1].report
    assert any("TestsReviewer" in note for note in report.footnotes)
    assert [(f.severity, f.file, f.line) for f in report.findings] == [
        ("critical", "app.py", 42),
        ("minor", "app.py", 7),
        ("minor", "util.py", 3),
    ]


@pytest.mark.asyncio
async def test_reviewer_guardrail_trip_still_refuses_despite_containment(
    monkeypatch,
) -> None:
    """The NFR-4 containment must not swallow a guardrail trip: the trip
    branch is checked first, so the review is refused, never partialled."""
    real_run_reviewer = pipeline_module.run_reviewer

    async def tripping_run_reviewer(agent, diff, ctx, **kwargs):
        if agent.name == "SecurityReviewer":
            registry = current_secrets.get()
            planted = registry.patterns if registry else []
            text = f"quoted {planted[0]}" if planted else "clean"
            raise OutputGuardrailTripwireTriggered(
                OutputGuardrailResult(
                    guardrail=secret_guardrail,
                    agent_output=text,
                    agent=agent,
                    output=GuardrailFunctionOutput(
                        output_info={
                            "reason": "quoted a credential; refused.",
                            "matches": ["API_***"],
                        },
                        tripwire_triggered=True,
                    ),
                )
            )
        return await real_run_reviewer(agent, diff, ctx, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_reviewer", tripping_run_reviewer)

    for mode in ("concurrent", "sequential"):
        runner = StubRunner(default_reviewer_results())
        events = await collect(
            run_review(
                DIFF,
                make_ctx(),
                runner=runner,
                reviewers_factory=stub_reviewers,
                mode=mode,
            )
        )
        types = [type(event).__name__ for event in events]
        assert types[-2:] == ["GuardrailRefused", "ReviewComplete"], mode
        assert "PartialReview" not in types, mode
        report = events[-1].report
        assert report.refused is True and report.findings == []


# --- FR-6: remediation handoff surfaces as an event ------------------------------


@pytest.mark.asyncio
async def test_handoff_to_remediation_offers_the_patch() -> None:
    def handing_off_desk(agent, input, kwargs):
        # Mirrors the SDK's on_handoff callback: the escalation stays set for
        # the pipeline to read after the run (no reset — that is the design).
        last_escalation.set(
            Escalation(
                finding=finding("app.py", 42, "critical", "SQL injection risk"),
                reviewer="SecurityReviewer",
                reason="Critical injection finding needs a patch.",
            )
        )
        return desk_result(
            summary="```diff\n+os.environ['API_KEY']\n```",
            last_agent_name="RemediationSpecialist",
        )

    runner = StubRunner(default_reviewer_results(), desk_behaviour=handing_off_desk)
    events = await collect(
        run_review(DIFF, make_ctx(), runner=runner, reviewers_factory=stub_reviewers)
    )
    offered = next(event for event in events if isinstance(event, RemediationOffered))
    assert offered.escalation is not None
    assert offered.escalation.reviewer == "SecurityReviewer"
    assert "os.environ" in offered.text
    report = events[-1].report
    assert report.remediation == offered.text


# --- Modes: sequential comparison helper (FR-5) -----------------------------------


@pytest.mark.asyncio
async def test_sequential_mode_streams_one_reviewer_at_a_time() -> None:
    runner = StubRunner(default_reviewer_results())
    runner.desk_behaviour = lambda agent, input, kwargs: desk_result()
    events = await collect(
        run_review(
            DIFF,
            make_ctx(),
            runner=runner,
            reviewers_factory=stub_reviewers,
            mode="sequential",
        )
    )
    types = [type(event).__name__ for event in events]
    assert types[0] == "ReviewStarted"
    # Per-reviewer streaming: each reviewer announces, lands, then the next
    # starts — one at a time, in fan-out order.
    assert types[1:7] == [
        "ReviewerStarted",
        "FindingsLanded",
        "ReviewerStarted",
        "FindingsLanded",
        "ReviewerStarted",
        "FindingsLanded",
    ]
    landed = [event for event in events if isinstance(event, FindingsLanded)]
    assert [event.reviewer for event in landed] == [
        "SecurityReviewer",
        "TestsReviewer",
        "StyleReviewer",
    ]
    assert types[-3:] == ["MergedReport", "RemediationOffered", "ReviewComplete"]


# --- Cross-context finalization: the hardened contextvar resets --------------------


@pytest.mark.asyncio
async def test_abandoned_stream_finalized_in_a_fresh_context_does_not_raise() -> None:
    """A consumer abandoning the stream leaves the generator suspended; when
    it is finalized from a FRESH task (a different Context — the GC finalizer
    does exactly this), the closing contextvar resets must swallow the
    cross-context ValueError instead of crashing the finalizer."""

    class ExplodingRunner:
        """Any further run would explode — so the consumer abandons the stream."""

        async def run(self, *args: object, **kwargs: object):
            raise RuntimeError("model exploded")

    gen = run_review(
        DIFF,
        make_ctx(),
        runner=ExplodingRunner(),
        reviewers_factory=stub_reviewers,
    )
    first = await gen.__anext__()  # the contextvar set()s ran in THIS context
    assert isinstance(first, ReviewStarted)

    async def _finalize() -> None:
        await gen.aclose()  # resumes the body in the FRESH task's context

    await asyncio.ensure_future(_finalize())  # must not raise ValueError
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()  # the generator is cleanly closed


# --- DiffError propagates by design (CLI/UI catch it) -----------------------------


@pytest.mark.asyncio
async def test_malformed_diff_raises_diff_error_before_any_model_call() -> None:
    runner = StubRunner(default_reviewer_results())
    with pytest.raises(DiffError):
        await collect(run_review("not a diff at all", make_ctx(), runner=runner))
    assert runner.calls == []  # no run ever happened: split-before-model gate held


# --- render_footer unit checks -----------------------------------------------------


def test_render_footer_handles_empty_rows() -> None:
    report = ReviewReport(
        request_id="rev_x",
        repo="r",
        language="python",
        ruleset_id="default",
        strictness="normal",
        findings=[],
        per_reviewer=[],
    )
    footer = render_footer(report)
    assert "| **Total** | 0 | 0 | 0 |" in footer


# --- CLI (FR-1): friendly sentences, never a traceback -----------------------------


@pytest.fixture(autouse=True)
def quiet_tracing(monkeypatch):
    import cli

    # Hermetic CLI tests: no .env loading (a real .env may exist locally), no
    # global tracing changes.
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(cli, "setup_tracing", lambda: None)


def _cli():
    import cli

    return cli


def test_cli_help_exits_cleanly(capsys) -> None:
    cli = _cli()
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(cli.main(["--help"]))
    assert excinfo.value.code == 0
    assert "--diff" in capsys.readouterr().out


def test_cli_missing_gemini_key_is_one_sentence(monkeypatch, capsys) -> None:
    cli = _cli()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    code = asyncio.run(cli.main(["--diff", "examples/three_file_issue.diff"]))
    out = capsys.readouterr().out
    assert code == 1
    assert "GEMINI_API_KEY" in out
    assert "Traceback" not in out


def test_cli_missing_diff_file_is_one_sentence(monkeypatch, capsys, tmp_path) -> None:
    cli = _cli()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    code = asyncio.run(cli.main(["--diff", str(tmp_path / "no_such.diff")]))
    out = capsys.readouterr().out
    assert code == 1
    assert "no_such.diff" in out
    assert "Traceback" not in out


def test_cli_malformed_diff_is_a_message_not_a_traceback(
    monkeypatch, capsys, tmp_path
) -> None:
    cli = _cli()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    bad = tmp_path / "bad.diff"
    bad.write_text("this is not a unified diff", encoding="utf-8")
    code = asyncio.run(cli.main(["--diff", str(bad), "--repo", "demo", "--language", "python"]))
    out = capsys.readouterr().out
    assert code == 1
    assert "diff" in out.lower()
    assert "Traceback" not in out


def _fake_report(request_id: str) -> ReviewReport:
    report = ReviewReport(
        request_id=request_id,
        repo="demo",
        language="python",
        ruleset_id="default",
        strictness="normal",
        findings=[],
        per_reviewer=[
            pipeline_module.ReviewerStats(
                reviewer_name="SecurityReviewer",
                latency_ms=12.0,
                tokens_in=540,
                tokens_out=210,
                requests=2,
            )
        ],
    )
    report.footer = render_footer(report)
    return report


def test_cli_happy_path_prints_progressive_status_and_report(
    monkeypatch, capsys, tmp_path
) -> None:
    cli = _cli()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    diff_file = tmp_path / "ok.diff"
    diff_file.write_text(DIFF, encoding="utf-8")

    async def fake_run_review(diff_text, ctx, **kwargs):
        yield ReviewStarted(request_id="rev_cli01", context=ctx, n_chunks=1)
        yield ReviewerStarted(reviewer="SecurityReviewer")
        yield FindingsLanded(
            reviewer="SecurityReviewer",
            findings=[],
            partial=False,
            partial_reason=None,
            latency_ms=12.0,
        )
        yield ReviewComplete(report=_fake_report("rev_cli01"))

    monkeypatch.setattr(cli, "run_review", fake_run_review)
    code = asyncio.run(
        cli.main(
            [
                "--diff",
                str(diff_file),
                "--repo",
                "demo",
                "--language",
                "python",
            ]
        )
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "SecurityReviewer landed" in out  # progressive status line
    assert "Code Review Desk" in out  # the rendered report
    assert "| **Total** |" in out  # the FR-10 footer
    assert "Traceback" not in out


def test_cli_concurrent_timing_prints_both_wall_clocks(
    monkeypatch, capsys, tmp_path
) -> None:
    cli = _cli()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    diff_file = tmp_path / "ok.diff"
    diff_file.write_text(DIFF, encoding="utf-8")

    async def fake_run_review(diff_text, ctx, *, mode="concurrent", **kwargs):
        yield ReviewStarted(request_id="rev_cli02", context=ctx, n_chunks=1)
        yield ReviewComplete(report=_fake_report("rev_cli02"))

    monkeypatch.setattr(cli, "run_review", fake_run_review)
    code = asyncio.run(
        cli.main(["--diff", str(diff_file), "--concurrent-timing"])
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Concurrent:" in out and "Sequential:" in out
    assert "slowest reviewer" in out


def test_cli_unexpected_error_is_one_sentence(monkeypatch, capsys, tmp_path) -> None:
    cli = _cli()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-tests")
    diff_file = tmp_path / "ok.diff"
    diff_file.write_text(DIFF, encoding="utf-8")

    async def exploding_run_review(diff_text, ctx, **kwargs):
        raise RuntimeError("router exploded")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(cli, "run_review", exploding_run_review)
    code = asyncio.run(cli.main(["--diff", str(diff_file)]))
    out = capsys.readouterr().out
    assert code == 1
    assert "review could not be completed" in out
    assert "Traceback" not in out
