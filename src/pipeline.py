"""Pipeline: the review orchestrator, the typed event stream and the report
(AD-8, AD-9, AD-10; FR-5, FR-7, FR-8, FR-10, FR-11, FR-12, FR-13).

:func:`run_review` is an async generator — the single path every entry point
(CLI, Chainlit UI) drives. It:

1. intakes the diff (``DiffError`` **propagates by design** — the CLI/UI catch
   it and render the friendly message; this module never turns a validation
   problem into an event),
2. opens ONE trace for the whole review (``with trace(workflow_name=...,
   group_id=request_id)``), so nested reviewer runs join the same trace with
   overlapping spans (FR-13),
3. fans the three reviewers out concurrently and yields ``FindingsLanded`` as
   each lands (FR-5, FR-12), flagging turn-ceiling partials (FR-9),
4. plants the diff's credential patterns (FR-8 layer b) BEFORE the Desk run,
   tags findings per reviewer, and runs the Desk (merge tool + remediation
   handoff), catching the secret guardrail's tripwire (FR-8) and the Desk's
   turn ceiling,
5. yields ``MergedReport`` / ``RemediationOffered`` / ``ReviewComplete`` with a
   fully rendered :class:`ReviewReport` whose footer carries per-reviewer
   latency and REAL token usage from the run contexts (FR-10).

Every run goes through the ledger runner by default, so one review produces
one ledger line per run (FR-11). FR-7 is honoured without edits: a
``model_override`` is delivered purely via ``RunConfig(model=...)`` on the
SAME reviewer objects.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import secrets as _secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from agents import MaxTurnsExceeded, RunConfig, Runner
from agents.tracing import trace

from src import observe
from src.intake import ReviewContext, intake
from src.model_config import current_model_name, model_for_run
from src.observe import (
    MetricsRunHooks,
    ReviewerStats,
    WORKFLOW_NAME,
    attach_agent_hooks,
    current_run_hooks,
)
from src.review import (
    Finding,
    ReviewerOutcome,
    make_reviewers,
    run_reviewer,
)
from src.specialists import (
    DESK_MAX_TURNS,
    Escalation,
    MergedFinding,
    OutputGuardrailTripwireTriggered,
    dedupe_and_order,
    extract_credential_patterns,
    guard_against_secrets,
    last_escalation,
    make_desk_agent,
    reset_current_secrets,
    set_current_secrets,
    tag_findings,
)

__all__ = [
    "FindingsLanded",
    "GuardrailRefused",
    "MergedReport",
    "PartialReview",
    "RemediationOffered",
    "ReviewComplete",
    "ReviewReport",
    "ReviewerStarted",
    "ReviewStarted",
    "render_footer",
    "render_report_markdown",
    "run_review",
]


# --- Typed events (AD-10): what streams out of run_review ---


@dataclass
class ReviewStarted:
    """The review has opened its trace and intaken the diff."""

    request_id: str
    context: ReviewContext
    n_chunks: int


@dataclass
class ReviewerStarted:
    """One reviewer has been launched."""

    reviewer: str


@dataclass
class FindingsLanded:
    """One reviewer's findings have landed (partial reviews flagged)."""

    reviewer: str
    findings: list[Finding]
    partial: bool
    partial_reason: str | None
    latency_ms: float | None


@dataclass
class PartialReview:
    """A reviewer hit its turn ceiling — surfaced as its own event (FR-9)."""

    reviewer: str
    reason: str


@dataclass
class MergedReport:
    """The deduplicated, severity-ordered merged findings."""

    findings: list[MergedFinding]


@dataclass
class RemediationOffered:
    """A remediation proposal, or the absence note when none was needed."""

    escalation: Escalation | None
    text: str


@dataclass
class GuardrailRefused:
    """The secret guardrail tripped; the output was refused, never echoed."""

    reason: str
    masked: str | None


@dataclass
class ReviewComplete:
    """The final, fully rendered report."""

    report: "ReviewReport"


# --- The report and its renderers (FR-10 footer, Task 6 consumer) ---


@dataclass
class ReviewReport:
    """Everything one review produced, safe fields only.

    ``per_reviewer`` rows carry latency and REAL token usage from the run
    contexts (never estimated). ``footnotes`` lists partial reviews. Probe
    events come from the agent-level hooks on the one probed reviewer.
    """

    request_id: str
    repo: str
    language: str
    ruleset_id: str
    strictness: str
    findings: list[MergedFinding]
    per_reviewer: list[ReviewerStats]
    footnotes: list[str] = field(default_factory=list)
    remediation: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    trace_group_id: str = ""
    duration_ms: float = 0.0
    footer: str = ""
    probe_events: list[str] = field(default_factory=list)


def render_footer(report: ReviewReport) -> str:
    """The FR-10 footer: one markdown row per reviewer plus a total row."""
    lines = [
        "| Reviewer | Latency (ms) | Tokens in | Tokens out |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_in = 0
    total_out = 0
    total_ms = 0.0
    for stats in report.per_reviewer:
        total_in += stats.tokens_in
        total_out += stats.tokens_out
        total_ms += stats.latency_ms
        lines.append(
            f"| {stats.reviewer_name} | {stats.latency_ms:.0f} "
            f"| {stats.tokens_in} | {stats.tokens_out} |"
        )
    lines.append(
        f"| **Total** | {total_ms:.0f} | {total_in} | {total_out} |"
    )
    return "\n".join(lines)


def render_report_markdown(report: ReviewReport) -> str:
    """Render the full report as severity-styled markdown (Task 6 consumes)."""
    lines: list[str] = [
        f"# Code Review Desk — report `{report.request_id}`",
        "",
        f"Repo: `{report.repo}` | Language: `{report.language}` "
        f"| Ruleset: `{report.ruleset_id}` | Strictness: `{report.strictness}`",
        "",
    ]
    if report.refused:
        lines.append(
            "## Refused\n\n"
            f"{report.refusal_reason or 'The output was refused by the secret guardrail.'}\n\n"
            "No findings are shown: the report would have quoted a credential."
        )
    else:
        criticals = sum(1 for f in report.findings if f.severity == "critical")
        lines.append(
            f"**{len(report.findings)}** merged finding(s), {criticals} critical.\n"
        )
        if not report.findings:
            lines.append("No findings — the diff looks clean to the desk.\n")
        for f in report.findings:
            badge = {"critical": "🔴 CRITICAL", "major": "🟠 MAJOR", "minor": "🟡 MINOR"}.get(
                f.severity, f.severity.upper()
            )
            sources = (
                f" *(sources: {', '.join(f.sources)})*" if f.sources else ""
            )
            lines.append(f"- **{badge}** `{f.file}:{f.line}` — {f.message}{sources}")
    if report.remediation:
        lines.extend(["", "## Remediation proposal", "", report.remediation])
    if report.footnotes:
        lines.extend(["", "## Footnotes (partial reviews)"])
        lines.extend(f"- {note}" for note in report.footnotes)
    lines.extend(["", "## Measurements", "", report.footer])
    if report.probe_events:
        shown = report.probe_events[:20]
        lines.extend(
            [
                "",
                "## Security reviewer probe (agent-level hook events)",
                "",
                ", ".join(shown) + (" …" if len(report.probe_events) > len(shown) else ""),
            ]
        )
    return "\n".join(lines)


# --- Internals ---


def _build_run_config(request_id: str, model_override: str | None) -> RunConfig:
    """One fresh RunConfig per run, always naming the workflow and group (FR-13).

    FR-7 lives here: ``model_override`` rides ONLY on ``RunConfig(model=...)`` —
    the same reviewer objects are re-run under another model with zero edits.
    """
    return RunConfig(
        workflow_name=WORKFLOW_NAME,
        group_id=request_id,
        model=model_for_run(model_override) if model_override else None,
    )


def _model_label(agent: Any, model_override: str | None) -> str | None:
    """Cheap model-name introspection for the footer (no hardcoded names)."""
    if model_override:
        try:
            return current_model_name(model_for_run(model_override))
        except Exception:  # noqa: BLE001 - a label never breaks a review
            return model_override
    model = getattr(agent, "model", None)
    if isinstance(model, str):
        return model
    if model is None:
        return None
    try:
        return current_model_name(model)
    except Exception:  # noqa: BLE001
        return None


def _stats_for(
    outcomes: list[ReviewerOutcome],
    hooks_stats: dict[str, ReviewerStats],
    agents_by_name: dict[str, Any],
    model_override: str | None,
) -> list[ReviewerStats]:
    """Per-reviewer rows: real usage from the run-context hooks, latency fallback.

    The hooks' per-run usage (read off each run's own context wrapper) is the
    source of truth for tokens; the outcome's own latency backs the hooks up
    when a custom runner did not fire them. Rows follow the AGENT order (the
    fan-out roster), not landing order, so the same review input always yields
    the same footer.
    """
    outcomes_by_name = {outcome.reviewer_name: outcome for outcome in outcomes}
    # agents_by_name preserves construction order; outcomes from agents
    # outside the roster (defensive) keep their landing order after it.
    ordered_names = list(agents_by_name) + [
        name for name in outcomes_by_name if name not in agents_by_name
    ]
    rows: list[ReviewerStats] = []
    for name in ordered_names:
        outcome = outcomes_by_name.get(name)
        if outcome is None:
            continue
        stats = hooks_stats.get(name)
        if stats is None:
            stats = ReviewerStats(reviewer_name=name)
        if not stats.latency_ms and outcome.latency_ms is not None:
            stats.latency_ms = outcome.latency_ms
        agent = agents_by_name.get(name)
        stats.model = stats.model or _model_label(agent, model_override)
        rows.append(stats)
    return rows


def _parse_merged_output(output: Any) -> list[MergedFinding] | None:
    """Parse a merge tool output (JSON string or list) into MergedFindings."""
    data = output
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    if not isinstance(data, list):
        return None
    try:
        return [MergedFinding.model_validate(entry) for entry in data]
    except Exception:  # noqa: BLE001 - any shape mismatch falls back
        return None


def _merged_from_tool_output(result: Any) -> list[MergedFinding] | None:
    """Extract the merged list from the Desk run's merge tool output, if any.

    The Desk's merge tool output - the agentic path, visible in the trace - is
    preferred whenever it parses cleanly and is non-empty; the deterministic
    :func:`dedupe_and_order` result is the fallback when it does not.
    """
    for item in getattr(result, "new_items", None) or []:
        if type(item).__name__ != "ToolCallOutputItem":
            continue
        parsed = _parse_merged_output(getattr(item, "output", None))
        if parsed:
            return parsed
    return None


def _desk_findings_message(tagged: list[MergedFinding]) -> str:
    """The Desk's user message: the tagged findings as JSON plus instruction."""
    return (
        "The reviewers' findings follow as a JSON array "
        "(file, line, severity, message, sources):\n\n"
        + json.dumps([f.model_dump(mode="json") for f in tagged], indent=2)
        + "\n\nCall the merge_findings tool once with this array, then write "
        "the short summary report (or hand off to remediation if a CRITICAL "
        "security finding needs a patch)."
    )


def _refusal_from_trip(trip: Any) -> tuple[str, str | None]:
    """Extract (reason, masked) from a guardrail tripwire exception (FR-8).

    The one place a trip is translated into a refusal: the reason comes from
    the guardrail's ``output_info`` (already masked by the guardrail — never a
    full secret); a missing structure falls back to a generic sentence.
    """
    info = getattr(getattr(trip, "guardrail_result", None), "output", None)
    output_info = getattr(info, "output_info", None) or {}
    reason = output_info.get("reason") or (
        "The desk output quoted credential-shaped text found in "
        "the diff; it was refused rather than echoed."
    )
    matches = output_info.get("matches")
    masked = ", ".join(str(m) for m in matches) if isinstance(matches, list) else None
    return reason, masked


def _as_agents(reviewers: Any) -> list[Any]:
    """Accept a name-keyed dict or a sequence of reviewer agents."""
    if isinstance(reviewers, dict):
        return list(reviewers.values())
    return list(reviewers)


# --- The orchestrator ---


async def run_review(
    diff_text: str,
    ctx: ReviewContext,
    *,
    model_override: str | None = None,
    runner: Runner | None = None,
    mode: Literal["concurrent", "sequential"] = "concurrent",
    reviewers_factory: Callable[[], Any] | None = None,
    secret_planter: Callable[[list[str]], Any] = set_current_secrets,
) -> AsyncIterator[Any]:
    """Run one full review, streaming typed events as they happen (AD-10).

    Args:
        diff_text: the whole unified diff (intake splits it before any model
            call — that split IS the pre-model gate, FR-1).
        ctx: the review context handed to every run (FR-2).
        model_override: optional router model name delivered via
            ``RunConfig(model=...)`` — FR-7 with zero agent edits.
        runner: injectable runner; defaults to the ledger runner so every run
            lands in ``ledger.jsonl`` (FR-11). Pass a stub for offline tests.
        mode: ``"concurrent"`` (default, FR-5) or ``"sequential"`` — the latter
            exists only for the wall-clock comparison.
        reviewers_factory: injectable reviewer maker for offline tests.
        secret_planter: injectable for tests; default :func:`set_current_secrets`.

    Yields:
        ``ReviewStarted``, ``ReviewerStarted`` (per reviewer), ``FindingsLanded``
        (per reviewer, as it lands) plus ``PartialReview`` when a ceiling is
        hit, ``MergedReport``, ``RemediationOffered``, then ``ReviewComplete``
        with the rendered report. On a tripped secret guardrail:
        ``GuardrailRefused`` then a refused ``ReviewComplete``. A reviewer
        whose run dies on an unexpected model error lands as a partial
        outcome (footnote + ``PartialReview``) instead of crashing the
        review — NFR-4 containment, in both modes.

    Raises:
        DiffError: from intake on empty/malformed diffs — callers catch it and
            render ``.message``; no event stands in for a validation problem.

    Note:
        Consumers should drain the generator (or close it promptly); the
        closing contextvar resets are hardened, so a generator finalized in a
        different Context — e.g. a UI consumer abandoning the stream — can
        never crash, and at worst a stale context retains the planted
        patterns.
    """
    chosen_runner = runner if runner is not None else getattr(observe, "LEDGER_RUNNER", Runner)
    request_id = f"rev_{_secrets.token_hex(4)}"
    started_at = time.perf_counter()
    chunks = await intake(diff_text)
    full_diff = diff_text

    # FR-8 layer (b): plant the diff's credential patterns for the WHOLE review
    # — reviewers included — so the guardrail exact-matches what the diff held
    # no matter which voice quotes it. Reset when the review ends.
    planted_token: contextvars.Token | None = secret_planter(
        extract_credential_patterns(full_diff)
    )

    hooks_stats: dict[str, ReviewerStats] = {}
    run_hooks = MetricsRunHooks(hooks_stats)
    footnotes: list[str] = []
    outcomes: list[ReviewerOutcome] = []
    agents_by_name: dict[str, Any] = {}

    hooks_token = current_run_hooks.set(run_hooks)
    try:
        with trace(workflow_name=WORKFLOW_NAME, group_id=request_id):
            yield ReviewStarted(request_id=request_id, context=ctx, n_chunks=len(chunks))

            reviewers = (
                reviewers_factory() if reviewers_factory is not None else make_reviewers()
            )
            agents = _as_agents(reviewers)
            probe = attach_agent_hooks(reviewers)
            agents_by_name = {agent.name: agent for agent in agents}

            # AD-6: every voice carries the secret guardrail. A reviewer that
            # quotes a credential trips HERE, before its findings can reach any
            # report; the trip becomes a refusal, not a crash.
            for agent in agents:
                agent.output_guardrails = [guard_against_secrets]

            async def _guarded_reviewer(agent: Any) -> Any:
                """One reviewer run; a guardrail trip becomes a sentinel and a
                model error becomes a partial outcome (NFR-4: one reviewer's
                failure must never crash the whole review)."""
                try:
                    return await run_reviewer(
                        agent,
                        full_diff,
                        ctx,
                        runner=chosen_runner,
                        run_config=_build_run_config(request_id, model_override),
                    )
                except OutputGuardrailTripwireTriggered as trip:
                    return trip
                except Exception as exc:  # noqa: BLE001 - NFR-4 containment
                    # Deliberately NOT BaseException: CancelledError (raised
                    # when the refusal path cancels stragglers) and
                    # KeyboardInterrupt/SystemExit still propagate.
                    return ReviewerOutcome(
                        reviewer_name=agent.name,
                        findings=[],
                        partial=True,
                        partial_reason=(
                            f"{agent.name} could not complete its review "
                            f"({type(exc).__name__}); the review continues "
                            "without it."
                        ),
                    )

            refusal_reason: str | None = None
            masked_text: str | None = None

            # Launch every reviewer, then announce them; findings stream as they land.
            if mode == "sequential":
                # One reviewer at a time, still streaming: announce one
                # reviewer, run it, yield its findings, then move to the next.
                for agent in agents:
                    yield ReviewerStarted(reviewer=agent.name)
                    try:
                        outcome = await run_reviewer(
                            agent,
                            full_diff,
                            ctx,
                            runner=chosen_runner,
                            run_config=_build_run_config(request_id, model_override),
                        )
                    except OutputGuardrailTripwireTriggered as trip:
                        refusal_reason, masked_text = _refusal_from_trip(trip)
                        break
                    except Exception as exc:  # noqa: BLE001 - NFR-4 containment
                        # Same containment as the concurrent path (except
                        # Exception, so CancelledError still propagates): a
                        # model error becomes a partial outcome and the loop
                        # moves on to the next reviewer.
                        outcome = ReviewerOutcome(
                            reviewer_name=agent.name,
                            findings=[],
                            partial=True,
                            partial_reason=(
                                f"{agent.name} could not complete its review "
                                f"({type(exc).__name__}); the review continues "
                                "without it."
                            ),
                        )
                    outcomes.append(outcome)
                    yield FindingsLanded(
                        reviewer=outcome.reviewer_name,
                        findings=outcome.findings,
                        partial=outcome.partial,
                        partial_reason=outcome.partial_reason,
                        latency_ms=outcome.latency_ms,
                    )
            else:
                tasks = [
                    asyncio.ensure_future(_guarded_reviewer(agent)) for agent in agents
                ]
                for agent in agents:
                    yield ReviewerStarted(reviewer=agent.name)
                for done in asyncio.as_completed(tasks):
                    landed = await done
                    if isinstance(landed, OutputGuardrailTripwireTriggered):
                        refusal_reason, masked_text = _refusal_from_trip(landed)
                        for task in tasks:
                            task.cancel()
                        # Retrieve every straggler's result — some may have
                        # finished with the sentinel or a real exception just
                        # before the cancel landed — so nothing later surfaces
                        # the "Task exception was never retrieved" noise.
                        try:
                            await asyncio.gather(*tasks, return_exceptions=True)
                        except Exception:  # noqa: BLE001 - cleanup is best-effort
                            pass
                        break
                    outcomes.append(landed)
                    yield FindingsLanded(
                        reviewer=landed.reviewer_name,
                        findings=landed.findings,
                        partial=landed.partial,
                        partial_reason=landed.partial_reason,
                        latency_ms=landed.latency_ms,
                    )

            for outcome in outcomes:
                if outcome.partial and outcome.partial_reason:
                    footnotes.append(outcome.partial_reason)
                    yield PartialReview(reviewer=outcome.reviewer_name, reason=outcome.partial_reason)

            desk_result: Any = None
            remediation_text: str | None = None
            escalation: Escalation | None = None
            handed_off = False

            if refusal_reason is None:
                # Tag findings per reviewer so merged rows can name their sources.
                # Tagging follows the REVIEWER ORDER (not landing order) so the
                # merged output stays deterministic for the same review input.
                outcomes_by_name = {
                    outcome.reviewer_name: outcome
                    for outcome in outcomes
                    if outcome.reviewer_name in agents_by_name
                }
                tagged: list[MergedFinding] = []
                for agent in agents:
                    outcome = outcomes_by_name.get(agent.name)
                    if outcome is not None:
                        tagged.extend(
                            tag_findings(outcome.findings, outcome.reviewer_name)
                        )

                try:
                    desk_result = await chosen_runner.run(
                        make_desk_agent(),
                        _desk_findings_message(tagged),
                        context=ctx,
                        max_turns=DESK_MAX_TURNS,
                        run_config=_build_run_config(request_id, model_override),
                    )
                    handed_off = (
                        getattr(getattr(desk_result, "last_agent", None), "name", "")
                        == "RemediationSpecialist"
                    )
                except OutputGuardrailTripwireTriggered as trip:
                    # THE line that catches the guardrail at desk level (FR-8):
                    # refusal, never a crash, never an echo of the secret.
                    refusal_reason, masked_text = _refusal_from_trip(trip)
                except MaxTurnsExceeded:
                    footnotes.append(
                        f"The desk coordinator hit its turn ceiling of {DESK_MAX_TURNS} "
                        "turns; the report below was assembled directly from the "
                        "reviewers' merged findings without a desk summary."
                    )

            if refusal_reason is not None:
                report = ReviewReport(
                    request_id=request_id,
                    repo=ctx.repo,
                    language=ctx.language,
                    ruleset_id=ctx.ruleset_id,
                    strictness=ctx.strictness,
                    findings=[],
                    per_reviewer=_stats_for(
                        outcomes, hooks_stats, agents_by_name, model_override
                    ),
                    footnotes=footnotes,
                    remediation=None,
                    refused=True,
                    refusal_reason=refusal_reason,
                    trace_group_id=request_id,
                    duration_ms=(time.perf_counter() - started_at) * 1000.0,
                    probe_events=list(probe.events),
                )
                report.footer = render_footer(report)
                yield GuardrailRefused(reason=refusal_reason, masked=masked_text)
                yield ReviewComplete(report=report)
                return

            # Prefer the Desk's merge tool output (the agentic path, visible in
            # the trace); the deterministic dedupe_and_order result is the
            # fallback when the tool output does not parse or is empty.
            merged = dedupe_and_order(tagged)
            if desk_result is not None:
                tool_merged = _merged_from_tool_output(desk_result)
                if tool_merged:
                    merged = tool_merged

            if handed_off and desk_result is not None:
                final_output = getattr(desk_result, "final_output", None)
                remediation_text = (
                    final_output if isinstance(final_output, str) else str(final_output)
                )
                escalation = last_escalation.get()
            yield MergedReport(findings=merged)
            yield RemediationOffered(
                escalation=escalation,
                text=remediation_text
                or "No remediation needed — no critical security finding required a patch.",
            )

            report = ReviewReport(
                request_id=request_id,
                repo=ctx.repo,
                language=ctx.language,
                ruleset_id=ctx.ruleset_id,
                strictness=ctx.strictness,
                findings=merged,
                per_reviewer=_stats_for(outcomes, hooks_stats, agents_by_name, model_override),
                footnotes=footnotes,
                remediation=remediation_text,
                refused=False,
                refusal_reason=None,
                trace_group_id=request_id,
                duration_ms=(time.perf_counter() - started_at) * 1000.0,
                probe_events=list(probe.events),
            )
            report.footer = render_footer(report)
            yield ReviewComplete(report=report)
    finally:
        # Cross-context hardening: an async generator finalized in a different
        # Context than the one where set() ran (e.g. a UI consumer abandoning
        # the stream) makes ContextVar.reset(token) raise ValueError. Swallow
        # it — an abandoned stream must not crash its finalizer; a stale
        # context then merely retains the planted patterns (see docstring).
        try:
            current_run_hooks.reset(hooks_token)
        except ValueError:
            pass
        if isinstance(planted_token, contextvars.Token):
            try:
                reset_current_secrets(planted_token)
            except ValueError:
                pass
