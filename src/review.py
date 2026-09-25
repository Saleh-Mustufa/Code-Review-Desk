"""Reviewers: Finding, the base reviewer, its three clones, fan-out and ceilings.

The review core (FR-3, FR-4, FR-5, FR-9). One base reviewer carries the shared
prompt logic; SecurityReviewer, TestsReviewer and StyleReviewer are clones that
differ only in focus instructions and bounded model settings. Every reviewer:

- has ``output_type=list[Finding]`` (FR-3), so ``final_output`` is a plain list;
- assembles its system prompt at run time from the context's ruleset and
  language (FR-4) — context content flows through ``RunContextWrapper``, never
  into prompt text (FR-2: no repository name appears in any prompt);
- is forced to consult the ruleset tool first (``tool_choice="required"`` with
  ``reset_tool_choice=True``, FR-9), and every tool failure is a sentence,
  never an exception into the runner (NFR-4);
- runs under a turn ceiling of 4 (FR-9): exceeding it becomes a *partial
  review* outcome, never a raise.

Models come only from the router (:func:`src.model_config.get_model`); no
model name is hardcoded here. Guardrails are Task 4's concern and are wired
onto specialists, not onto this module.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    Runner,
    RunConfig,
    RunContextWrapper,
    function_tool,
)
from agents.run_error_handlers import RunErrorHandlerInput, RunErrorHandlerResult
from pydantic import BaseModel

from src.intake import ReviewContext, load_ruleset
from src.model_config import get_model

__all__ = [
    "FOCUS_PROMPTS",
    "REVIEWER_MAX_TURNS",
    "Finding",
    "ReviewerOutcome",
    "base_reviewer_instructions",
    "make_base_reviewer",
    "make_reviewers",
    "read_ruleset",
    "run_reviewer",
    "run_reviewers_concurrently",
    "run_reviewers_sequentially",
]

REVIEWER_MAX_TURNS = 4
"""Turn ceiling for every reviewer run (FR-9).

The minimum viable review is 2 turns: one forced ruleset tool call, then the
final structured answer. 4 gives headroom for a ruleset re-read or an extra
loop the model needs before committing to findings, without letting a confused
run burn the rate-limit budget.
"""


class Finding(BaseModel):
    """One review finding — exactly the PDF shape (FR-3)."""

    file: str
    line: int
    severity: Literal["critical", "major", "minor"]
    message: str


@dataclass
class ReviewerOutcome:
    """What one reviewer run produced, typed across the boundary.

    ``partial`` marks a review that hit its turn ceiling (or finished without a
    usable findings list); ``partial_reason`` carries the human-readable
    sentence explaining it. ``usage_summary`` is a placeholder one-liner —
    Task 5 owns real usage metrics and latency plumbing.
    """

    reviewer_name: str
    findings: list[Finding] = field(default_factory=list)
    partial: bool = False
    partial_reason: str | None = None
    usage_summary: str | None = None
    latency_ms: float | None = None


# --- The ruleset tool (FR-2, FR-9, NFR-4) ---


@function_tool
async def read_ruleset(ctx: RunContextWrapper[ReviewContext]) -> str:
    """Load the repository review ruleset named by the review context."""
    # load_ruleset never raises (NFR-4): every failure already comes back as a
    # sentence the model can act on. The try/except is belt and braces so no
    # unexpected error can ever raise into the runner.
    try:
        return load_ruleset(ctx.context.ruleset_id)
    except Exception as exc:  # noqa: BLE001 - tools never raise into the runner
        return (
            "Error: the ruleset could not be loaded "
            f"({type(exc).__name__}: {exc}). Continue using built-in judgement and say so."
        )


# --- Dynamic instructions (FR-4): one assembler, shared by base and clones ---


FOCUS_PROMPTS: dict[str, str] = {
    "SecurityReviewer": (
        "Focus on security defects: injection, unsafe deserialisation, hardcoded "
        "credentials, broken authentication, TLS or authorisation, and every "
        "security rule in the ruleset."
    ),
    "TestsReviewer": (
        "Focus on test coverage gaps: behaviour changes without corresponding "
        "test changes, assertion-free tests, and untested error paths or "
        "boundary values named by the ruleset."
    ),
    "StyleReviewer": (
        "Focus on style and maintainability defects: dead code, misleading "
        "names, missing docstrings, and the structural limits stated by the "
        "ruleset."
    ),
}


def _assemble_reviewer_instructions(context: ReviewContext, *, focus: str) -> str:
    """Build one reviewer system prompt from the run context (FR-4).

    Shared by the base reviewer and every clone — the only varying input is the
    focus text. The ruleset is loaded at request time via
    ``load_ruleset(context.ruleset_id)``. Under ``strictness="strict"`` the
    framing is visibly terser: short imperative lines, no elaboration section.

    Context content never enters the prompt: ``context.repo`` is deliberately
    unread here (FR-2), so grepping any prompt for a repository name finds
    nothing.
    """
    strict = context.strictness.strip().lower() == "strict"
    ruleset_text = load_ruleset(context.ruleset_id)
    focus_text = focus.strip()

    if strict:
        lines = [
            "Strict review mode. Be terse.",
            f"Language: {context.language}.",
            "Ruleset (loaded at request time):",
            ruleset_text,
        ]
        if focus_text:
            lines.append(focus_text)
        lines.extend(
            [
                "Review the diff in the user message.",
                "Defects only. One terse line per finding. No praise. No summaries.",
                "Severity: critical, major or minor. File and line come from the diff.",
                "Clean diff: return [].",
            ]
        )
        return "\n".join(lines)

    parts = [
        "You are a meticulous code reviewer on a shared review desk.",
        f"Language under review: {context.language}.",
        "The review ruleset for this session, loaded at request time:\n" + ruleset_text,
    ]
    if focus_text:
        parts.append(focus_text)
    parts.extend(
        [
            "How to work:",
            "1. The diff you must review arrives in the user message; read every "
            "hunk before judging anything.",
            "2. The read_ruleset tool re-issues the ruleset above whenever you "
            "need it again mid-review.",
            "3. Report only real, actionable defects as Finding objects; no "
            "praise, no summaries.",
            "4. Take file and line numbers from the diff itself; severity is "
            "critical, major or minor.",
            "5. If something is genuinely ambiguous, say so inside the finding "
            "message instead of guessing.",
            "Output: a list of Finding objects (file, line, severity, message); "
            "return an empty list when the diff is clean.",
        ]
    )
    return "\n\n".join(parts)


def base_reviewer_instructions(
    ctx: RunContextWrapper[ReviewContext], agent: Agent[ReviewContext]
) -> str:
    """Dynamic instructions for the base reviewer, assembled at run time (FR-4)."""
    return _assemble_reviewer_instructions(ctx.context, focus="")


def _focused_instructions(
    focus: str,
) -> Callable[[RunContextWrapper[ReviewContext], Agent[ReviewContext]], str]:
    """Build a clone-specific dynamic-instructions callable.

    All clones share the one assembly function (:func:`_assemble_reviewer_instructions`);
    a clone differs only in the focus text closed over here.
    """

    def instructions(
        ctx: RunContextWrapper[ReviewContext], agent: Agent[ReviewContext]
    ) -> str:
        return _assemble_reviewer_instructions(ctx.context, focus=focus)

    return instructions


# --- Agent construction (models come only from the router) ---


def make_base_reviewer() -> Agent[ReviewContext]:
    """The base reviewer: dynamic instructions, forced ruleset tool, typed output.

    - ``output_type=list[Finding]`` (FR-3): the SDK wraps the list root under a
      ``response`` key for the schema (strict schemas must be objects), while
      ``final_output`` stays a plain Python list.
    - ``tool_choice="required"`` with ``reset_tool_choice=True`` (the default,
      FR-9): the reviewer must call read_ruleset first, then is free to finish
      with structured output.
    - The model comes from the router (``get_model()``); no model name here.
    """
    return Agent(
        name="BaseReviewer",
        instructions=base_reviewer_instructions,
        model=get_model(),
        tools=[read_ruleset],
        output_type=list[Finding],
        model_settings=ModelSettings(
            tool_choice="required",
            temperature=0.2,
            max_tokens=2048,
        ),
    )


_MAX_TOKENS_BY_REVIEWER: dict[str, int] = {
    "SecurityReviewer": 2048,
    "TestsReviewer": 2048,
    "StyleReviewer": 1536,
}
"""Bounded per-clone budgets (NFR-2): style findings are one-liners, so it
gets slightly less room than security/tests detail."""


def make_reviewers() -> dict[str, Agent[ReviewContext]]:
    """The three specialist clones of the base reviewer.

    Each is ``base.clone(...)`` — same object, same tools and output type —
    differing only in focus instructions (via :data:`FOCUS_PROMPTS`) and its
    own bounded :class:`ModelSettings` (NFR-2). Returned as a dict keyed by
    agent name, in fan-out order.
    """
    base = make_base_reviewer()
    reviewers: dict[str, Agent[ReviewContext]] = {}
    for name, focus in FOCUS_PROMPTS.items():
        reviewers[name] = base.clone(
            name=name,
            instructions=_focused_instructions(focus),
            model_settings=ModelSettings(
                tool_choice="required",
                temperature=0.2,
                max_tokens=_MAX_TOKENS_BY_REVIEWER[name],
            ),
        )
    return reviewers


# --- Turn ceiling (FR-9): exceeded turns become a partial review, not a raise ---


def _partial_review_handler(
    handler_input: RunErrorHandlerInput[ReviewContext],
) -> RunErrorHandlerResult:
    """SDK ``error_handlers["max_turns"]`` target (FR-9).

    Converts ``MaxTurnsExceeded`` into an empty structured final output so the
    run completes as a partial review. ``run_reviewer`` watches for this
    handler having fired and flags the outcome ``partial=True``.
    """
    return RunErrorHandlerResult(final_output=[], include_in_history=True)


def _ceiling_reason(reviewer_name: str, max_turns: int) -> str:
    return (
        f"Reviewer '{reviewer_name}' hit the turn ceiling of {max_turns} turns "
        "before finishing; returning a partial review with the findings "
        "gathered so far."
    )


def _invalid_output_handler(
    handler_input: RunErrorHandlerInput[ReviewContext],
) -> RunErrorHandlerResult:
    """SDK ``error_handlers["invalid_final_output"]`` target.

    Converts a ``ModelBehaviorError`` (the model emitted malformed structured
    output for ``output_type=list[Finding]``) into an empty structured final
    output so the run COMPLETES as a partial review instead of raising.
    ``run_reviewer`` watches for this handler having fired and flags the
    outcome ``partial=True``.
    """
    return RunErrorHandlerResult(final_output=[], include_in_history=True)


def _unparseable_reason(reviewer_name: str) -> str:
    return (
        f"Reviewer '{reviewer_name}' produced an unparseable final answer; "
        "reported as a partial review."
    )


def _coerce_findings(raw: Any) -> list[Finding]:
    """Normalise a run's final output into ``list[Finding]``.

    The SDK guarantees a plain list for ``output_type=list[Finding]``; dict
    items (e.g. from a stub or lenient model) are validated into Finding.
    Anything else raises, and the caller converts that into a partial outcome.
    """
    if not isinstance(raw, list):
        raise TypeError(f"expected a list of findings, got {type(raw).__name__}")
    return [item if isinstance(item, Finding) else Finding.model_validate(item) for item in raw]


def _usage_summary(result: Any) -> str | None:
    """Best-effort one-line usage summary from the run context.

    Placeholder plumbing — Task 5 owns real metrics; this only leaves the
    field populated when a Usage object is present.
    """
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return None
    requests = getattr(usage, "requests", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if None in (requests, input_tokens, output_tokens, total_tokens):
        return None
    return (
        f"requests={requests} input_tokens={input_tokens} "
        f"output_tokens={output_tokens} total_tokens={total_tokens}"
    )


async def run_reviewer(
    agent: Agent[ReviewContext],
    diff: str,
    ctx: ReviewContext,
    *,
    max_turns: int = REVIEWER_MAX_TURNS,
    runner: Any = Runner,
    run_config: RunConfig | None = None,
) -> ReviewerOutcome:
    """Run one reviewer against a diff chunk or a full diff, under the ceiling.

    Thin async wrapper around ``runner.run`` (the runner is injectable for
    offline tests) that always:

    - passes the ``ReviewContext`` through ``context=`` (FR-2) — the diff
      itself is the run input, the context never enters prompt text;
    - registers ``error_handlers={"max_turns": ..., "invalid_final_output": ...}``
      (FR-9) so an exceeded ceiling AND a malformed structured answer each
      become a partial review; belt-and-braces ``except MaxTurnsExceeded`` /
      ``except ModelBehaviorError`` back that up for runners that ignore the
      handlers;
    - records per-reviewer latency (``perf_counter``) on the outcome.

    Model/SDK errors other than the ceiling and the malformed final answer
    propagate to the caller (Task 5 wraps them); an unusable final output is
    converted into a partial outcome with a sentence, because a missing
    reviewer outcome would break the fan-out group.
    """
    started = time.perf_counter()
    hit_ceiling = False
    hit_invalid_output = False

    def _on_max_turns(
        handler_input: RunErrorHandlerInput[ReviewContext],
    ) -> RunErrorHandlerResult:
        nonlocal hit_ceiling
        hit_ceiling = True
        return _partial_review_handler(handler_input)

    def _on_invalid_final_output(
        handler_input: RunErrorHandlerInput[ReviewContext],
    ) -> RunErrorHandlerResult:
        nonlocal hit_invalid_output
        hit_invalid_output = True
        return _invalid_output_handler(handler_input)

    try:
        result = await runner.run(
            agent,
            diff,
            context=ctx,
            max_turns=max_turns,
            error_handlers={
                "max_turns": _on_max_turns,
                "invalid_final_output": _on_invalid_final_output,
            },
            run_config=run_config,
        )
    except MaxTurnsExceeded:
        # Belt and braces: a runner that ignores error_handlers must still not
        # leak the ceiling.
        return ReviewerOutcome(
            reviewer_name=agent.name,
            findings=[],
            partial=True,
            partial_reason=_ceiling_reason(agent.name, max_turns),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except ModelBehaviorError:
        # Belt and braces: a runner without invalid_final_output handler
        # support must still not leak a malformed structured answer.
        return ReviewerOutcome(
            reviewer_name=agent.name,
            findings=[],
            partial=True,
            partial_reason=_unparseable_reason(agent.name),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    usage_summary = _usage_summary(result)

    if hit_ceiling:
        try:
            findings = _coerce_findings(result.final_output)
        except Exception:
            findings = []
        return ReviewerOutcome(
            reviewer_name=agent.name,
            findings=findings,
            partial=True,
            partial_reason=_ceiling_reason(agent.name, max_turns),
            usage_summary=usage_summary,
            latency_ms=latency_ms,
        )

    if hit_invalid_output:
        # The invalid_final_output handler resolved the malformed structured
        # answer into an empty list, so the run COMPLETED; salvage nothing and
        # report the partial.
        return ReviewerOutcome(
            reviewer_name=agent.name,
            findings=[],
            partial=True,
            partial_reason=_unparseable_reason(agent.name),
            usage_summary=usage_summary,
            latency_ms=latency_ms,
        )

    try:
        findings = _coerce_findings(result.final_output)
    except Exception as exc:  # noqa: BLE001 - unusable output -> reported partial
        return ReviewerOutcome(
            reviewer_name=agent.name,
            findings=[],
            partial=True,
            partial_reason=(
                f"Reviewer '{agent.name}' finished without a usable findings "
                f"list ({type(exc).__name__}: {exc}); reporting a partial review."
            ),
            usage_summary=usage_summary,
            latency_ms=latency_ms,
        )

    return ReviewerOutcome(
        reviewer_name=agent.name,
        findings=findings,
        usage_summary=usage_summary,
        latency_ms=latency_ms,
    )


def _reviewer_list(
    reviewers: Iterable[Agent[ReviewContext]] | dict[str, Agent[ReviewContext]],
) -> list[Agent[ReviewContext]]:
    """Accept either a sequence of agents or a name-keyed dict from make_reviewers."""
    if isinstance(reviewers, dict):
        return list(reviewers.values())
    return list(reviewers)


async def run_reviewers_concurrently(
    reviewers: Iterable[Agent[ReviewContext]] | dict[str, Agent[ReviewContext]],
    diff: str,
    ctx: ReviewContext,
    *,
    runner: Any = Runner,
    run_config: RunConfig | None = None,
) -> list[ReviewerOutcome]:
    """FR-5: launch every reviewer at once and await the group.

    ``asyncio.gather`` keeps all three model calls in flight, so the wall clock
    approximates the slowest reviewer, not the sum. Outcomes come back in the
    same order as the input reviewers.
    """
    agents = _reviewer_list(reviewers)
    outcomes = await asyncio.gather(
        *(
            run_reviewer(agent, diff, ctx, runner=runner, run_config=run_config)
            for agent in agents
        )
    )
    return list(outcomes)


async def run_reviewers_sequentially(
    reviewers: Iterable[Agent[ReviewContext]] | dict[str, Agent[ReviewContext]],
    diff: str,
    ctx: ReviewContext,
    *,
    runner: Any = Runner,
    run_config: RunConfig | None = None,
) -> list[ReviewerOutcome]:
    """FR-5 comparison helper: the same runs awaited one by one.

    Exists only so the concurrent and sequential wall clocks can be shown side
    by side as evidence that concurrency is real.
    """
    agents = _reviewer_list(reviewers)
    outcomes: list[ReviewerOutcome] = []
    for agent in agents:
        outcomes.append(
            await run_reviewer(agent, diff, ctx, runner=runner, run_config=run_config)
        )
    return outcomes
