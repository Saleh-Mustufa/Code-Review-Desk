"""Specialists: the Desk orchestrator, the Merge tool, Remediation handoff and
the secret-quoting output guardrail (FR-6, FR-8; AD-5, AD-6).

Three voices live here:

- **MergeSpecialist** is exposed to the Desk with ``Agent.as_tool`` — merging
  is a *service the Desk consumes*: the Desk calls ``merge_findings`` with the
  reviewers' findings JSON array and keeps the conversation and the report
  (FR-6, first justification sentence). The tool is *typed*: its parameters
  schema exposes one ``findings`` array (no context wrapper), and an input
  builder hands the array to the specialist as JSON text.
- **RemediationSpecialist** is reached with ``handoff(..., input_type=Escalation)``
  — remediation is a *change of who speaks*: once a critical security finding
  exists the user should be talking to the fixer, so control genuinely
  transfers (FR-6, second sentence). The typed ``Escalation`` input names the
  exact triggering finding, the reviewer who reported it, and why.
- **secret_guardrail** is an output guardrail attached to every voice built in
  this module (Desk, Merge, Remediation). Reviewers get it too at run level —
  Task 5's pipeline attaches it there, because reviewers are built in
  ``review.py``; this module exports the guardrail object for that wiring
  (AD-6: guardrails on every voice).

**FR-8 guardrail design — two layers, no diff content in prompts.**

Layer (a): *generic credential shapes*. The final output is rendered to text
and scanned by :func:`extract_credential_patterns` — ``sk-`` style keys,
``AKIA…`` access-key ids, GitHub/Slack tokens, bearer tokens, quoted
``password = "..."``-style assignments, and long high-entropy tokens. The same
function is the one the pipeline uses to scan diff chunks, so output-side and
diff-side detection share one rule set.

Layer (b): *planted secrets, exact match*. Generic shapes cannot recognise a
credential that fits no known shape, so the pipeline (Task 5) scans the diff
chunks with :func:`extract_credential_patterns` and stores the matches for the
current review via :func:`set_current_secrets`. The guardrail then also checks
the output for *exact occurrences* of those planted patterns.

The matches travel in a :func:`contextvars.ContextVar` (:data:`current_secrets`),
not on the frozen ``ReviewContext`` dataclass (Task 2's shape is fixed) and not
in any prompt: contextvars are copied per asyncio task, so concurrent reviews
cannot cross-contaminate their secret sets, and context content still never
enters prompt text (FR-2). ``output_info`` never carries a full secret — every
reported match is masked middle-out by :func:`_mask_secret`.
"""

from __future__ import annotations

import contextvars
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from agents import (
    Agent,
    FunctionTool,
    GuardrailFunctionOutput,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    output_guardrail,
)
from agents.agent_tool_input import StructuredToolInputBuilderOptions
from agents.handoffs import handoff
from pydantic import BaseModel

from src.intake import ReviewContext
from src.model_config import get_model
from src.review import Finding

__all__ = [
    "DESK_MAX_TURNS",
    "Escalation",
    "MergedFinding",
    "OutputGuardrailTripwireTriggered",
    "ReviewSecrets",
    "SEVERITY_ORDER",
    "current_secrets",
    "dedupe_and_order",
    "extract_credential_patterns",
    "guard_against_secrets",
    "last_escalation",
    "make_desk_agent",
    "make_merge_tool",
    "merge_specialist",
    "remediation_specialist",
    "reset_current_secrets",
    "secret_guardrail",
    "set_current_secrets",
    "tag_findings",
]

# --- Merge machinery (FR-6): the deterministic rule the tool performs ---


SEVERITY_ORDER: dict[str, int] = {"critical": 0, "major": 1, "minor": 2}
"""Sort rank per severity: critical findings come first in merged output."""


class MergedFinding(Finding):
    """A Finding plus the reviewers that reported it.

    ``Finding``'s fields are kept identical (file, line, severity, message);
    ``sources`` records which reviewers reported the finding and defaults to an
    empty list so single-source findings work unannotated.
    """

    sources: list[str] = []


def tag_findings(findings: list[Finding], reviewer: str) -> list[MergedFinding]:
    """Tag every finding with the reviewer that produced it.

    The pipeline calls this per reviewer before merging, so
    :func:`dedupe_and_order` can annotate ``sources``; plain ``Finding``
    objects carry no provenance of their own.
    """
    return [
        MergedFinding(
            file=f.file,
            line=f.line,
            severity=f.severity,
            message=f.message,
            sources=[reviewer],
        )
        for f in findings
    ]


def dedupe_and_order(findings: list[Finding]) -> list[MergedFinding]:
    """Merge overlapping findings into a deterministic, severity-ordered list.

    Documented rule (kept mechanical so the MergeSpecialist's prompt states the
    same rule):

    - Two findings are **duplicates** when they share the same *file*, *line*
      *and severity*: keep the first message, drop the rest, and union the
      ``sources``. Near-identical messages at *different* lines are not
      collapsed — the rule stays positional, not textual.
    - Findings sharing *file* and *line* but **differing in severity** are one
      issue reported twice: keep the highest severity and merge the distinct
      messages (original order, joined with ``"; "``).
    - Sort by severity (critical -> major -> minor), then file name, then line.
    - ``sources`` flow through from pre-tagged :class:`MergedFinding` inputs
      (see :func:`tag_findings`) and are otherwise empty.

    Pure function: no model, no I/O, same input -> same output.
    """
    merged: list[MergedFinding] = []
    positions: dict[tuple[str, int], int] = {}

    for finding in findings:
        key = (finding.file, finding.line)
        existing_index = positions.get(key)
        if existing_index is None:
            positions[key] = len(merged)
            merged.append(_as_merged(finding))
            continue

        existing = merged[existing_index]
        existing.sources = _union(existing.sources, _as_merged(finding).sources)
        if existing.severity == finding.severity:
            continue  # duplicate: keep the first message, sources already unioned
        # Same issue, two severities: keep the highest, merge distinct messages.
        if SEVERITY_ORDER[finding.severity] < SEVERITY_ORDER[existing.severity]:
            existing.severity = finding.severity
        existing.message = _merge_messages(existing.message, finding.message)

    return sorted(
        merged,
        key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line),
    )


def _as_merged(finding: Finding) -> MergedFinding:
    """View any Finding as a MergedFinding, carrying sources when pre-tagged."""
    if isinstance(finding, MergedFinding):
        return finding
    return MergedFinding(
        file=finding.file,
        line=finding.line,
        severity=finding.severity,
        message=finding.message,
        sources=[],
    )


def _union(left: list[str], right: list[str]) -> list[str]:
    """Order-preserving union of two source lists."""
    return left + [item for item in right if item not in left]


def _merge_messages(first: str, second: str) -> str:
    """Merge two messages of one collapsed finding, dropping exact repeats."""
    if first.strip() == second.strip():
        return first
    return f"{first}; {second}"


# --- Credential-shape detection (FR-8, layer a): one rule set, two uses ---

_CREDENTIAL_SHAPES: tuple[re.Pattern[str], ...] = (
    # sk-... style keys (OpenAI-shaped and friends).
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    # AWS access key ids.
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    # GitHub tokens: classic ghp_/gho_/ghu_/ghs_/ghr_ and fine-grained github_pat_.
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # Slack tokens.
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    # Quoted credential assignments: password = "..." / api_key: "..." etc.
    re.compile(
        r"\b(?:password|passwd|pwd|api[_-]?key|apikey|secret|access[_-]?key|"
        r"secret[_-]?key|client[_-]?secret|auth[_-]?token|access[_-]?token|token)"
        r"\s*[:=]\s*[\"'][^\"']{1,}[\"']",
        re.IGNORECASE,
    ),
    # Bearer tokens in headers.
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
)

_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")
"""A long bare token; only credential-shaped when the entropy heuristic agrees."""

_LONG_TOKEN_MIN_ENTROPY = 3.5
"""Minimum Shannon entropy (bits/char) for a long bare token to look secret-like.

Random tokens from a 60+ symbol alphabet measure ~4.5-5.7 bits/char; the 3.5
floor rejects repeated words and padded identifiers. The entropy check alone
still admits lowercase-hex strings (git SHAs measure ~3.7), so a token must
*also* use at least 3 of the 4 character classes (lower, upper, digit, ``-``/``
_``) — SHAs use two. Deliberate tradeoff: an all-lowercase secret with no
separators can slip past layer (a); layer (b) exact-matches whatever the
pipeline planted from the diff, so real leaked values are still caught.
"""

_CHAR_CLASSES: tuple[re.Pattern[str], ...] = (
    re.compile(r"[a-z]"),
    re.compile(r"[A-Z]"),
    re.compile(r"[0-9]"),
    re.compile(r"[-_]"),
)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of *text* in bits per character."""
    if not text:
        return 0.0
    total = len(text)
    return -sum(
        (count / total) * math.log2(count / total) for count in Counter(text).values()
    )


def _token_is_credential_shaped(token: str) -> bool:
    """Heuristic for a long bare token: high entropy plus mixed char classes."""
    if _shannon_entropy(token) < _LONG_TOKEN_MIN_ENTROPY:
        return False
    classes = sum(1 for pattern in _CHAR_CLASSES if pattern.search(token))
    return classes >= 3


def extract_credential_patterns(text: str) -> list[str]:
    """Return credential-shaped strings found in *text*, first-seen order.

    Shared rule set for both FR-8 uses: the pipeline runs this over diff chunks
    to plant layer-(b) patterns, and the guardrail runs it over agent output for
    layer (a). Matches overlap-checked (most specific shape wins) and deduped.

    The returned strings may embed real secrets — callers must treat them as
    sensitive and mask before display (:func:`_mask_secret`).
    """
    found: list[tuple[int, int, str]] = []
    for pattern in _CREDENTIAL_SHAPES:
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), match.group(0)))

    for match in _LONG_TOKEN.finditer(text):
        start, end = match.span()
        if any(start < other_end and other_start < end for other_start, other_end, _ in found):
            continue  # already covered by a more specific shape
        if _token_is_credential_shaped(match.group(0)):
            found.append((start, end, match.group(0)))

    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, match_text in sorted(found, key=lambda item: item[0]):
        if match_text not in seen:
            seen.add(match_text)
            ordered.append(match_text)
    return ordered


def _mask_secret(match_text: str) -> str:
    """Mask a credential match middle-out, safe for ``output_info`` and logs."""
    if len(match_text) <= 8:
        return match_text[:2] + "***"
    return f"{match_text[:4]}***{match_text[-2:]}"


# --- Planted secrets for the current review (FR-8, layer b) ---


@dataclass
class ReviewSecrets:
    """Credential-shaped strings extracted from the current review's diff.

    Populated by the pipeline (Task 5) via :func:`set_current_secrets` before a
    Desk run and cleared afterwards; read only by the output guardrail. Never
    serialised into prompts, ledgers or reports.
    """

    patterns: list[str] = field(default_factory=list)


current_secrets: contextvars.ContextVar[ReviewSecrets | None] = contextvars.ContextVar(
    "current_secrets", default=None
)
"""Per-asyncio-task registry of the current review's planted secret patterns.

ContextVar (not a module-level mutable object, not a ReviewContext field) so
concurrent reviews each see their own set — asyncio copies the context per
task — and so no secret can reach prompt text (FR-2) or another session.
"""


def set_current_secrets(patterns: list[str]) -> contextvars.Token[ReviewSecrets | None]:
    """Plant the current review's secret patterns; returns a reset token.

    Pipeline usage: ``token = set_current_secrets(extract_credential_patterns(diff))``
    around the Desk run, then ``reset_current_secrets(token)`` in ``finally``.
    """
    return current_secrets.set(ReviewSecrets(patterns=list(patterns)))


def reset_current_secrets(token: contextvars.Token[ReviewSecrets | None]) -> None:
    """Restore the previous secrets registry (clears the planting)."""
    current_secrets.reset(token)


# --- The output guardrail (FR-8, AD-6) ---


def _render_output_text(output: object) -> str:
    """Render any agent final output to the text the guardrail scans.

    ``list`` outputs (``list[Finding]`` / ``list[MergedFinding]``) render as
    the join of their items' ``message`` fields; strings render as themselves;
    anything else falls back to ``str``.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        parts: list[str] = []
        for item in output:
            message = getattr(item, "message", None)
            parts.append(message if isinstance(message, str) else str(item))
        return "\n".join(parts)
    if isinstance(output, BaseModel):
        message = getattr(output, "message", None)
        return message if isinstance(message, str) else output.model_dump_json()
    return str(output)


@output_guardrail
async def guard_against_secrets(
    ctx: RunContextWrapper[ReviewContext], agent: Agent[Any], output: Any
) -> GuardrailFunctionOutput:
    """Refuse any output that quotes a credential (FR-8).

    Runs on an agent's final output. Two layers: (a) generic credential shapes
    in the rendered output text; (b) exact occurrences of the patterns the
    pipeline planted for this review in :data:`current_secrets`. On a trip the
    runner raises ``OutputGuardrailTripwireTriggered`` (caught by the pipeline,
    rendered as a refusal); ``output_info`` names the layers and the masked
    matches — never a full secret.
    """
    del ctx  # planted patterns arrive via the contextvar, not the context object
    text = _render_output_text(output)

    shape_matches = extract_credential_patterns(text) if text else []
    registry = current_secrets.get()
    planted_matches = (
        [pattern for pattern in registry.patterns if pattern and pattern in text]
        if registry
        else []
    )

    layers: list[str] = []
    if shape_matches:
        layers.append("generic-shape")
    if planted_matches:
        layers.append("planted-secret")

    if not layers:
        return GuardrailFunctionOutput(
            output_info={
                "guardrail": "secret_guardrail",
                "clean": True,
                "reason": "No credential-shaped text detected in the output.",
            },
            tripwire_triggered=False,
        )

    masked = [_mask_secret(match) for match in dict.fromkeys(shape_matches + planted_matches)]
    return GuardrailFunctionOutput(
        output_info={
            "guardrail": "secret_guardrail",
            "clean": False,
            "layers": layers,
            "matches": masked,
            "reason": (
                "Output quotes credential-shaped text "
                f"(layers: {', '.join(layers)}; {len(masked)} masked match(es)); "
                "the run is refused rather than echoing the secret."
            ),
        },
        tripwire_triggered=True,
    )


secret_guardrail = guard_against_secrets
"""The guardrail object to attach: ``Agent(output_guardrails=[secret_guardrail])``.

Alias of :func:`guard_against_secrets` (which the ``@output_guardrail``
decorator already turned into an ``OutputGuardrail``); the alias is the name
used in agent wiring and Task 5's pipeline.
"""


# --- MergeSpecialist (FR-6): the Desk consumes it as a tool ---


class MergeToolInput(BaseModel):
    """Typed input for the ``merge_findings`` tool: the findings JSON array.

    Keeps the tool's schema to a single ``findings`` array (no context wrapper,
    FR-2); the Desk model passes the reviewers' findings verbatim.
    """

    findings: list[Finding]


def _merge_tool_input_builder(options: StructuredToolInputBuilderOptions) -> str:
    """Build the MergeSpecialist's user message: the findings as a JSON array."""
    params = options.get("params") or {}
    findings = params.get("findings", []) if isinstance(params, dict) else []
    return json.dumps(findings, indent=2)


_MERGE_INSTRUCTIONS = """You are the merge specialist on a code review desk.
The user message contains reviewer findings as a JSON array of objects with \
file, line, severity (critical, major or minor) and message — findings may \
also carry a "sources" list naming the reviewers that reported them.

Merge them deterministically:
1. Findings sharing the same file, line AND severity are duplicates: keep the \
first message, drop the rest, and union their "sources".
2. Findings sharing the same file and line but differing in severity are one \
issue: keep the highest severity and merge the distinct messages, joined with \
"; ".
3. Order the merged list: critical first, then major, then minor; within the \
same severity by file name, then by line number.
4. Annotate every merged finding with "sources": the reviewers that reported \
it (empty list when unknown).
5. Never invent, reword beyond the message-merge rule, or drop findings. \
Return exactly the merged list."""

merge_specialist: Agent[ReviewContext] = Agent(
    name="MergeSpecialist",
    model=get_model(),
    output_type=list[MergedFinding],
    model_settings=ModelSettings(temperature=0.0, max_tokens=2048),
    instructions=_MERGE_INSTRUCTIONS,
    output_guardrails=[secret_guardrail],
)
"""Dedupe + severity-order service, called by the Desk via ``as_tool`` (FR-6).

``output_type=list[MergedFinding]`` so the tool result flowing back to the Desk
is the structured merged list. Bounded settings (NFR-2); temperature 0.0 — the
merge is a deterministic function the model merely performs.
"""


def make_merge_tool() -> FunctionTool:
    """The Desk's ``merge_findings`` tool: MergeSpecialist via ``as_tool``.

    Typed parameters (``findings`` array, no context wrapper) and an input
    builder that hands the array to the specialist as JSON text. The nested run
    carries this module's output guardrail, so a tool-level result is checked
    too (AD-6).
    """
    return merge_specialist.as_tool(
        tool_name="merge_findings",
        tool_description=(
            "Deduplicate and severity-order reviewer findings; pass the "
            "findings as a JSON array"
        ),
        parameters=MergeToolInput,
        input_builder=_merge_tool_input_builder,
    )


# --- RemediationSpecialist (FR-6): reached by typed handoff ---


_REMEDIATION_INSTRUCTIONS = """You are a remediation engineer on a code review desk.
A handoff delivers an Escalation naming one CRITICAL security finding: the \
exact triggering finding (file, line, severity, message), the reviewer who \
found it, and why it was escalated.

Produce, in plain text:
1. A concrete proposed fix as a fenced ```diff block containing a unified diff \
against the affected file(s).
2. Two to four sentences of rationale: what the vulnerability is, why the \
patch closes it, and any follow-up the developer should do.

Hard rules:
- Never include real credentials in your patch or prose. If the finding quotes \
a secret, refer to it redacted (for example <redacted-api-key>) and replace \
the hardcoded value with an environment-variable lookup or a secret manager \
call.
- Patch only what the finding describes; do not refactor unrelated code.
- No preamble or meta-commentary; the diff plus rationale is the whole answer."""

remediation_specialist: Agent[ReviewContext] = Agent(
    name="RemediationSpecialist",
    model=get_model(),
    output_type=str,
    model_settings=ModelSettings(temperature=0.2, max_tokens=2048),
    instructions=_REMEDIATION_INSTRUCTIONS,
    output_guardrails=[secret_guardrail],
)
"""The fixer the Desk transfers to on critical security findings (FR-6).

Reached only via the Desk's handoff with typed ``Escalation`` input; its final
answer (a proposed patch) is the run's final output, so its guardrail is the
one that protects it (AD-6).
"""


# --- Escalation: the typed handoff input ---


class Escalation(BaseModel):
    """Typed input the Desk provides when transferring to Remediation.

    Names WHICH finding triggered the handoff — the triggering finding itself,
    the reviewer who reported it, and one sentence of reason — so the
    remediation engineer starts from the exact issue, not a paraphrase.
    """

    finding: Finding
    reviewer: str
    reason: str


last_escalation: contextvars.ContextVar[Escalation | None] = contextvars.ContextVar(
    "last_escalation", default=None
)
"""The most recent Escalation accepted by the Desk's handoff.

Bookkeeping for the pipeline/UI (Task 5 renders the ``RemediationOffered``
event from it); asyncio-task scoped like :data:`current_secrets`. Never
rendered into prompts.
"""


async def _record_escalation(
    ctx: RunContextWrapper[ReviewContext], escalation: Escalation
) -> None:
    """``on_handoff`` callback: remember the escalation when the Desk transfers."""
    del ctx
    last_escalation.set(escalation)


# --- The Desk (FR-6, FR-9): coordinator with merge tool + remediation handoff ---


DESK_MAX_TURNS = 8
"""Turn ceiling for the Desk run (FR-9).

Typical Desk run: merge tool call + structured continuation + optional handoff
+ final summary = 4-6 turns. 8 gives headroom for one extra model nudge without
letting a confused run loop against the rate-limit budget.
"""


def _desk_instructions(
    ctx: RunContextWrapper[ReviewContext], agent: Agent[ReviewContext]
) -> str:
    """Dynamic Desk instructions, assembled at request time.

    Reads language and strictness from the context like the reviewers do;
    ``context.repo`` is deliberately unread so grepping prompts for a
    repository name finds nothing (FR-2).
    """
    del agent
    strict = ctx.context.strictness.strip().lower() == "strict"
    framing = (
        "Be terse: header, counts, findings, nothing else."
        if strict
        else "Tone: factual and brief; no praise."
    )
    return "\n".join(
        [
            "You are the review desk coordinator for code reviews.",
            f"Language under review: {ctx.context.language}. "
            f"Strictness: {ctx.context.strictness}.",
            "",
            "Workflow:",
            "1. The reviewers' findings arrive as a JSON array in the user "
            "message.",
            "2. Call the merge_findings tool exactly once, passing the full "
            "JSON array as its `findings` argument. Do not filter or edit the "
            "array yourself.",
            "3. When the merged, severity-ordered list returns, decide:",
            "   - If any merged finding is a CRITICAL security finding "
            "(credentials, injection, auth, TLS), use the transfer tool to "
            "hand off to the remediation engineer. The handoff input must name "
            "the exact triggering finding (file, line, severity, message), the "
            "reviewer who reported it, and one sentence on why it needs a "
            "patch.",
            "   - Otherwise, write the final report summary for the user.",
            "4. The summary is short: a header line, counts by severity, then "
            "the top findings (critical and major first, one line each).",
            "5. Never reveal secrets: findings must not quote credential "
            "values; refer to them redacted.",
            "",
            framing,
        ]
    )


def make_desk_agent() -> Agent[ReviewContext]:
    """The Desk orchestrator: merge tool, typed remediation handoff, guardrail.

    - ``tools=[merge_findings]`` — the MergeSpecialist as a tool, so the Desk
      keeps the conversation and the report (AD-5);
    - ``handoffs=[handoff(remediation_specialist, input_type=Escalation)]`` —
      control genuinely transfers to the fixer on critical security findings
      (AD-5);
    - ``output_guardrails=[secret_guardrail]`` — the Desk's own final summary is
      checked before it reaches the user (AD-6);
    - bounded ``ModelSettings`` (NFR-2) and the router model — no model name
      here.
    """
    return Agent(
        name="ReviewDesk",
        model=get_model(),
        output_type=str,
        model_settings=ModelSettings(temperature=0.2, max_tokens=1024),
        instructions=_desk_instructions,
        tools=[make_merge_tool()],
        handoffs=[
            handoff(
                remediation_specialist,
                on_handoff=_record_escalation,
                input_type=Escalation,
                tool_description_override=(
                    "Hand off to the remediation engineer when a CRITICAL "
                    "security finding needs a proposed patch; provide the "
                    "finding, the reviewer who found it, and why"
                ),
            )
        ],
        output_guardrails=[secret_guardrail],
    )
