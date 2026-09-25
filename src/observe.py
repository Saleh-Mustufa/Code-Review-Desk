"""Observability: run-level hooks, agent-level probe hooks, the ledger runner
and tracing setup (FR-10, FR-11, FR-13; NFR-3).

Four voices live here:

- **MetricsRunHooks** is a *run-level* hooks object (``RunHooks[ReviewContext]``).
  It records per-agent latency and the REAL token usage read from the run
  context's ``Usage`` (requests, input/output tokens — never estimated) into a
  shared ``dict[str, ReviewerStats]``. The pipeline hands one instance per
  review to the runner, and prefers the per-run usage it collects for the
  report footer (FR-10).
- **SecurityProbeHooks** is an *agent-level* hooks object attached to exactly
  ONE reviewer (the SecurityReviewer, via :func:`attach_agent_hooks`). Agent
  hooks see per-event detail — each LLM start/end, each tool call — that
  run-level hooks aggregate away (FR-10, second half).
- **LedgerRunner** is a ``Runner`` subclass whose instance ``run`` wraps the
  standard run, times it, and appends exactly ONE JSON line per run to
  ``ledger.jsonl`` (FR-11). It is instantiated ONCE at startup as
  :data:`LEDGER_RUNNER` at the bottom of this module; the pipeline falls back
  to it via ``getattr``, so **removing that one registration line is the only
  change needed to switch the ledger off** — no agent definition mentions it.
- :func:`setup_tracing` exports traces under ``OPENAI_API_KEY``; when the key
  is missing it disables tracing and returns one sentence for the caller to
  log — the app still runs (FR-13, NFR-3).

Ledger hygiene (NFR-1): a ledger line carries timestamps, ids, counts and
token totals only — never the diff, never tool outputs, never secrets.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents import (
    AgentHooks,
    Runner,
    RunHooks,
    set_tracing_disabled,
    set_tracing_export_api_key,
)

from src.intake import ReviewContext
from src.model_config import current_model_name

__all__ = [
    "LEDGER_RUNNER",
    "LedgerRunner",
    "MetricsRunHooks",
    "ReviewerStats",
    "SecurityProbeHooks",
    "append_ledger_line",
    "attach_agent_hooks",
    "current_run_hooks",
    "default_ledger_path",
    "setup_tracing",
]

_LOGGER = logging.getLogger(__name__)

WORKFLOW_NAME = "Code Review Desk"
"""The workflow_name every review trace and every RunConfig carries (FR-13)."""


# --- Per-reviewer stats (FR-10) ---


@dataclass
class ReviewerStats:
    """One reviewer's measured footprint: latency plus REAL token usage.

    ``tokens_in``/``tokens_out``/``requests`` come from the run context's
    ``Usage`` (never estimated). ``model`` is a name string from router
    introspection when cheaply available, else ``None`` — never hardcoded.
    """

    reviewer_name: str
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    requests: int = 0
    model: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


# --- Run-level hooks (FR-10, first half) ---


class MetricsRunHooks(RunHooks[ReviewContext]):
    """Run-level hooks recording per-agent latency and real usage.

    Takes a shared ``dict[str, ReviewerStats]`` keyed by agent name.
    ``on_agent_start`` records the start time; ``on_agent_end`` records the
    latency and the REAL usage from the run context (``context.usage`` — the
    run's own ``RunContextWrapper.usage``, which is complete at agent end
    because each reviewer is a separate ``Runner.run`` call).

    The pipeline also reads per-result usage directly where it can; these
    hooks prove the run-level lifecycle fires and are the one source that
    works uniformly through the ledger runner.
    """

    def __init__(self, stats: dict[str, ReviewerStats] | None = None) -> None:
        self.stats: dict[str, ReviewerStats] = stats if stats is not None else {}
        self._started: dict[str, float] = {}

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        name = getattr(agent, "name", "unknown")
        self._started[name] = time.perf_counter()
        self.stats.setdefault(name, ReviewerStats(reviewer_name=name))

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        name = getattr(agent, "name", "unknown")
        stats = self.stats.setdefault(name, ReviewerStats(reviewer_name=name))
        started = self._started.get(name)
        if started is not None:
            stats.latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(context, "usage", None)
        if usage is not None:
            stats.tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
            stats.tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
            stats.requests = int(getattr(usage, "requests", 0) or 0)


# --- Agent-level hooks (FR-10, second half): per-event probe on ONE reviewer ---


class SecurityProbeHooks(AgentHooks):
    """Agent-level hooks on exactly one reviewer, logging every event.

    These see the per-event detail run-level hooks aggregate away: each LLM
    start/end and each tool call of THIS agent. Events are collected as short
    strings (names only — never tool outputs, prompts or findings, so no
    secret or diff content can land here) into ``self.events`` for the report.
    """

    def __init__(self, events: list[str] | None = None, *, max_events: int = 100) -> None:
        self.events: list[str] = events if events is not None else []
        self._max_events = max_events

    def _record(self, event: str) -> None:
        if len(self.events) < self._max_events:
            self.events.append(event)

    async def on_start(self, context: Any, agent: Any) -> None:
        self._record(f"start:{getattr(agent, 'name', 'unknown')}")

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        self._record(f"end:{getattr(agent, 'name', 'unknown')}")

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        self._record(f"handoff:{getattr(source, 'name', '?')}->{getattr(agent, 'name', '?')}")

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self._record(f"tool_start:{getattr(tool, 'name', 'unknown_tool')}")

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self._record(f"tool_end:{getattr(tool, 'name', 'unknown_tool')}")

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        self._record(f"llm_start:{getattr(agent, 'name', 'unknown')}")

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self._record(f"llm_end:{getattr(agent, 'name', 'unknown')}")


def attach_agent_hooks(reviewers: Any, hooks: SecurityProbeHooks | None = None) -> SecurityProbeHooks:
    """Attach agent-level hooks to EXACTLY ONE reviewer (FR-10).

    The SecurityReviewer gets the probe; if no such agent is present the first
    reviewer takes it, so exactly one agent is ever probed. Returns the hooks
    object so the caller can hand ``.events`` to the report.
    """
    probe = hooks if hooks is not None else SecurityProbeHooks()
    agents = list(reviewers.values()) if isinstance(reviewers, dict) else list(reviewers)
    target = next(
        (agent for agent in agents if getattr(agent, "name", "") == "SecurityReviewer"),
        agents[0] if agents else None,
    )
    if target is not None:
        target.hooks = probe
    return probe


# --- The ledger runner (FR-11) ---


current_run_hooks: contextvars.ContextVar[RunHooks[Any] | None] = contextvars.ContextVar(
    "current_run_hooks", default=None
)
"""Run-level hooks for the next runs dispatched through :class:`LedgerRunner`.

The pipeline sets this (per review, per asyncio task) before fanning out; the
ledger runner injects it into each ``run(...)`` call that did not pass its own
hooks. ContextVar scoping keeps concurrent reviews from crossing hooks.
"""


def default_ledger_path() -> Path:
    """``ledger.jsonl`` at the project root, or the LEDGER_PATH override."""
    env_path = os.environ.get("LEDGER_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "ledger.jsonl"


def _utc_now() -> str:
    """ISO8601 UTC timestamp with a trailing Z."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def append_ledger_line(
    path: str | Path,
    *,
    ts: str,
    request_id: str,
    agent: str,
    ms: int,
    findings: int,
    model: str | None = None,
    tokens: int | None = None,
) -> None:
    """Append ONE JSON line to the ledger file (the FR-11 line format).

    Module-level so it is unit-testable without a real run. Keys: ``ts``,
    ``request_id``, ``agent``, ``ms``, ``findings``, plus ``model`` and
    ``tokens`` when known. Never receives — and never writes — diff text,
    tool output or secret material.
    """
    line: dict[str, Any] = {
        "ts": ts,
        "request_id": request_id,
        "agent": agent,
        "ms": ms,
        "findings": findings,
    }
    if model:
        line["model"] = model
    if tokens is not None:
        line["tokens"] = tokens
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def _run_config_value(kwargs: dict[str, Any], key: str) -> Any:
    """Read one field off the run_config kwarg (RunConfig object or dict)."""
    run_config = kwargs.get("run_config")
    if isinstance(run_config, dict):
        return run_config.get(key)
    return getattr(run_config, key, None)


def _model_name_for_ledger(kwargs: dict[str, Any], starting_agent: Any) -> str | None:
    """Best-effort model name for the ledger: run_config model, else the agent's.

    Names come from router introspection (:func:`current_model_name`) — no
    hardcoded model names anywhere.
    """
    model = _run_config_value(kwargs, "model")
    if model is None:
        model = getattr(starting_agent, "model", None)
    if model is None:
        return None
    try:
        return current_model_name(model)
    except Exception:  # noqa: BLE001 - the ledger never breaks a review
        return None


def _last_agent_name(result: Any) -> str:
    return getattr(getattr(result, "last_agent", None), "name", "unknown")


def _findings_count(result: Any) -> int:
    """List outputs count their items; string outputs count 0."""
    output = getattr(result, "final_output", None)
    return len(output) if isinstance(output, list) else 0


def _total_tokens(result: Any) -> int | None:
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return None
    try:
        return int(getattr(usage, "total_tokens", 0) or 0)
    except (TypeError, ValueError):
        return None


class LedgerRunner(Runner):
    """A ``Runner`` that appends exactly one JSON line per run (FR-11).

    ``Runner`` itself exposes ``run`` as a *classmethod* delegating to the SDK's
    default agent runner; this subclass adds an instance ``run`` (so the
    pipeline calls ``LEDGER_RUNNER.run(agent, ...)``) that:

    1. injects the task's :data:`current_run_hooks` when the caller passed none,
    2. awaits the standard run via ``super().run(...)``,
    3. appends one ledger line (ts, request_id from the run config's
       ``group_id``, final agent name, wall-clock ms, findings count, model
       name, total tokens).

    The record step is best-effort: a ledger failure is logged, never raised
    into the review.
    """

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.ledger_path = (
            Path(ledger_path) if ledger_path is not None else default_ledger_path()
        )

    async def run(
        self,
        starting_agent: Any,
        input: Any,
        **kwargs: Any,
    ) -> Any:
        hooks = current_run_hooks.get()
        if hooks is not None:
            kwargs.setdefault("hooks", hooks)
        request_id = _run_config_value(kwargs, "group_id") or ""
        model_name = _model_name_for_ledger(kwargs, starting_agent)
        started = time.perf_counter()
        result = await super().run(starting_agent, input, **kwargs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            self._record(
                request_id=request_id,
                agent=_last_agent_name(result),
                ms=elapsed_ms,
                findings=_findings_count(result),
                model=model_name,
                tokens=_total_tokens(result),
            )
        except Exception as exc:  # noqa: BLE001 - the ledger never breaks a review
            _LOGGER.warning("ledger append failed (%s: %s)", type(exc).__name__, exc)
        return result

    def _record(
        self,
        *,
        request_id: str,
        agent: str,
        ms: int,
        findings: int,
        model: str | None,
        tokens: int | None,
    ) -> None:
        append_ledger_line(
            self.ledger_path,
            ts=_utc_now(),
            request_id=request_id,
            agent=agent,
            ms=ms,
            findings=findings,
            model=model,
            tokens=tokens,
        )


LEDGER_RUNNER = LedgerRunner()
"""The one ledger runner for this process, registered once at startup (FR-11).

The pipeline uses it as its default runner, so every run of every review is
ledgered. To switch the ledger off, delete this line (and nothing else): the
pipeline looks it up with ``getattr`` and falls back to the plain ``Runner``.
"""

# --- Tracing setup (FR-13, NFR-3) ---


def setup_tracing() -> str | None:
    """Export traces under OPENAI_API_KEY; disable with a sentence when absent.

    Returns ``None`` when tracing is configured, or one sentence the caller
    logs — the app always runs. Never raises; never echoes the key.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        try:
            set_tracing_export_api_key(key)
            return None
        except Exception as exc:  # noqa: BLE001 - tracing setup never raises
            set_tracing_disabled(True)
            return (
                f"Trace export could not be configured ({type(exc).__name__}); "
                "tracing is disabled and the review still runs."
            )
    set_tracing_disabled(True)
    return (
        "OPENAI_API_KEY is not set, so trace export is disabled — reviews run "
        "normally, just without exported traces."
    )
