"""Unit tests for src/observe.py — offline, no network, no real model runs.

The ledger line format and one-line-per-run behaviour are the load-bearing
contracts (FR-11), so they are tested directly: the runner plumbing is
exercised by monkeypatching the SDK's default agent runner with a fake, and
the ledger append with a spy, keeping everything offline.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import agents.run
import pytest
from agents import RunConfig, Runner
from agents.usage import Usage

from src import observe
from src.intake import ReviewContext
from src.observe import (
    LEDGER_RUNNER,
    LedgerRunner,
    MetricsRunHooks,
    ReviewerStats,
    SecurityProbeHooks,
    append_ledger_line,
    attach_agent_hooks,
    default_ledger_path,
    setup_tracing,
)
from src.review import make_reviewers

# --- Fixtures and fakes --------------------------------------------------------


class FakeAgent:
    """Just enough agent for the observer: a name and a model slot."""

    def __init__(self, name: str = "BaseReviewer") -> None:
        self.name = name
        self.model = None


class FakeResult:
    """A RunResult-shaped stub: final output, last agent, run-context usage."""

    def __init__(
        self,
        final_output: object,
        last_agent: FakeAgent,
        usage: Usage | None = None,
    ) -> None:
        self.final_output = final_output
        self.last_agent = last_agent
        self.context_wrapper = SimpleNamespace(usage=usage)


class FakeAgentRunner:
    """Stands in for the SDK's default agent runner (no model traffic)."""

    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def run(self, starting_agent: object, input: object, **kwargs: object) -> FakeResult:
        self.calls.append({"agent": starting_agent, "input": input, "kwargs": kwargs})
        return self.result


@pytest.fixture
def ctx() -> ReviewContext:
    return ReviewContext(repo="demo", language="python", ruleset_id="default")


def make_usage() -> Usage:
    """Real, non-estimated token counts as they'd arrive from a model call."""
    return Usage(requests=2, input_tokens=540, output_tokens=210, total_tokens=750)


# --- append_ledger_line: the FR-11 line format ---------------------------------


def test_append_ledger_line_writes_one_json_line(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_ledger_line(
        path,
        ts="2026-01-01T00:00:00.000Z",
        request_id="rev_ab12cd34",
        agent="SecurityReviewer",
        ms=1234,
        findings=3,
        model="gemini-2.5-flash",
        tokens=750,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # exactly ONE line per run
    line = json.loads(lines[0])
    assert line == {
        "ts": "2026-01-01T00:00:00.000Z",
        "request_id": "rev_ab12cd34",
        "agent": "SecurityReviewer",
        "ms": 1234,
        "findings": 3,
        "model": "gemini-2.5-flash",
        "tokens": 750,
    }


def test_append_ledger_line_omits_optional_fields_when_unknown(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_ledger_line(
        path,
        ts="2026-01-01T00:00:00.000Z",
        request_id="",
        agent="ReviewDesk",
        ms=10,
        findings=0,
    )
    line = json.loads(path.read_text(encoding="utf-8"))
    assert "model" not in line and "tokens" not in line
    assert line["findings"] == 0  # string outputs count zero findings


def test_append_ledger_line_appends_across_runs(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    for agent in ("SecurityReviewer", "TestsReviewer"):
        append_ledger_line(
            path,
            ts="2026-01-01T00:00:00.000Z",
            request_id="rev_x",
            agent=agent,
            ms=5,
            findings=1,
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # one line PER RUN, appended not overwritten
    assert [json.loads(line)["agent"] for line in lines] == [
        "SecurityReviewer",
        "TestsReviewer",
    ]


def test_default_ledger_path_is_project_root_or_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LEDGER_PATH", raising=False)
    assert default_ledger_path().name == "ledger.jsonl"
    monkeypatch.setenv("LEDGER_PATH", str(tmp_path / "elsewhere.jsonl"))
    assert default_ledger_path() == tmp_path / "elsewhere.jsonl"


# --- LedgerRunner: one line per run, runner plumbing intact ---------------------


def test_ledger_runner_is_a_runner_registered_once() -> None:
    assert isinstance(LEDGER_RUNNER, LedgerRunner)
    assert isinstance(LEDGER_RUNNER, Runner)  # the pipeline can run it as a runner


@pytest.mark.asyncio
async def test_ledger_runner_appends_one_line_per_run(monkeypatch, tmp_path) -> None:
    agent = FakeAgent("SecurityReviewer")
    result = FakeResult([1, 2, 3], agent, make_usage())
    fake_runner = FakeAgentRunner(result)
    monkeypatch.setattr(agents.run, "DEFAULT_AGENT_RUNNER", fake_runner)

    ledger = LedgerRunner(ledger_path=tmp_path / "ledger.jsonl")
    run_config = RunConfig(
        workflow_name="Code Review Desk",
        group_id="rev_ab12cd34",
        model=SimpleNamespace(model="gemini-test-model"),
    )
    out = await ledger.run(agent, "the diff text", context=None, run_config=run_config)

    assert out is result  # the run result passes through untouched
    assert len(fake_runner.calls) == 1  # the underlying run happened exactly once
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert line["request_id"] == "rev_ab12cd34"  # from RunConfig group_id
    assert line["agent"] == "SecurityReviewer"  # the run's last agent
    assert line["findings"] == 3  # list output -> item count
    assert line["model"] == "gemini-test-model"  # introspected, not hardcoded
    assert line["tokens"] == 750  # REAL total from the run context usage
    assert isinstance(line["ms"], int) and line["ms"] >= 0
    assert line["ts"].endswith("Z")
    # The ledger never sees the run input (the diff) or any output content.
    assert "the diff text" not in lines[0]


@pytest.mark.asyncio
async def test_ledger_runner_string_output_counts_zero_findings(monkeypatch, tmp_path) -> None:
    result = FakeResult("a plain summary", FakeAgent("ReviewDesk"), make_usage())
    monkeypatch.setattr(agents.run, "DEFAULT_AGENT_RUNNER", FakeAgentRunner(result))
    ledger = LedgerRunner(ledger_path=tmp_path / "ledger.jsonl")
    await ledger.run(FakeAgent("ReviewDesk"), "x", run_config=None)
    line = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert line["findings"] == 0
    assert line["request_id"] == ""  # no run config -> empty id, never a crash
    assert "model" not in line


@pytest.mark.asyncio
async def test_ledger_runner_failure_never_breaks_the_run(monkeypatch, tmp_path) -> None:
    result = FakeResult([], FakeAgent("StyleReviewer"), make_usage())
    monkeypatch.setattr(agents.run, "DEFAULT_AGENT_RUNNER", FakeAgentRunner(result))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(observe, "append_ledger_line", boom)
    ledger = LedgerRunner(ledger_path=tmp_path / "ledger.jsonl")
    out = await ledger.run(FakeAgent("StyleReviewer"), "x", run_config=None)
    assert out is result  # a ledger failure is logged, never raised


@pytest.mark.asyncio
async def test_ledger_runner_injects_current_run_hooks(monkeypatch, tmp_path) -> None:
    result = FakeResult([], FakeAgent("SecurityReviewer"), make_usage())
    fake_runner = FakeAgentRunner(result)
    monkeypatch.setattr(agents.run, "DEFAULT_AGENT_RUNNER", fake_runner)
    hooks = MetricsRunHooks()
    token = observe.current_run_hooks.set(hooks)
    try:
        ledger = LedgerRunner(ledger_path=tmp_path / "ledger.jsonl")
        await ledger.run(FakeAgent("SecurityReviewer"), "x", run_config=None)
    finally:
        observe.current_run_hooks.reset(token)
    assert fake_runner.calls[0]["kwargs"].get("hooks") is hooks


@pytest.mark.asyncio
async def test_ledger_runner_respects_explicit_hooks(monkeypatch, tmp_path) -> None:
    result = FakeResult([], FakeAgent("SecurityReviewer"), make_usage())
    fake_runner = FakeAgentRunner(result)
    monkeypatch.setattr(agents.run, "DEFAULT_AGENT_RUNNER", fake_runner)
    current = MetricsRunHooks()
    explicit = MetricsRunHooks()
    token = observe.current_run_hooks.set(current)
    try:
        ledger = LedgerRunner(ledger_path=tmp_path / "ledger.jsonl")
        await ledger.run(FakeAgent("SecurityReviewer"), "x", run_config=None, hooks=explicit)
    finally:
        observe.current_run_hooks.reset(token)
    assert fake_runner.calls[0]["kwargs"].get("hooks") is explicit


# --- MetricsRunHooks: run-level hooks record real usage (FR-10) -----------------


async def _fire_lifecycle(hooks: MetricsRunHooks, agent: FakeAgent, usage: Usage) -> None:
    context = SimpleNamespace(usage=usage)
    await hooks.on_agent_start(context, agent)
    await asyncio.sleep(0.005)  # a measurable slice of latency
    await hooks.on_agent_end(context, agent, output=[])


@pytest.mark.asyncio
async def test_metrics_hooks_record_latency_and_real_usage() -> None:
    stats: dict[str, ReviewerStats] = {}
    hooks = MetricsRunHooks(stats)
    agent = FakeAgent("SecurityReviewer")
    await _fire_lifecycle(hooks, agent, make_usage())

    row = stats["SecurityReviewer"]
    assert row.reviewer_name == "SecurityReviewer"
    assert row.latency_ms > 0
    assert row.tokens_in == 540  # real input tokens from the run context
    assert row.tokens_out == 210
    assert row.requests == 2
    assert row.total_tokens == 750


@pytest.mark.asyncio
async def test_metrics_hooks_shared_dict_across_agents() -> None:
    stats: dict[str, ReviewerStats] = {}
    hooks = MetricsRunHooks(stats)
    for name, usage in (
        ("SecurityReviewer", Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)),
        ("TestsReviewer", Usage(requests=3, input_tokens=20, output_tokens=8, total_tokens=28)),
    ):
        await _fire_lifecycle(hooks, FakeAgent(name), usage)
    assert set(stats) == {"SecurityReviewer", "TestsReviewer"}
    assert stats["TestsReviewer"].requests == 3


@pytest.mark.asyncio
async def test_metrics_hooks_tolerate_missing_context_and_unstarted_agents() -> None:
    hooks = MetricsRunHooks()
    agent = FakeAgent("StyleReviewer")
    await hooks.on_agent_end(SimpleNamespace(usage=None), agent, output=None)
    row = hooks.stats["StyleReviewer"]
    assert row.tokens_in == 0 and row.tokens_out == 0 and row.requests == 0
    await hooks.on_agent_end(None, FakeAgent("MysteryAgent"), output=None)
    assert "MysteryAgent" in hooks.stats  # still recorded, zeros, no crash


# --- SecurityProbeHooks: agent-level per-event detail on ONE reviewer (FR-10) ---


@pytest.mark.asyncio
async def test_probe_hooks_see_per_event_detail_without_content() -> None:
    probe = SecurityProbeHooks()
    agent = FakeAgent("SecurityReviewer")
    tool = SimpleNamespace(name="read_ruleset")
    context = SimpleNamespace(usage=make_usage())

    await probe.on_start(context, agent)
    await probe.on_llm_start(context, agent, "system prompt text", [])
    await probe.on_tool_start(context, agent, tool)
    await probe.on_tool_end(context, agent, tool, "tool output that could hold secrets")
    await probe.on_llm_end(context, agent, SimpleNamespace())
    await probe.on_end(context, agent, "final output that could hold secrets")
    await probe.on_handoff(context, FakeAgent("RemediationSpecialist"), agent)

    assert probe.events == [
        "start:SecurityReviewer",
        "llm_start:SecurityReviewer",
        "tool_start:read_ruleset",
        "tool_end:read_ruleset",
        "llm_end:SecurityReviewer",
        "end:SecurityReviewer",
        "handoff:SecurityReviewer->RemediationSpecialist",
    ]
    # No tool output or final output content may ever land in the event log.
    assert not any("secrets" in event for event in probe.events)


@pytest.mark.asyncio
async def test_probe_hooks_cap_the_event_list() -> None:
    probe = SecurityProbeHooks(max_events=3)
    agent = FakeAgent("SecurityReviewer")
    context = SimpleNamespace(usage=None)
    for _ in range(10):
        await probe.on_start(context, agent)
    assert len(probe.events) == 3


def test_attach_agent_hooks_targets_exactly_one_reviewer() -> None:
    reviewers = make_reviewers()
    probe = attach_agent_hooks(reviewers)
    assert reviewers["SecurityReviewer"].hooks is probe
    assert reviewers["TestsReviewer"].hooks is None
    assert reviewers["StyleReviewer"].hooks is None


def test_attach_agent_hooks_falls_back_to_first_reviewer() -> None:
    reviewers = {"TestsReviewer": FakeAgent("TestsReviewer")}
    probe = attach_agent_hooks(reviewers)
    assert reviewers["TestsReviewer"].hooks is probe


# --- setup_tracing (FR-13, NFR-3) -----------------------------------------------


def test_setup_tracing_exports_under_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    exported: list[str] = []
    monkeypatch.setattr(
        observe, "set_tracing_export_api_key", lambda key: exported.append(key)
    )
    disabled: list[bool] = []
    monkeypatch.setattr(observe, "set_tracing_disabled", lambda flag: disabled.append(flag))
    assert setup_tracing() is None
    assert exported == ["sk-test-not-a-real-key"]
    assert disabled == []


def test_setup_tracing_disables_with_sentence_when_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    disabled: list[bool] = []
    monkeypatch.setattr(observe, "set_tracing_disabled", lambda flag: disabled.append(flag))
    exported: list[str] = []
    monkeypatch.setattr(
        observe, "set_tracing_export_api_key", lambda key: exported.append(key)
    )
    sentence = setup_tracing()
    assert disabled == [True]
    assert exported == []
    assert isinstance(sentence, str) and sentence.endswith(".")


def test_setup_tracing_never_raises_and_never_echoes_the_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret-value")

    def explode(key: str) -> None:
        raise RuntimeError(f"exporter rejected {key}")

    monkeypatch.setattr(observe, "set_tracing_export_api_key", explode)
    disabled: list[bool] = []
    monkeypatch.setattr(observe, "set_tracing_disabled", lambda flag: disabled.append(flag))
    sentence = setup_tracing()  # must not raise
    assert disabled == [True]
    assert "sk-very-secret-value" not in sentence  # the key is never echoed
