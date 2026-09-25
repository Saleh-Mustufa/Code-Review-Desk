"""Unit tests for src/review.py.

Fully offline: stub runners, no network, no real model calls, no API key.
The runner is injected into run_reviewer (DI), so tests substitute a
StubRunner whose async ``run`` mimics ``Runner.run``'s signature.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from agents import MaxTurnsExceeded, ModelBehaviorError, RunContextWrapper
from agents.agent_output import AgentOutputSchema
from agents.run_error_handlers import RunErrorHandlerInput
from agents.tool_context import ToolContext
from pydantic import ValidationError

from src import review as review_module
from src.intake import ReviewContext
from src.model_config import FailoverModel
from src.review import (
    FOCUS_PROMPTS,
    REVIEWER_MAX_TURNS,
    Finding,
    ReviewerOutcome,
    _invalid_output_handler,
    _partial_review_handler,
    base_reviewer_instructions,
    make_base_reviewer,
    make_reviewers,
    read_ruleset,
    run_reviewer,
    run_reviewers_concurrently,
    run_reviewers_sequentially,
)

# --- Offline fakes -----------------------------------------------------------


def make_ctx(
    repo: str = "desk-demo-repo-xyz",
    language: str = "python",
    ruleset_id: str = "default",
    strictness: str = "normal",
) -> ReviewContext:
    return ReviewContext(
        repo=repo, language=language, ruleset_id=ruleset_id, strictness=strictness
    )


def sample_findings() -> list[Finding]:
    return [
        Finding(file="src/app.py", line=10, severity="critical", message="Hardcoded API key."),
        Finding(file="src/app.py", line=42, severity="minor", message="Unused import."),
    ]


class StubUsage:
    requests = 1
    input_tokens = 120
    output_tokens = 45
    total_tokens = 165


class StubContextWrapper:
    usage = StubUsage()


class StubRunResult:
    """RunResult-shaped stand-in: final_output is a plain list, usage present."""

    def __init__(self, final_output: Any) -> None:
        self.final_output = final_output
        self.context_wrapper = StubContextWrapper()
        self.new_items: list[Any] = []
        self.last_agent: Any = None


class StubRunner:
    """Offline stand-in for Runner.run: records calls, optional delay/raise."""

    def __init__(
        self,
        final_output: Any = None,
        delay: float = 0.0,
        exc: BaseException | None = None,
    ) -> None:
        self.final_output = [] if final_output is None else final_output
        self.delay = delay
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def run(self, agent: Any, input: Any, **kwargs: Any) -> StubRunResult:
        self.calls.append({"agent": agent, "input": input, "kwargs": kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return StubRunResult(list(self.final_output))


class HandlerAwareStubRunner(StubRunner):
    """Mimics the real SDK path: routes the failure through error_handlers."""

    def __init__(
        self,
        final_output: Any = None,
        delay: float = 0.0,
        exc: BaseException | None = None,
        kind: str = "max_turns",
    ) -> None:
        super().__init__(final_output=final_output, delay=delay, exc=exc)
        self.kind = kind

    async def run(self, agent: Any, input: Any, **kwargs: Any) -> StubRunResult:
        self.calls.append({"agent": agent, "input": input, "kwargs": kwargs})
        errors = {
            "max_turns": MaxTurnsExceeded("Max turns (4) exceeded"),
            "invalid_final_output": ModelBehaviorError("Invalid JSON in final answer"),
        }
        handler = kwargs["error_handlers"][self.kind]
        handler_input = SimpleNamespace(
            error=errors[self.kind],
            context=None,
            run_data=None,
        )
        result = handler(handler_input)
        return StubRunResult(result.final_output)


def invoke_tool(wrapper: RunContextWrapper[ReviewContext], args: str = "{}") -> Any:
    """Invoke the read_ruleset FunctionTool the way the runner would."""
    tool_context = ToolContext(
        context=wrapper.context,
        tool_name="read_ruleset",
        tool_call_id="call-test-1",
        tool_arguments=args,
    )
    return read_ruleset.on_invoke_tool(tool_context, args)


# --- FR-3: the typed Finding model -------------------------------------------


class TestFindingModel:
    @pytest.mark.parametrize("severity", ["critical", "major", "minor"])
    def test_accepts_the_three_severities(self, severity: str) -> None:
        finding = Finding(file="a.py", line=1, severity=severity, message="m")
        assert finding.severity == severity

    @pytest.mark.parametrize("severity", ["severe", "info", "", "CRITICAL", None])
    def test_rejects_anything_else(self, severity: Any) -> None:
        with pytest.raises(ValidationError):
            Finding(file="a.py", line=1, severity=severity, message="m")

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            Finding(file="a.py", line=1, severity="minor")  # type: ignore[call-arg]


# --- FR-4: dynamic instructions, two contexts -> two prompts ------------------


class TestDynamicInstructions:
    def test_normal_and_strict_contexts_produce_different_prompts(self) -> None:
        agent = make_base_reviewer()
        normal = base_reviewer_instructions(
            RunContextWrapper(context=make_ctx()), agent
        )
        strict = base_reviewer_instructions(
            RunContextWrapper(
                context=make_ctx(ruleset_id="strict", strictness="strict")
            ),
            agent,
        )
        assert normal != strict
        assert len(strict) < len(normal), "strict prompt must be terser"
        # The normal prompt has the elaboration section; strict does not.
        assert "How to work:" in normal
        assert "How to work:" not in strict
        # Both assemble their ruleset at request time.
        assert "Default Desk Rules" in normal
        assert "Strict Desk Rules" in strict
        # Language flows from the context into the prompt.
        assert "python" in normal and "python" in strict

    def test_no_repository_name_in_any_prompt(self) -> None:
        repo = "super-secret-repo-acme-corp"
        agent = make_base_reviewer()
        contexts = [
            make_ctx(repo=repo),
            make_ctx(repo=repo, ruleset_id="strict", strictness="strict"),
        ]
        for ctx in contexts:
            prompt = base_reviewer_instructions(RunContextWrapper(context=ctx), agent)
            assert repo not in prompt

    def test_strict_prompt_is_short_imperative_lines(self) -> None:
        agent = make_base_reviewer()
        strict = base_reviewer_instructions(
            RunContextWrapper(
                context=make_ctx(ruleset_id="strict", strictness="strict")
            ),
            agent,
        )
        prose_lines = [
            line
            for line in strict.splitlines()
            if line and not line.startswith("-") and not line.startswith("[")
        ]
        assert prose_lines, "prompt must have framing lines"
        assert max(len(line) for line in prose_lines) < 120


# --- FR-2/FR-9: the ruleset tool ----------------------------------------------


class TestReadRulesetTool:
    def test_schema_has_no_context_wrapper_parameter(self) -> None:
        schema = read_ruleset.params_json_schema
        properties = schema.get("properties", {})
        assert "ctx" not in properties
        assert "context" not in properties
        assert not any(
            "RunContextWrapper" in str(value) for value in properties.values()
        )

    async def test_returns_ruleset_text_for_a_valid_id(self) -> None:
        wrapper = RunContextWrapper(context=make_ctx(ruleset_id="default"))
        output = await invoke_tool(wrapper)
        assert "Default Desk Rules" in output

    async def test_missing_ruleset_returns_a_sentence(self) -> None:
        wrapper = RunContextWrapper(context=make_ctx(ruleset_id="no-such-ruleset"))
        output = await invoke_tool(wrapper)
        assert output.startswith("Error: ruleset 'no-such-ruleset' could not be loaded")
        assert "built-in judgement" in output

    async def test_load_ruleset_crash_is_converted_to_a_sentence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(ruleset_id: str, base_dir: Any = None) -> str:
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(review_module, "load_ruleset", boom)
        wrapper = RunContextWrapper(context=make_ctx(ruleset_id="default"))
        output = await invoke_tool(wrapper)
        assert output.startswith("Error:")
        assert "disk on fire" in output


# --- Agent structure: router model, forced tool, typed output -----------------


class TestBaseReviewerAgent:
    def test_built_from_the_router_with_list_output(self) -> None:
        agent = make_base_reviewer()
        assert agent.name == "BaseReviewer"
        assert agent.output_type == list[Finding]
        # Model comes from the router (a FailoverModel), never a hardcoded name.
        assert isinstance(agent.model, FailoverModel)
        # Instructions are dynamic (a callable), not a frozen string.
        assert callable(agent.instructions)

    def test_model_settings_are_bounded_and_force_tool_use(self) -> None:
        settings = make_base_reviewer().model_settings
        assert settings.tool_choice == "required"
        assert settings.temperature == 0.2
        assert settings.max_tokens == 2048

    def test_reset_tool_choice_stays_true(self) -> None:
        # After the forced first tool call the model must be free to answer.
        assert make_base_reviewer().reset_tool_choice is True

    def test_carries_exactly_the_ruleset_tool(self) -> None:
        assert [tool.name for tool in make_base_reviewer().tools] == ["read_ruleset"]

    def test_list_output_schema_wraps_under_a_response_key(self) -> None:
        # list[Finding] is not a JSON-schema object, so the SDK wraps it in an
        # object with a single "response" key; final_output stays a plain list.
        schema = AgentOutputSchema(make_base_reviewer().output_type)
        assert schema.is_plain_text() is False
        js = schema.json_schema()
        assert js["type"] == "object"
        response = js["properties"]["response"]
        assert response["type"] == "array"
        assert "response" in js.get("required", [])


# --- FR-5: the three clones ----------------------------------------------------


class TestReviewerClones:
    def test_three_distinct_named_clones(self) -> None:
        reviewers = make_reviewers()
        assert list(reviewers) == ["SecurityReviewer", "TestsReviewer", "StyleReviewer"]
        assert len({agent.name for agent in reviewers.values()}) == 3

    def test_focus_descriptors_cover_exactly_the_clones(self) -> None:
        assert set(FOCUS_PROMPTS) == {
            "SecurityReviewer",
            "TestsReviewer",
            "StyleReviewer",
        }

    def test_clones_inherit_tools_and_output_and_bound_settings(self) -> None:
        for agent in make_reviewers().values():
            assert agent.output_type == list[Finding]
            assert [tool.name for tool in agent.tools] == ["read_ruleset"]
            settings = agent.model_settings
            assert settings.tool_choice == "required"
            assert settings.temperature == 0.2
            assert settings.max_tokens is not None
            assert 512 <= settings.max_tokens <= 4096

    def test_clone_instructions_are_dynamic_distinct_and_share_base_logic(self) -> None:
        wrapper = RunContextWrapper(context=make_ctx())
        base = make_base_reviewer()
        resolved: dict[str, str] = {}
        for name, agent in make_reviewers().items():
            assert callable(agent.instructions)
            assert agent.instructions is not base.instructions
            resolved[name] = agent.instructions(wrapper, agent)
        assert len(set(resolved.values())) == 3, "each clone resolves its own focus"
        assert "Focus on security" in resolved["SecurityReviewer"]
        assert "Focus on test coverage" in resolved["TestsReviewer"]
        assert "Focus on style" in resolved["StyleReviewer"]
        for text in resolved.values():
            # Shared base logic still present: language + ruleset, never repo.
            assert "python" in text
            assert "Default Desk Rules" in text
            assert "desk-demo-repo-xyz" not in text


# --- run_reviewer: happy path and the turn ceiling (FR-9) ----------------------


class TestRunReviewer:
    async def test_happy_path_returns_findings_and_metadata(self) -> None:
        findings = sample_findings()
        stub = StubRunner(final_output=findings)
        ctx = make_ctx()
        outcome = await run_reviewer(make_base_reviewer(), "diff --git a/x b/x", ctx, runner=stub)

        assert isinstance(outcome, ReviewerOutcome)
        assert outcome.reviewer_name == "BaseReviewer"
        assert isinstance(outcome.findings, list)
        assert outcome.findings == findings
        assert outcome.partial is False
        assert outcome.partial_reason is None
        assert outcome.latency_ms is not None and outcome.latency_ms >= 0.0
        assert outcome.usage_summary == (
            "requests=1 input_tokens=120 output_tokens=45 total_tokens=165"
        )

        call = stub.calls[0]
        assert call["input"] == "diff --git a/x b/x"
        assert call["kwargs"]["context"] is ctx
        assert call["kwargs"]["max_turns"] == REVIEWER_MAX_TURNS == 4
        assert "max_turns" in call["kwargs"]["error_handlers"]
        assert "invalid_final_output" in call["kwargs"]["error_handlers"]

    async def test_custom_max_turns_is_forwarded(self) -> None:
        stub = StubRunner(final_output=[])
        await run_reviewer(
            make_base_reviewer(), "diff", make_ctx(), runner=stub, max_turns=2
        )
        assert stub.calls[0]["kwargs"]["max_turns"] == 2

    async def test_turn_ceiling_never_escapes_as_a_raise(self) -> None:
        # A runner that ignores error_handlers and raises: belt and braces.
        stub = StubRunner(exc=MaxTurnsExceeded("Max turns (4) exceeded"))
        outcome = await run_reviewer(make_base_reviewer(), "diff", make_ctx(), runner=stub)
        assert outcome.partial is True
        assert outcome.findings == []
        assert outcome.partial_reason is not None
        assert "turn ceiling" in outcome.partial_reason
        assert outcome.latency_ms is not None

    async def test_error_handler_path_marks_the_outcome_partial(self) -> None:
        # A runner that honours error_handlers like the real SDK does.
        stub = HandlerAwareStubRunner()
        outcome = await run_reviewer(make_base_reviewer(), "diff", make_ctx(), runner=stub)
        assert outcome.partial is True
        assert outcome.findings == []
        assert "turn ceiling" in (outcome.partial_reason or "")

    async def test_model_behavior_error_never_escapes_as_a_raise(self) -> None:
        # Belt and braces: a runner without invalid_final_output handler
        # support raises straight through run_reviewer, which must degrade it.
        stub = StubRunner(exc=ModelBehaviorError("Invalid JSON in model output"))
        outcome = await run_reviewer(make_base_reviewer(), "diff", make_ctx(), runner=stub)
        assert outcome.partial is True
        assert outcome.findings == []
        assert outcome.reviewer_name == "BaseReviewer"
        assert "produced an unparseable final answer" in (outcome.partial_reason or "")
        assert "reported as a partial review" in (outcome.partial_reason or "")
        assert outcome.latency_ms is not None

    async def test_invalid_final_output_handler_path_marks_the_outcome_partial(
        self,
    ) -> None:
        # A runner that honours error_handlers like the real SDK does: the
        # malformed structured answer is routed through the registered
        # invalid_final_output handler and the run COMPLETES with the
        # handler's empty final_output — run_reviewer flags it partial.
        stub = HandlerAwareStubRunner(kind="invalid_final_output")
        outcome = await run_reviewer(make_base_reviewer(), "diff", make_ctx(), runner=stub)
        assert outcome.partial is True
        assert outcome.findings == []
        assert "produced an unparseable final answer" in (outcome.partial_reason or "")
        assert outcome.usage_summary is not None  # the run itself completed

    def test_partial_review_handler_returns_empty_final_output(self) -> None:
        handler_input = RunErrorHandlerInput(
            error=MaxTurnsExceeded("Max turns (4) exceeded"),
            context=RunContextWrapper(context=make_ctx()),
            run_data=None,
        )
        result = _partial_review_handler(handler_input)
        assert result.final_output == []
        assert isinstance(result.include_in_history, bool)

    def test_invalid_output_handler_returns_empty_final_output(self) -> None:
        handler_input = RunErrorHandlerInput(
            error=ModelBehaviorError("Invalid JSON in model output"),
            context=RunContextWrapper(context=make_ctx()),
            run_data=None,
        )
        result = _invalid_output_handler(handler_input)
        assert result.final_output == []
        assert isinstance(result.include_in_history, bool)

    async def test_unusable_final_output_becomes_a_partial_outcome(self) -> None:
        stub = StubRunner(final_output="not a list")
        outcome = await run_reviewer(make_base_reviewer(), "diff", make_ctx(), runner=stub)
        assert outcome.partial is True
        assert outcome.findings == []
        assert "usable findings list" in (outcome.partial_reason or "")

    async def test_dict_findings_are_validated_into_finding(self) -> None:
        raw = [{"file": "a.py", "line": 3, "severity": "major", "message": "x"}]
        stub = StubRunner(final_output=raw)
        outcome = await run_reviewer(make_base_reviewer(), "diff", make_ctx(), runner=stub)
        assert outcome.findings == [
            Finding(file="a.py", line=3, severity="major", message="x")
        ]
        assert outcome.partial is False


# --- FR-5: concurrent fan-out vs sequential wall clock --------------------------


class TestFanOutConcurrency:
    async def test_concurrent_wall_clock_is_materially_faster_than_sequential(self) -> None:
        reviewers = list(make_reviewers().values())
        stub = StubRunner(final_output=[], delay=0.2)
        ctx = make_ctx()

        started = time.perf_counter()
        concurrent = await run_reviewers_concurrently(reviewers, "diff", ctx, runner=stub)
        concurrent_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        sequential = await run_reviewers_sequentially(reviewers, "diff", ctx, runner=stub)
        sequential_ms = (time.perf_counter() - started) * 1000.0

        assert len(concurrent) == 3
        assert len(sequential) == 3
        assert all(isinstance(o, ReviewerOutcome) for o in concurrent + sequential)
        assert all(o.latency_ms is not None for o in concurrent + sequential)
        # Concurrent ≈ slowest reviewer (0.2s), not the sum (0.6s).
        assert concurrent_ms < 500.0
        assert sequential_ms >= 550.0
        assert concurrent_ms < sequential_ms * 0.7

    async def test_concurrent_accepts_a_dict_of_reviewers(self) -> None:
        stub = StubRunner(final_output=[])
        outcomes = await run_reviewers_concurrently(
            make_reviewers(), "diff", make_ctx(), runner=stub
        )
        assert [o.reviewer_name for o in outcomes] == [
            "SecurityReviewer",
            "TestsReviewer",
            "StyleReviewer",
        ]

    async def test_sequential_runs_one_at_a_time(self) -> None:
        stub = StubRunner(final_output=[])
        outcomes = await run_reviewers_sequentially(
            list(make_reviewers().values()), "diff", make_ctx(), runner=stub
        )
        assert len(stub.calls) == 3
        assert [o.reviewer_name for o in outcomes] == [
            "SecurityReviewer",
            "TestsReviewer",
            "StyleReviewer",
        ]
