"""Unit tests for the model router (src/model_config.py).

No test in this file touches the network: the failover chain runs over stub
underlying models whose scripted outcomes are plain exceptions and
ModelResponse objects, and the shared Gemini client is built offline against a
dummy key (constructing AsyncOpenAI sends nothing).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx2
import openai
import pytest
from agents import Model, ModelResponse, ModelSettings, ModelTracing, Usage, UserError
from agents.agent_output import AgentOutputSchemaBase
from agents.models import _openai_shared

from src import model_config
from src.model_config import (
    DEFAULT_PRIORITY_MODEL,
    MODEL_CHAIN,
    ModelChainExhausted,
    PRIORITY_MODEL,
    RATE_LIMIT_TABLE,
)

# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------

CHAIN_A = "gemini-2.5-flash"
CHAIN_B = "gemini-3.5-flash-lite"
CHAIN_C = "gemini-3.1-flash-lite"

EXPECTED_DEFAULT_CHAIN = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash",
]

CALL_KWARGS = dict(
    previous_response_id=None,
    conversation_id=None,
    prompt=None,
)


@pytest.fixture(autouse=True)
def _isolated_router(monkeypatch):
    """Fresh router caches per test; a dummy key keeps client building offline."""
    model_config._reset_for_tests()
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key-never-used")
    yield
    model_config._reset_for_tests()


class StubModel(Model):
    """Underlying model stand-in: scripted outcomes plus call recording.

    Outcomes are consumed in order; the last one repeats for any further
    calls. Exceptions are raised, ModelResponse objects are returned.
    """

    def __init__(self, name: str, outcomes: list[Any]) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls: list[Any] = []
        self.settings: list[Any] = []
        self.schemas: list[Any] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        self.calls.append(input)
        self.settings.append(model_settings)
        self.schemas.append(output_schema)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> Any:
        self.calls.append(input)
        self.settings.append(model_settings)
        self.schemas.append(output_schema)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        yield {"type": "response.output_text.delta", "delta": self.name}


def _ok_response(response_id: str = "resp-ok") -> ModelResponse:
    return ModelResponse(output=[], usage=Usage(), response_id=response_id)


def _api_error(status: int, message: str) -> openai.APIStatusError:
    """Build a real openai status error over a synthetic httpx2 response."""
    request = httpx2.Request(
        "POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    response = httpx2.Response(status, request=request, headers={"x-request-id": "test"})
    if status == 429:
        return openai.RateLimitError(message, response=response, body=None)
    if status == 404:
        return openai.NotFoundError(message, response=response, body=None)
    if status == 401:
        return openai.AuthenticationError(message, response=response, body=None)
    if status == 400:
        return openai.BadRequestError(message, response=response, body=None)
    if status >= 500:
        return openai.InternalServerError(message, response=response, body=None)
    return openai.APIStatusError(message, response=response, body=None)


def _failover(
    outcomes: dict[str, list[Any]],
    chain: list[str] | None = None,
    cooldown_seconds: float = 60.0,
) -> tuple[model_config.FailoverModel, dict[str, StubModel]]:
    """A FailoverModel wired to stub underlying models (no client, no network)."""
    chain = chain or [CHAIN_A, CHAIN_B]
    stubs = {name: StubModel(name, script) for name, script in outcomes.items()}
    model = model_config.FailoverModel(
        chain,
        cooldown_seconds=cooldown_seconds,
        models_factory=lambda name: stubs[name],
    )
    return model, stubs


async def _call(model: Model, input_text: str = "review this diff"):
    return await model.get_response(
        None, input_text, ModelSettings, [], None, [], ModelTracing.DISABLED, **CALL_KWARGS
    )


# ---------------------------------------------------------------------------
# Table, chain, and priority model
# ---------------------------------------------------------------------------


def test_rate_limit_table_is_verbatim():
    assert RATE_LIMIT_TABLE == {
        "gemini-3.5-flash-lite": 15,
        "gemini-3.1-flash-lite": 15,
        "gemini-3.8-flash": 5,
        "gemini-3.7-flash": 5,
        "gemini-3.6-flash": 5,
        "gemini-3.5-flash": 5,
        "gemini-3-flash": 5,
        "gemini-2.5-flash": 5,
    }


def test_default_priority_model_and_env_override(monkeypatch):
    monkeypatch.delenv("PRIORITY_MODEL", raising=False)
    assert model_config._priority_from_env() == "gemini-2.5-flash"
    assert DEFAULT_PRIORITY_MODEL == "gemini-2.5-flash"
    monkeypatch.setenv("PRIORITY_MODEL", "gemini-3.6-flash")
    assert model_config._priority_from_env() == "gemini-3.6-flash"


def test_chain_order_priority_first_then_table_order(monkeypatch):
    monkeypatch.setattr(model_config, "PRIORITY_MODEL", "gemini-2.5-flash")
    assert model_config.build_chain() == EXPECTED_DEFAULT_CHAIN
    assert MODEL_CHAIN == EXPECTED_DEFAULT_CHAIN
    assert PRIORITY_MODEL == "gemini-2.5-flash"


def test_chain_priority_override_dedupes_and_keeps_table_order():
    chain = model_config.build_chain("gemini-3.6-flash")
    assert chain[0] == "gemini-3.6-flash"
    assert chain.count("gemini-3.6-flash") == 1
    assert chain[1:] == [name for name in RATE_LIMIT_TABLE if name != "gemini-3.6-flash"]


def test_chain_priority_outside_table_leads_all_table_models():
    chain = model_config.build_chain("gemini-9.9-flash")
    assert chain[0] == "gemini-9.9-flash"
    assert chain[1:] == list(RATE_LIMIT_TABLE)


# ---------------------------------------------------------------------------
# get_model / model_for_run / client factory
# ---------------------------------------------------------------------------


def test_get_model_default_anchors_at_priority_model(monkeypatch):
    monkeypatch.setattr(model_config, "PRIORITY_MODEL", "gemini-2.5-flash")
    default_model = model_config.get_model()
    assert isinstance(default_model, model_config.FailoverModel)
    assert default_model.chain[0] == "gemini-2.5-flash"
    assert default_model.chain == EXPECTED_DEFAULT_CHAIN


def test_get_model_is_cached_per_anchor():
    first = model_config.get_model("gemini-3.6-flash")
    again = model_config.get_model("gemini-3.6-flash")
    default_one = model_config.get_model()
    default_two = model_config.get_model()
    assert first is again
    assert default_one is default_two
    assert first is not default_one
    assert first.chain[0] == "gemini-3.6-flash"


def test_model_for_run_override_and_default(monkeypatch):
    monkeypatch.setattr(model_config, "PRIORITY_MODEL", "gemini-2.5-flash")
    run_model = model_config.model_for_run("gemini-3.1-flash-lite")
    assert isinstance(run_model, model_config.FailoverModel)
    assert run_model.chain[0] == "gemini-3.1-flash-lite"
    assert model_config.model_for_run() is model_config.get_model()


def test_shared_client_is_explicit_gemini_client_with_no_retries():
    client = model_config.get_gemini_client()
    assert model_config.get_gemini_client() is client  # one shared client
    assert str(client.base_url).rstrip("/") == model_config.GEMINI_BASE_URL.rstrip("/")
    assert client.max_retries == 0  # the router retries itself
    assert _openai_shared.get_default_openai_client() is None  # no global default set


def test_underlying_models_are_cached_and_named():
    first = model_config.get_model("gemini-3.6-flash")
    second = model_config.get_model("gemini-3.6-flash")
    underlying_a = model_config._underlying_model("gemini-3.6-flash")
    underlying_b = model_config._underlying_model("gemini-3.6-flash")
    assert underlying_a is underlying_b
    assert underlying_a.model == "gemini-3.6-flash"
    assert first is second
    assert model_config.current_model_name(first) == "gemini-3.6-flash"


def test_missing_gemini_api_key_raises_one_clear_sentence(monkeypatch):
    model_config._reset_for_tests()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(UserError, match="GEMINI_API_KEY"):
        model_config.get_gemini_client()


# ---------------------------------------------------------------------------
# Failover behaviour (all offline, over stubs)
# ---------------------------------------------------------------------------


async def test_429_switches_to_next_chain_model():
    rate_limited = _api_error(429, "Rate limit reached for gemini-2.5-flash")
    first_answer, second_answer = _ok_response("resp-1"), _ok_response("resp-2")
    model, stubs = _failover(
        {CHAIN_A: [rate_limited, first_answer], CHAIN_B: [second_answer, first_answer]}
    )

    response = await _call(model)

    assert response is second_answer  # the switched model answered
    assert len(stubs[CHAIN_A].calls) == 1  # exactly one failed attempt
    assert len(stubs[CHAIN_B].calls) == 1
    assert model.switch_count == 1
    assert model.current_name == CHAIN_B


async def test_subsequent_call_keeps_using_switched_model():
    rate_limited = _api_error(429, "Rate limit reached")
    model, stubs = _failover(
        {CHAIN_A: [rate_limited, _ok_response("resp-a2")], CHAIN_B: [_ok_response("resp-b1")]}
    )
    await _call(model)

    second = await _call(model)

    assert second.response_id == "resp-b1"
    assert len(stubs[CHAIN_A].calls) == 1  # the failed model is not retried
    assert len(stubs[CHAIN_B].calls) == 2  # the switch is visible to later calls


async def test_usage_counters_switches_and_cooldown_are_recorded():
    model, stubs = _failover(
        {CHAIN_A: [_api_error(429, "quota exceeded")], CHAIN_B: [_ok_response()]}
    )
    await _call(model)

    usage = model.usage
    assert usage["switches"] == 1
    assert usage["current_model"] == CHAIN_B
    assert usage["per_model"][CHAIN_A] == {"requests": 1, "succeeded": 0, "failed": 1}
    assert usage["per_model"][CHAIN_B] == {"requests": 1, "succeeded": 1, "failed": 0}
    assert model.is_cooling(CHAIN_A) is True
    assert model.is_cooling(CHAIN_B) is False
    assert 0 < model.cooldown_remaining(CHAIN_A) <= model.cooldown_seconds
    assert model.cooldown_seconds == 60.0  # default cooldown, configurable


async def test_clean_call_passes_through_untouched():
    answer = _ok_response("resp-clean")
    model, stubs = _failover({CHAIN_A: [answer], CHAIN_B: [answer]})

    response = await _call(model, "the diff text")

    assert response is answer
    assert stubs[CHAIN_A].calls == ["the diff text"]  # arguments forwarded as-is
    assert stubs[CHAIN_B].calls == []
    assert model.switch_count == 0
    assert model.current_name == CHAIN_A
    assert model.usage["per_model"][CHAIN_A] == {"requests": 1, "succeeded": 1, "failed": 0}


async def test_model_not_found_404_triggers_failover():
    deprecated = _api_error(404, "model gemini-2.5-flash is no longer available to new users")
    answer = _ok_response("resp-404")
    model, stubs = _failover({CHAIN_A: [deprecated], CHAIN_B: [answer]})

    response = await _call(model)

    assert response is answer
    assert model.switch_count == 1
    assert model.is_cooling(CHAIN_A) is True


def test_is_failover_trigger_covers_availability_layers():
    """Unit view of the trigger table: 5xx-availability and connection
    failures trigger, 4xx request problems never do (404 is the exception)."""
    assert model_config.is_failover_trigger(_api_error(429, "quota exceeded")) is True
    assert model_config.is_failover_trigger(_api_error(404, "model gone")) is True
    assert model_config.is_failover_trigger(_api_error(500, "server error")) is True
    assert model_config.is_failover_trigger(
        _api_error(
            503,
            "This model is currently experiencing high demand "
            "[UNAVAILABLE]. Please try again later.",
        )
    ) is True
    request = httpx2.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai")
    assert model_config.is_failover_trigger(openai.APIConnectionError(request=request)) is True
    assert model_config.is_failover_trigger(openai.APITimeoutError(request=request)) is True
    assert model_config.is_failover_trigger(_api_error(400, "bad request")) is False
    assert model_config.is_failover_trigger(_api_error(401, "bad key")) is False
    assert model_config.is_failover_trigger(_api_error(422, "unprocessable")) is False


async def test_503_availability_error_switches_to_next_chain_model():
    """A provider-availability 5xx (high demand / UNAVAILABLE) must advance
    the chain exactly like a rate limit: switch, cooldown, usage recorded."""
    overloaded = _api_error(
        503,
        "This model is currently experiencing high demand "
        "[UNAVAILABLE]. Please try again later.",
    )
    first_answer, second_answer = _ok_response("resp-1"), _ok_response("resp-2")
    model, stubs = _failover(
        {CHAIN_A: [overloaded, first_answer], CHAIN_B: [second_answer, first_answer]}
    )

    response = await _call(model)

    assert response is second_answer  # the switched model answered
    assert len(stubs[CHAIN_A].calls) == 1  # exactly one failed attempt
    assert len(stubs[CHAIN_B].calls) == 1
    assert model.switch_count == 1
    assert model.current_name == CHAIN_B
    assert model.is_cooling(CHAIN_A) is True
    assert model.usage["per_model"][CHAIN_A] == {"requests": 1, "succeeded": 0, "failed": 1}


async def test_connection_and_timeout_errors_trigger_failover():
    """APIConnectionError and its APITimeoutError subclass are provider-side
    availability failures: the chain advances and nothing propagates."""
    request = httpx2.Request(
        "POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    for connection_error in (
        openai.APIConnectionError(request=request),
        openai.APITimeoutError(request=request),
    ):
        model, stubs = _failover(
            {CHAIN_A: [connection_error], CHAIN_B: [_ok_response("resp-conn")]}
        )

        response = await _call(model)

        assert response.response_id == "resp-conn"  # the next model answered
        assert model.switch_count == 1
        assert model.is_cooling(CHAIN_A) is True
        assert stubs[CHAIN_B].calls == ["review this diff"]


async def test_bad_request_400_does_not_switch_chain_models():
    """400 BadRequestError is a request problem: it propagates immediately —
    no switch, no cooldown, no wasted attempt on the next model."""
    bad_request = _api_error(
        400,
        "Forced function calling (ANY mode) with a response mime type: "
        "'application/json' is unsupported",
    )
    model, stubs = _failover({CHAIN_A: [bad_request], CHAIN_B: [_ok_response("resp-b")]})

    with pytest.raises(openai.BadRequestError):
        await _call(model)

    assert model.switch_count == 0
    assert model.current_name == CHAIN_A
    assert stubs[CHAIN_B].calls == []  # a 400 is never retried on the next model
    assert model.is_cooling(CHAIN_A) is False
    assert model.usage["per_model"][CHAIN_A] == {"requests": 1, "succeeded": 0, "failed": 0}


async def test_5xx_across_whole_chain_exhausts_it():
    """Every model 503ing is real unavailability: the chain is exhausted and
    the raised error names the chain and the last underlying error."""
    chain = [CHAIN_A, CHAIN_B, CHAIN_C]
    model, stubs = _failover(
        {name: [_api_error(503, f"{name} is overloaded")] for name in chain},
        chain=chain,
    )

    with pytest.raises(ModelChainExhausted) as excinfo:
        await _call(model)

    for name in chain:
        assert len(stubs[name].calls) == 1  # one attempt each, no loops
    assert all(stat["failed"] == 1 for stat in model.usage["per_model"].values())
    assert CHAIN_A in str(excinfo.value) and CHAIN_C in str(excinfo.value)


async def test_authentication_error_propagates_without_switch():
    unauthorized = _api_error(401, "Incorrect API key provided")
    model, stubs = _failover({CHAIN_A: [unauthorized], CHAIN_B: [_ok_response()]})

    with pytest.raises(openai.AuthenticationError):
        await _call(model)

    assert model.switch_count == 0
    assert stubs[CHAIN_B].calls == []  # auth failures are not solved by switching


async def test_chain_exhaustion_raises_after_trying_every_model_once():
    chain = [CHAIN_A, CHAIN_B, CHAIN_C]
    model, stubs = _failover(
        {name: [_api_error(429, f"quota exceeded for {name}")] for name in chain},
        chain=chain,
    )

    with pytest.raises(ModelChainExhausted) as excinfo:
        await _call(model)

    for name in chain:
        assert len(stubs[name].calls) == 1  # each model got exactly one attempt
    assert CHAIN_A in str(excinfo.value) and CHAIN_C in str(excinfo.value)
    assert all(stat["failed"] == 1 for stat in model.usage["per_model"].values())


async def test_cooldown_skips_failed_model_at_call_start():
    answer = _ok_response()
    model, stubs = _failover({CHAIN_A: [answer], CHAIN_B: [answer]})
    model.cooldowns[CHAIN_A] = time.monotonic() + 60.0  # CHAIN_A just went quiet

    await _call(model)

    assert stubs[CHAIN_A].calls == []  # skipped while cooling
    assert len(stubs[CHAIN_B].calls) == 1


async def test_cooldown_expiry_makes_model_eligible_again():
    answer = _ok_response()
    model, stubs = _failover({CHAIN_A: [answer], CHAIN_B: [answer]}, cooldown_seconds=0.05)
    model.cooldowns[CHAIN_A] = time.monotonic() + 0.05

    await asyncio.sleep(0.08)
    await _call(model)

    assert len(stubs[CHAIN_A].calls) == 1  # cooldown expired, model is used again


async def test_stream_fails_over_before_first_event():
    rate_limited = _api_error(429, "Rate limit reached")
    model, stubs = _failover({CHAIN_A: [rate_limited], CHAIN_B: [_ok_response()]})

    events = [
        event
        async for event in model.stream_response(
            None, "diff", ModelSettings, [], None, [], ModelTracing.DISABLED, **CALL_KWARGS
        )
    ]

    assert events == [{"type": "response.output_text.delta", "delta": CHAIN_B}]
    assert model.switch_count == 1
    assert len(stubs[CHAIN_B].calls) == 1


async def test_chain_wrap_around_does_not_loop_forever():
    rate_limited = _api_error(429, "Rate limit reached")
    answer = _ok_response()
    model, stubs = _failover(
        {CHAIN_A: [rate_limited], CHAIN_B: [answer, rate_limited]},
    )

    assert await _call(model) is answer  # A fails, B answers; position rests on B

    with pytest.raises(ModelChainExhausted):  # B fails, wraps to A, A fails: stop
        await _call(model)

    assert len(stubs[CHAIN_A].calls) == 2  # wrapped to once per call, no infinite loop
    assert len(stubs[CHAIN_B].calls) == 2


def test_failover_model_rejects_empty_chain_and_dedupes():
    with pytest.raises(UserError):
        model_config.FailoverModel([])
    deduped = model_config.FailoverModel([CHAIN_A, CHAIN_A, CHAIN_B])
    assert deduped.chain == [CHAIN_A, CHAIN_B]
    assert deduped.cooldown_seconds == 60.0


async def test_current_model_name_helper():
    model, _stubs = _failover({CHAIN_A: [_ok_response()]})
    assert model_config.current_model_name(model) == CHAIN_A
    underlying = model_config._underlying_model("gemini-3.6-flash")
    assert model_config.current_model_name(underlying) == "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# First-turn mediation: forced tool call vs structured output (Gemini 400)
# ---------------------------------------------------------------------------


class FakeOutputSchema(AgentOutputSchemaBase):
    """Minimal output-schema stand-in: mediation must only ever drop or keep it."""

    def __init__(self, name: str = "findings") -> None:
        self._name = name

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self._name

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    def is_strict_json_schema(self) -> bool:
        return True

    def validate_json(self, json_str: str) -> Any:
        return json_str


def _turn2_input() -> list[dict[str, Any]]:
    """A later-turn input: user message, the forced function_call, its output."""
    return [
        {"type": "message", "role": "user", "content": "review this diff"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_ruleset",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "rules text"},
    ]


async def _call_with_schema(
    model: Model,
    *,
    settings: ModelSettings,
    input_items: Any = "review this diff",
    schema: Any,
):
    return await model.get_response(
        None, input_items, settings, [], schema, [], ModelTracing.DISABLED, **CALL_KWARGS
    )


def test_input_tool_history_detection():
    assert model_config._input_has_tool_history("a plain diff") is False
    assert (
        model_config._input_has_tool_history(
            [{"type": "message", "role": "user", "content": "review"}]
        )
        is False
    )
    assert (
        model_config._input_has_tool_history(
            [{"type": "function_call", "call_id": "c", "name": "t", "arguments": "{}"}]
        )
        is True
    )
    assert (
        model_config._input_has_tool_history(
            [{"type": "function_call_output", "call_id": "c", "output": "o"}]
        )
        is True
    )


async def test_first_turn_mediation_drops_schema_keeps_forced_tool_choice():
    """Turn 1 with tool_choice="required" + a schema: the underlying request
    must lose the response_format (Gemini 400 fix) but keep the forcing."""
    schema = FakeOutputSchema()
    answer = _ok_response("resp-med")
    model, stubs = _failover({CHAIN_A: [answer]})
    forced = ModelSettings(tool_choice="required")

    response = await _call_with_schema(model, settings=forced, schema=schema)

    assert response is answer
    assert stubs[CHAIN_A].calls == ["review this diff"]
    assert stubs[CHAIN_A].schemas == [None]  # response_format dropped on turn 1
    assert stubs[CHAIN_A].settings[0].tool_choice == "required"  # forcing intact


async def test_mediation_forwards_schema_once_tool_history_exists():
    """Turn 2+: the run already did tool I/O, so structured output is forwarded."""
    schema = FakeOutputSchema()
    model, stubs = _failover({CHAIN_A: [_ok_response("resp-t2")]})
    forced = ModelSettings(tool_choice="required")

    await _call_with_schema(
        model, settings=forced, input_items=_turn2_input(), schema=schema
    )

    assert stubs[CHAIN_A].schemas == [schema]  # original schema, unchanged
    assert stubs[CHAIN_A].calls == [_turn2_input()]


async def test_no_forced_tool_choice_is_never_mediated():
    """Desk/Merge/Remediation agents (no tool_choice, or auto/none) must never
    lose their first-turn schema — their first turn IS the structured turn."""
    schema = FakeOutputSchema()
    model, stubs = _failover({CHAIN_A: [_ok_response()]})

    await _call_with_schema(model, settings=ModelSettings(), schema=schema)
    await _call_with_schema(
        model, settings=ModelSettings(tool_choice="auto"), schema=schema
    )
    await _call_with_schema(
        model, settings=ModelSettings(tool_choice="none"), schema=schema
    )

    assert stubs[CHAIN_A].schemas == [schema, schema, schema]


async def test_failover_still_switches_on_429_with_mediation_active():
    """The mediation wraps each attempt inside the failover loop: a rate-limited
    first turn still advances the chain, mediated, with usage recorded."""
    schema = FakeOutputSchema()
    answer = _ok_response("resp-b")
    model, stubs = _failover(
        {CHAIN_A: [_api_error(429, "quota exceeded")], CHAIN_B: [answer]}
    )
    forced = ModelSettings(tool_choice="required")

    response = await _call_with_schema(model, settings=forced, schema=schema)

    assert response is answer
    assert model.switch_count == 1
    assert stubs[CHAIN_A].schemas == [None]  # failed attempt was mediated too
    assert stubs[CHAIN_B].schemas == [None]  # switched attempt mediated as well
    assert stubs[CHAIN_B].settings[0].tool_choice == "required"
    assert model.usage["per_model"][CHAIN_A] == {"requests": 1, "succeeded": 0, "failed": 1}
    assert model.usage["per_model"][CHAIN_B] == {"requests": 1, "succeeded": 1, "failed": 0}


async def test_stream_response_mediation_drops_schema_on_first_turn():
    """stream_response forwards the same way, so it mediates identically."""
    schema = FakeOutputSchema()
    model, stubs = _failover({CHAIN_A: [_ok_response()]})
    forced = ModelSettings(tool_choice="required")

    events = [
        event
        async for event in model.stream_response(
            None, "diff", forced, [], schema, [], ModelTracing.DISABLED, **CALL_KWARGS
        )
    ]

    assert events  # the stream still yields its event
    assert stubs[CHAIN_A].schemas == [None]
    assert stubs[CHAIN_A].settings[0].tool_choice == "required"
