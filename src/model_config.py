"""Model router: the single source of truth for every model concern.

All model traffic in Code Review Desk flows through this module. It owns the
Gemini rate-limit table, the failover chain, the one explicit Gemini client,
per-model usage tracking, the first-turn forced-tool/structured-output
mediation (see :func:`_mediated_first_turn_output_schema`), and the helpers
other modules use to obtain a model. No other module may hardcode a model
name, and no global default OpenAI client is ever set here (the SDK default
stays untouched).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from openai.types.responses.response_prompt_param import ResponsePromptParam

from agents import (
    AgentsException,
    Model,
    ModelResponse,
    ModelTracing,
    UserError,
)
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.tool import Tool

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
"""Google's OpenAI-compatible endpoint every model call goes to."""

DEFAULT_PRIORITY_MODEL = "gemini-2.5-flash"
"""Priority model used when the PRIORITY_MODEL env var is not set."""

DEFAULT_COOLDOWN_SECONDS = 60.0
"""How long a model that just failed is skipped before it is tried again."""

# The rate-limit table, encoded verbatim from the spec (requests per minute).
RATE_LIMIT_TABLE: dict[str, int] = {
    "gemini-3.5-flash-lite": 15,
    "gemini-3.1-flash-lite": 15,
    "gemini-3.8-flash": 5,
    "gemini-3.7-flash": 5,
    "gemini-3.6-flash": 5,
    "gemini-3.5-flash": 5,
    "gemini-3-flash": 5,
    "gemini-2.5-flash": 5,
}


def _priority_from_env() -> str:
    """Return the user-settable priority model: env override, else the default."""
    return os.environ.get("PRIORITY_MODEL", "").strip() or DEFAULT_PRIORITY_MODEL


def build_chain(priority: str | None = None) -> list[str]:
    """Build the failover chain: priority model first, then the table in order.

    Remaining table models follow in table order (15 RPM tier, then 5 RPM
    tier), with duplicates removed. A priority outside the table still leads
    the chain ahead of every table model.
    """
    anchor = priority or PRIORITY_MODEL
    rest = [name for name in RATE_LIMIT_TABLE if name != anchor]
    return [anchor, *rest]


PRIORITY_MODEL: str = _priority_from_env()
"""The priority model for this process (env PRIORITY_MODEL, default gemini-2.5-flash)."""

MODEL_CHAIN: list[str] = build_chain(PRIORITY_MODEL)
"""The default failover chain for this process, anchored at PRIORITY_MODEL."""


class ModelChainExhausted(AgentsException):
    """Every model in the failover chain failed its latest try."""


# Status codes that mean "this model is unavailable, move down the chain":
# 429 (rate limit / quota) and 404 (model deprecated for this key).
_FAILOVER_STATUS_CODES = frozenset({429, 404})
_FAILOVER_CLASS_MARKERS = ("ratelimit", "resourceexhausted")
_FAILOVER_MESSAGE_MARKERS = (
    "resource_exhausted",
    "resourceexhausted",
    "quota",
    "rate limit",
)


def is_failover_trigger(exc: BaseException) -> bool:
    """Decide whether an error means 'switch to the next model in the chain'.

    Triggers are rate-limit-type failures (429, RateLimitError, quota,
    ResourceExhausted) and model-not-found (404, the provider deprecates old
    models for new keys). Anything else (auth, network misuse, bugs) is not
    recoverable by switching models and is raised untouched.
    """
    if isinstance(exc, RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _FAILOVER_STATUS_CODES:
        return True
    class_name = type(exc).__name__.lower()
    if any(marker in class_name for marker in _FAILOVER_CLASS_MARKERS):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _FAILOVER_MESSAGE_MARKERS)


# --- First-turn mediation: forced tool call vs structured output ---
#
# Gemini's OpenAI-compatible endpoint rejects a single request that carries
# BOTH a forced tool choice ("required" -> forced function calling, ANY mode)
# and a JSON response format: 400 "Forced function calling (ANY mode) with a
# response mime type: 'application/json' is unsupported". The SDK's
# chat-completions model puts both on the run's FIRST request whenever an agent
# combines ModelSettings(tool_choice="required") (FR-9, forced ruleset call)
# with output_type=list[Finding] (FR-3). The SDK's reset_tool_choice only
# clears tool_choice for requests AFTER the first tool turn, so the conflict is
# first-turn-only. The router mediates: on a run's first turn it drops the
# output schema (no response_format) while keeping tool_choice, so Gemini
# forces the ruleset call as designed; from the second request onward the
# schema is forwarded unchanged and structured output works again.

# The input-item ``type`` values that mean "this run has already done tool
# I/O" (the SDK's TResponseInputItem discriminated unions: function calls and
# their outputs, plus the hosted/computer/shell/MCP tool variants).
_TOOL_ITEM_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "computer_call",
        "computer_call_output",
        "local_shell_call",
        "local_shell_call_output",
        "shell_call",
        "shell_call_output",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "tool_search_call",
        "apply_patch_call",
        "apply_patch_call_output",
        "hosted_tool_call",
    }
)


def _is_forced_tool_choice(tool_choice: Any) -> bool:
    """Whether ``tool_choice`` FORCES a tool call this turn ("required", etc.).

    ``"auto"`` and ``"none"`` leave the model free not to call a tool, so
    dropping the schema there could strand a first-turn final answer without
    structured output; mediation exists only for forced first turns.
    """
    if tool_choice is None:
        return False
    if tool_choice == "auto" or tool_choice == "none":
        return False
    return True


def _input_has_tool_history(input: str | list[TResponseInputItem]) -> bool:
    """Whether the input already carries tool-call/tool-output items.

    A plain string input (or an items list with only messages) is the model's
    first turn of the run; any tool item means a later turn.
    """
    if not isinstance(input, list):
        return False
    for item in input:
        if isinstance(item, dict):
            item_type = item.get("type")
        else:  # defensive: SDK item objects instead of their dict shapes
            item_type = getattr(item, "type", None)
        if isinstance(item_type, str) and item_type in _TOOL_ITEM_TYPES:
            return True
    return False


def _mediated_first_turn_output_schema(
    model_settings: ModelSettings,
    input: str | list[TResponseInputItem],
    output_schema: AgentOutputSchemaBase | None,
) -> AgentOutputSchemaBase | None:
    """Gemini OpenAI-compat mediation for the forced-first-turn conflict.

    When the settings force a tool call AND the run is still on its first turn
    (no tool-call/tool-output items in the input), the schema is dropped so the
    request carries ``tool_choice`` WITHOUT a JSON response format — the one
    combination Gemini rejects. Every other request is forwarded unchanged:

    - agents without a forced tool choice (Desk/Merge/Remediation) are never
      mediated — their first turn IS their final structured turn;
    - from turn 2 onward (tool history present) the schema is forwarded, and
      the SDK's ``reset_tool_choice`` has already cleared tool_choice there.

    The result is what the router passes to ONE underlying attempt; the caller
    applies it inside the failover loop, so failover and usage tracking are
    unaffected.
    """
    if output_schema is None:
        return None
    if not _is_forced_tool_choice(model_settings.tool_choice):
        return output_schema
    if _input_has_tool_history(input):
        return output_schema
    return None


@dataclass
class ModelUsage:
    """Per-model call counters tracked by the router."""

    requests: int = 0
    succeeded: int = 0
    failed: int = 0


# --- Explicit Gemini client factory (never a global default client) ---

_shared_client: AsyncOpenAI | None = None
_underlying_cache: dict[str, OpenAIChatCompletionsModel] = {}
_failover_cache: dict[str, "FailoverModel"] = {}


def _gemini_api_key() -> str:
    """Read GEMINI_API_KEY, raising one clear sentence when it is missing."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise UserError(
            "GEMINI_API_KEY is not set. Add it to the project .env file "
            "(never commit it); the router refuses to build a Gemini client "
            "without it."
        )
    return key


def get_gemini_client() -> AsyncOpenAI:
    """Return the one shared explicit Gemini AsyncOpenAI client.

    Built with an explicit api_key (so the SDK never falls back to
    OPENAI_API_KEY) and max_retries=0, because the router does its own
    retry/failover across the chain.
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = AsyncOpenAI(
            base_url=GEMINI_BASE_URL,
            api_key=_gemini_api_key(),
            max_retries=0,
        )
    return _shared_client


def _underlying_model(name: str) -> OpenAIChatCompletionsModel:
    """Return the cached OpenAIChatCompletionsModel for one chain model."""
    model = _underlying_cache.get(name)
    if model is None:
        model = OpenAIChatCompletionsModel(model=name, openai_client=get_gemini_client())
        _underlying_cache[name] = model
    return model


def _reset_for_tests() -> None:
    """Drop every cached client/model. Intended for the test suite only."""
    global _shared_client
    _shared_client = None
    _underlying_cache.clear()
    _failover_cache.clear()


class FailoverModel(Model):
    """An SDK Model that transparently fails over down the chain.

    get_response (and stream_response, until the first event) delegate to the
    underlying model for the current chain position, with one mediation: a
    run's FIRST forced-tool turn goes out without the output schema (the
    Gemini OpenAI-compat first-turn conflict, resolved at the router — see the
    module-level note and :func:`_mediated_first_turn_output_schema`). On a
    failover trigger the router advances to the next chain model, puts the
    failed model on cooldown, records usage, and retries - the caller (and the
    agent holding this object) never sees the switch. The position persists
    across calls, so a mid-run switch is visible to every subsequent call.
    Only when every model in the chain fails does it raise ModelChainExhausted.
    """

    def __init__(
        self,
        chain: list[str],
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        models_factory: Any = None,
    ) -> None:
        if not chain:
            raise UserError("FailoverModel needs a non-empty model chain.")
        self.chain = list(dict.fromkeys(chain))
        self.cooldown_seconds = cooldown_seconds
        self._models_factory = models_factory or _underlying_model
        self._position = 0
        self.switch_count = 0
        self.cooldowns: dict[str, float] = {}
        """model name -> time.monotonic() deadline while it is skipped."""
        self._per_model: dict[str, ModelUsage] = {}

    @property
    def current_name(self) -> str:
        """The chain model this router would call right now."""
        return self.chain[self._position]

    @property
    def usage(self) -> dict[str, Any]:
        """Snapshot of per-model counters and switches, for logs and tests."""
        return {
            "current_model": self.current_name,
            "switches": self.switch_count,
            "per_model": {name: asdict(stat) for name, stat in self._per_model.items()},
        }

    def is_cooling(self, name: str) -> bool:
        """Whether `name` failed recently and is still on cooldown."""
        return self.cooldown_remaining(name) > 0.0

    def cooldown_remaining(self, name: str) -> float:
        """Seconds left on `name`'s cooldown (0.0 when it is eligible)."""
        deadline = self.cooldowns.get(name, 0.0)
        return max(0.0, deadline - time.monotonic())

    def _usage_for(self, name: str) -> ModelUsage:
        return self._per_model.setdefault(name, ModelUsage())

    def _skip_cooling(self) -> None:
        """Advance the position past models on cooldown (all cooling: stay)."""
        for _ in range(len(self.chain)):
            if not self.is_cooling(self.current_name):
                return
            self._position = (self._position + 1) % len(self.chain)

    def _advance(self) -> None:
        """Move to the next chain model, wrapping around, and log the switch."""
        self._position = (self._position + 1) % len(self.chain)
        self.switch_count += 1

    def _record_failure(self, name: str, exc: BaseException) -> None:
        usage = self._usage_for(name)
        usage.failed += 1
        self.cooldowns[name] = time.monotonic() + self.cooldown_seconds

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        """Try the current chain model, failing over down the chain on triggers.

        Each model in the chain gets at most one attempt per call; a
        non-failover error is raised untouched; an exhausted chain raises
        ModelChainExhausted with the last underlying error.
        """
        self._skip_cooling()
        last_error: BaseException | None = None
        for _ in range(len(self.chain)):
            name = self.current_name
            usage = self._usage_for(name)
            usage.requests += 1
            # Gemini OpenAI-compat mediation, applied per attempt inside the
            # failover loop: a first forced-tool turn goes out without the
            # output schema (see the module-level mediation note). Failover,
            # cooldowns and usage tracking are untouched.
            turn_output_schema = _mediated_first_turn_output_schema(
                model_settings, input, output_schema
            )
            try:
                response = await self._models_factory(name).get_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    turn_output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
            except Exception as exc:
                if not is_failover_trigger(exc):
                    raise
                self._record_failure(name, exc)
                last_error = exc
                self._advance()
                continue
            usage.succeeded += 1
            return response
        raise ModelChainExhausted(
            f"Every model in the failover chain failed "
            f"({', '.join(self.chain)}); last error: {last_error!r}"
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream with failover until the first event, then commit to a model.

        A failover trigger raised before the first event moves down the chain;
        once events flow, the stream is committed and errors propagate
        (switching mid-stream would duplicate output).
        """
        self._skip_cooling()
        last_error: BaseException | None = None
        for _ in range(len(self.chain)):
            name = self.current_name
            usage = self._usage_for(name)
            usage.requests += 1
            # Same mediation as get_response: streams carry it per attempt too,
            # so a first forced-tool turn never ships the JSON mime type.
            turn_output_schema = _mediated_first_turn_output_schema(
                model_settings, input, output_schema
            )
            iterator = self._models_factory(name).stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                turn_output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )
            try:
                first_event = await anext(iterator)
            except StopAsyncIteration:
                usage.succeeded += 1
                return
            except Exception as exc:
                if not is_failover_trigger(exc):
                    raise
                self._record_failure(name, exc)
                last_error = exc
                self._advance()
                continue
            usage.succeeded += 1
            yield first_event
            async for event in iterator:
                yield event
            return
        raise ModelChainExhausted(
            f"Every model in the failover chain failed "
            f"({', '.join(self.chain)}); last error: {last_error!r}"
        )


def get_model(name: str | None = None) -> Model:
    """Return a cached FailoverModel anchored at `name` (default: priority).

    Cheap by design: one shared AsyncOpenAI client, one underlying
    OpenAIChatCompletionsModel per name, one FailoverModel per anchor.
    """
    anchor = name or PRIORITY_MODEL
    model = _failover_cache.get(anchor)
    if model is None:
        model = FailoverModel(build_chain(anchor))
        _failover_cache[anchor] = model
    return model


def model_for_run(override: str | None = None) -> Model:
    """FR-7: a model for RunConfig(model=...) with an optional run-level override."""
    return get_model(override)


def current_model_name(model: Model | None = None) -> str:
    """Best-effort name of the model actually in use, for logs and ledgers."""
    if model is None:
        model = get_model()
    if isinstance(model, FailoverModel):
        return model.current_name
    named = getattr(model, "model", None)
    return named if isinstance(named, str) else type(model).__name__
