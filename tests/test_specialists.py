"""Unit tests for src/specialists.py.

Fully offline: no network, no real model calls, no API key. Every agent is
only *constructed* and *wired* (tools, handoffs, guardrails, settings) — the
run-level behaviour is Task 5's pipeline integration; the merge/escalation
logic is tested at the pure-function level where it actually lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents import (
    FunctionTool,
    GuardrailFunctionOutput,
    ModelBehaviorError,
    RunContextWrapper,
)
from agents.guardrail import OutputGuardrail
from agents.handoffs import Handoff
from pydantic import ValidationError

from src.intake import ReviewContext
from src.model_config import RATE_LIMIT_TABLE, FailoverModel
from src.review import Finding
from src.specialists import (
    DESK_MAX_TURNS,
    SEVERITY_ORDER,
    Escalation,
    MergedFinding,
    ReviewSecrets,
    current_secrets,
    dedupe_and_order,
    extract_credential_patterns,
    guard_against_secrets,
    last_escalation,
    make_desk_agent,
    make_merge_tool,
    merge_specialist,
    remediation_specialist,
    reset_current_secrets,
    secret_guardrail,
    set_current_secrets,
    tag_findings,
)

# --- Fixtures and helpers -----------------------------------------------------


def make_ctx(
    repo: str = "desk-demo-repo-xyz",
    language: str = "python",
    ruleset_id: str = "default",
    strictness: str = "normal",
) -> ReviewContext:
    return ReviewContext(
        repo=repo, language=language, ruleset_id=ruleset_id, strictness=strictness
    )


def make_wrapper() -> RunContextWrapper[ReviewContext]:
    return RunContextWrapper(context=make_ctx())


def finding(
    file: str = "src/app.py",
    line: int = 10,
    severity: str = "major",
    message: str = "Something is wrong.",
) -> Finding:
    return Finding(file=file, line=line, severity=severity, message=message)


PLANTED_KEY = "sk-test-DO-NOT-USE-1234567890abcdef"


# --- dedupe_and_order: the deterministic merge rule ---------------------------


def test_duplicates_collapse_to_first_with_unioned_sources() -> None:
    findings = [
        *tag_findings([finding(message="Hardcoded API key.")], "SecurityReviewer"),
        *tag_findings([finding(message="Hardcoded API key.")], "TestsReviewer"),
        *tag_findings([finding(message="Hardcoded API key.")], "StyleReviewer"),
    ]

    merged = dedupe_and_order(findings)

    assert len(merged) == 1
    assert merged[0].file == "src/app.py"
    assert merged[0].line == 10
    assert merged[0].severity == "major"
    assert merged[0].message == "Hardcoded API key."  # the first message wins
    assert merged[0].sources == ["SecurityReviewer", "TestsReviewer", "StyleReviewer"]


def test_severity_ordering_critical_first_then_file_then_line() -> None:
    merged = dedupe_and_order(
        [
            finding(file="b.py", line=5, severity="minor"),
            finding(file="a.py", line=9, severity="major"),
            finding(file="a.py", line=1, severity="critical"),
            finding(file="a.py", line=4, severity="major"),
        ]
    )

    assert [(f.severity, f.file, f.line) for f in merged] == [
        ("critical", "a.py", 1),
        ("major", "a.py", 4),
        ("major", "a.py", 9),
        ("minor", "b.py", 5),
    ]


def test_same_file_line_different_severity_keeps_highest_and_merges_messages() -> None:
    merged = dedupe_and_order(
        [
            finding(severity="minor", message="Unused variable."),
            finding(severity="critical", message="Hardcoded credential."),
        ]
    )

    assert len(merged) == 1
    assert merged[0].severity == "critical"
    assert merged[0].message == "Unused variable.; Hardcoded credential."


def test_same_severity_duplicate_keeps_first_message_not_merged_text() -> None:
    merged = dedupe_and_order(
        [
            finding(severity="major", message="First wording."),
            finding(severity="major", message="Second wording."),
        ]
    )

    assert len(merged) == 1
    assert merged[0].message == "First wording."


def test_source_annotation_flows_through_and_defaults_empty() -> None:
    tagged = tag_findings([finding()], "SecurityReviewer")
    plain = finding(file="src/other.py", line=20)

    merged = dedupe_and_order([*tagged, plain])

    assert isinstance(merged[0], MergedFinding)
    by_file = {m.file: m for m in merged}
    assert by_file["src/app.py"].sources == ["SecurityReviewer"]
    assert by_file["src/other.py"].sources == []  # single-source default works


def test_merged_finding_keeps_finding_fields_identical() -> None:
    base_fields = set(Finding.model_fields)
    assert set(MergedFinding.model_fields) == base_fields | {"sources"}
    assert MergedFinding.model_fields["sources"].default == []


def test_empty_input_merges_to_empty_list() -> None:
    assert dedupe_and_order([]) == []


def test_severity_order_covers_every_literal_severity() -> None:
    assert SEVERITY_ORDER == {"critical": 0, "major": 1, "minor": 2}


# --- extract_credential_patterns: one rule set for diff and output ------------


def test_finds_sk_style_keys() -> None:
    matches = extract_credential_patterns(f"API_KEY = '{PLANTED_KEY}'")
    assert any(match.startswith("sk-") for match in matches)


def test_finds_aws_access_key_ids() -> None:
    matches = extract_credential_patterns("aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
    assert "AKIAIOSFODNN7EXAMPLE" in matches


def test_finds_password_assignments() -> None:
    assert extract_credential_patterns('password = "hunter2-secret"')
    assert extract_credential_patterns("api_key: 'abc123'")


def test_finds_long_high_entropy_tokens() -> None:
    token = "aBcD1234EfGh5678IjKl90Mn"  # 24 chars, mixed case + digits
    assert token in extract_credential_patterns(f"token = {token}")


def test_finds_github_and_slack_tokens() -> None:
    # Assembled at runtime so the source never contains a contiguous
    # provider-token format (GitHub push protection rejects those).
    github_token = "gh" + "p_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"
    slack_token = "xo" + "xb-123456789-abcdefghijklmnop"
    assert extract_credential_patterns(github_token)
    assert extract_credential_patterns(slack_token)


def test_finds_bearer_tokens() -> None:
    assert extract_credential_patterns("Authorization: Bearer Zx9-Qp7._Lm4/Te2B=")


def test_ignores_normal_code_lines() -> None:
    normal = "\n".join(
        [
            "def calculate_total(price, quantity):",
            "    total = price * quantity",
            "    return round(total, 2)",
            'EMAIL = "user@example.com"',
            'password = ""',  # empty value: not a credential
            "api_key = os.environ.get('API_KEY')",  # unquoted lookup, not a value
            "import os",
        ]
    )
    assert extract_credential_patterns(normal) == []


def test_ignores_git_shas_by_design() -> None:
    # 40-char lowercase hex (2 char classes) — documented layer-(a) tradeoff;
    # layer (b) exact-matches whatever the pipeline plants from the diff.
    sha = "e5fa44f2b31c1fb553b6021e7360d07d5d91ff5e"
    assert extract_credential_patterns(f"introduced in commit {sha}") == []


def test_matches_are_deduped_and_position_ordered() -> None:
    matches = extract_credential_patterns(f"{PLANTED_KEY} then {PLANTED_KEY}")
    assert matches == [PLANTED_KEY]


# --- secret guardrail (FR-8) --------------------------------------------------


async def test_guardrail_trips_on_credential_shaped_string() -> None:
    output = f"Fix the leak: the key {PLANTED_KEY} must move to the environment."

    result = await _run_guardrail(output)

    assert result.tripwire_triggered is True
    assert result.output_info["clean"] is False
    assert "generic-shape" in result.output_info["layers"]


async def test_guardrail_output_info_is_redacted_never_full_secret() -> None:
    result = await _run_guardrail(f"The key is {PLANTED_KEY} in config.py.")

    info_json = json.dumps(result.output_info)
    assert PLANTED_KEY not in info_json  # the full secret never escapes
    assert result.output_info["matches"]  # something tripped...
    assert all(PLANTED_KEY != match for match in result.output_info["matches"])
    assert result.output_info["matches"][0].startswith("sk-t")  # masked, not empty


async def test_guardrail_passes_clean_output() -> None:
    result = await _run_guardrail(
        "All clear: 3 minor style findings; move the API key to an env var."
    )

    assert result.tripwire_triggered is False
    assert result.output_info["clean"] is True


async def test_guardrail_checks_finding_list_output_by_joined_messages() -> None:
    findings = [
        finding(message="Rotating leaked key sk-abcdefgh12345678 is required."),
        finding(file="src/other.py", line=3, severity="minor", message="Unused import."),
    ]

    result = await _run_guardrail(findings)

    assert result.tripwire_triggered is True


async def test_guardrail_planted_secret_trips_via_contextvar_exact_match() -> None:
    planted = "PASTA-PLANTED-SECRET-42"  # fits no generic shape
    assert (await _run_guardrail(f"value {planted} here")).tripwire_triggered is False

    token = set_current_secrets([planted])
    try:
        result = await _run_guardrail(f"value {planted} here")
        assert result.tripwire_triggered is True
        assert "planted-secret" in result.output_info["layers"]
        assert planted not in json.dumps(result.output_info)
    finally:
        reset_current_secrets(token)
    assert current_secrets.get() is None


async def test_guardrail_planted_patterns_do_not_leak_across_tasks() -> None:
    # Contextvars are copied per asyncio task: a sibling review task created
    # BEFORE the planting keeps its own (empty) view, so concurrent reviews
    # cannot cross-contaminate their secret sets.
    import asyncio

    gate = asyncio.Event()
    observed: dict[str, ReviewSecrets | None] = {}

    async def sibling_review() -> None:
        await gate.wait()  # its context was copied before any planting
        observed["secrets"] = current_secrets.get()

    sibling = asyncio.create_task(sibling_review())
    token = set_current_secrets(["PASTA-PLANTED-SECRET-42"])
    try:
        gate.set()
        await sibling
        assert observed["secrets"] is None  # sibling unaffected by the planting
        assert isinstance(current_secrets.get(), ReviewSecrets)  # this task sees it
    finally:
        reset_current_secrets(token)
    assert current_secrets.get() is None


# --- Escalation: the typed handoff input ---------------------------------------


def test_escalation_round_trips_json() -> None:
    escalation = Escalation(
        finding=finding(severity="critical", message="Hardcoded credential."),
        reviewer="SecurityReviewer",
        reason="Attacker-readable key needs a patch now.",
    )

    restored = Escalation.model_validate_json(escalation.model_dump_json())

    assert restored == escalation
    assert restored.finding.file == "src/app.py"
    assert restored.reviewer == "SecurityReviewer"


def test_escalation_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        Escalation.model_validate({"reviewer": "SecurityReviewer"})


# --- Agent wiring: tools, handoffs, guardrails, bounded settings ---------------


def test_desk_has_typed_merge_tool_without_context_wrapper() -> None:
    desk = make_desk_agent()
    tools = [tool for tool in desk.tools if tool.name == "merge_findings"]

    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, FunctionTool)
    schema = tool.params_json_schema
    assert set(schema["properties"]) == {"findings"}  # no context wrapper (FR-2)
    assert schema["required"] == ["findings"]
    assert "JSON array" in tool.description


def test_merge_tool_builds_standalone_with_same_schema() -> None:
    tool = make_merge_tool()

    assert tool.name == "merge_findings"
    assert set(tool.params_json_schema["properties"]) == {"findings"}


def test_desk_has_one_handoff_targeting_remediation_with_escalation_input() -> None:
    desk = make_desk_agent()

    assert len(desk.handoffs) == 1
    handoff_obj = desk.handoffs[0]
    assert isinstance(handoff_obj, Handoff)
    assert handoff_obj.agent_name == "RemediationSpecialist"
    schema = handoff_obj.input_json_schema
    assert set(schema["properties"]) == {"finding", "reviewer", "reason"}
    assert set(schema["required"]) == {"finding", "reviewer", "reason"}
    assert "remediation engineer" in handoff_obj.tool_description


async def test_handoff_validates_escalation_json_and_records_it() -> None:
    desk = make_desk_agent()
    handoff_obj = desk.handoffs[0]
    escalation = Escalation(
        finding=finding(severity="critical", message="Hardcoded credential."),
        reviewer="SecurityReviewer",
        reason="Needs a proposed patch.",
    )

    target = await handoff_obj.on_invoke_handoff(
        make_wrapper(), escalation.model_dump_json()
    )

    assert target.name == "RemediationSpecialist"
    recorded = last_escalation.get()
    assert recorded == escalation  # on_handoff validated the JSON into Escalation


async def test_handoff_rejects_input_not_matching_escalation() -> None:
    desk = make_desk_agent()
    handoff_obj = desk.handoffs[0]

    with pytest.raises((ModelBehaviorError, ValidationError)):
        await handoff_obj.on_invoke_handoff(make_wrapper(), '{"finding": 1}')


@pytest.mark.parametrize("agent", [make_desk_agent(), merge_specialist, remediation_specialist])
def test_secret_guardrail_attached_to_every_specialist_voice(
    agent: object,
) -> None:
    guardrails = agent.output_guardrails  # type: ignore[attr-defined]
    assert len(guardrails) == 1
    assert isinstance(guardrails[0], OutputGuardrail)
    assert guardrails[0].get_name() == "guard_against_secrets"
    assert guardrails[0] is secret_guardrail


@pytest.mark.parametrize(
    ("agent", "temperature", "max_tokens"),
    [
        (lambda: make_desk_agent(), 0.2, 1024),
        (lambda: merge_specialist, 0.0, 2048),
        (lambda: remediation_specialist, 0.2, 2048),
    ],
)
def test_all_specialists_have_bounded_model_settings(
    agent: object, temperature: float, max_tokens: int
) -> None:
    settings = agent().model_settings  # type: ignore[operator]
    assert settings.temperature == temperature
    assert settings.max_tokens == max_tokens


@pytest.mark.parametrize(
    ("agent", "output_type"),
    [
        (lambda: make_desk_agent(), str),
        (lambda: merge_specialist, list[MergedFinding]),
        (lambda: remediation_specialist, str),
    ],
)
def test_output_types_are_typed_per_role(agent: object, output_type: object) -> None:
    assert agent().output_type == output_type  # type: ignore[operator]


def test_agent_names_match_the_spec() -> None:
    desk = make_desk_agent()
    assert desk.name == "ReviewDesk"
    assert merge_specialist.name == "MergeSpecialist"
    assert remediation_specialist.name == "RemediationSpecialist"


def test_desk_turn_ceiling_is_eight() -> None:
    # merge tool call + structured continuation + optional handoff + summary
    # = 4-6 typical; 8 leaves headroom without letting loops run (FR-9).
    assert DESK_MAX_TURNS == 8


def test_models_come_only_from_the_router_no_hardcoded_names() -> None:
    source = Path("src", "specialists.py").read_text(encoding="utf-8")
    for model_name in RATE_LIMIT_TABLE:
        assert model_name not in source
    assert "gemini" not in source.lower()
    assert "gpt" not in source.lower()

    desk = make_desk_agent()
    assert isinstance(desk.model, FailoverModel)
    assert isinstance(merge_specialist.model, FailoverModel)
    assert isinstance(remediation_specialist.model, FailoverModel)


def test_desk_instructions_use_context_but_never_the_repo_name() -> None:
    desk = make_desk_agent()
    assert callable(desk.instructions)
    prompt = desk.instructions(make_wrapper(), desk)  # type: ignore[call-arg]

    assert "python" in prompt  # language: fine (same as reviewers)
    assert "normal" in prompt  # strictness: fine
    assert "desk-demo-repo-xyz" not in prompt  # repo never enters prompt text (FR-2)
    assert "merge_findings" in prompt
    assert "remediation engineer" in prompt


# --- Helpers -------------------------------------------------------------------


async def _run_guardrail(output: object) -> GuardrailFunctionOutput:
    """Run the guardrail the way the SDK does; unwrap the GuardrailFunctionOutput.

    ``OutputGuardrail.run`` returns an ``OutputGuardrailResult`` whose
    ``.output`` holds the ``GuardrailFunctionOutput`` (tripwire + output_info).
    """
    result = await guard_against_secrets.run(make_wrapper(), make_desk_agent(), output)
    return result.output


def test_tripwire_exception_is_reexported_for_the_pipeline() -> None:
    # Task 5 catches the tripwire; the catching line imports it from here.
    from src.specialists import OutputGuardrailTripwireTriggered

    from agents import OutputGuardrailTripwireTriggered as _sdk_one

    assert OutputGuardrailTripwireTriggered is _sdk_one
