# Implementation Plan: Code Review Desk

## Overview

A fan-out/fan-in agentic pipeline: unified diff → per-file chunks (pre-model)
→ three cloned reviewer agents running concurrently → Desk agent with a Merge
specialist (as_tool) and a Remediation specialist (handoff) → typed report
with a latency/token footer → streamed into Chainlit, ledgered per run,
traced as one trace. All models flow through a failover router.

## Architecture Decisions

- **AD-1 — Gemini via OpenAI-compat layer.** `OpenAIChatCompletionsModel`
  over an explicit per-model `AsyncOpenAI(base_url=…googleapis…, api_key=…)`
  built by the router. Never `set_default_openai_client` (FR-1); never route
  model traffic to OpenAI.
- **AD-2 — Failover inside a `Model` wrapper.** A `FailoverModel` implements
  the SDK `Model` interface and delegates to the underlying chat-completions
  model; on 429/quota/rate-limit (or model-not-found) it advances down the
  chain, applies cooldowns, retries. Agents see one stable model object, so
  they never see the switch (FR-5 clones, FR-7 overrides all unaffected).
- **AD-3 — Cloning for reviewers.** One `base_reviewer` Agent
  (`output_type=list[Finding]`, ruleset tool, dynamic instructions callable,
  bounded `ModelSettings(tool_choice="required", temperature, max_tokens)`);
  three `.clone(name=…, instructions=…, model_settings=…)`. Clones share
  base tools/output wiring (viva q3) but own their instructions.
- **AD-4 — Concurrency at the pipeline layer.** `asyncio.gather` over three
  `Runner.run` calls of the clones (FR-5); progressive delivery via
  `asyncio.as_completed` yielding events to the UI as each reviewer lands.
- **AD-5 — Desk keeps the conversation.** A Desk agent carries
  `tools=[merge_specialist.as_tool(...)]` and
  `handoffs=[handoff(remediation_specialist, input_type=Escalation)]`
  (typed input names the triggering finding). Merge = tool because the Desk
  consumes a service and keeps speaking; remediation = handoff because the
  speaker must change to the fixer (FR-6 two-sentence justification lives in
  SPEC.md).
- **AD-6 — Guardrails on every voice.** The secret-quoting output guardrail
  is attached to all three reviewers, the Desk, and Remediation — whatever
  agent produces the final output, the run is protected; tripwire caught in
  the pipeline and rendered as a refusal (FR-8).
- **AD-7 — Ceiling via SDK-native handler.** `Runner.run(..., max_turns=N,
  error_handlers={"max_turns": handler})` returns a partial-review marker
  instead of raising into the caller (FR-9). Reviewers: 4 (min 2 = forced
  ruleset call + final; 4 = loop headroom). Desk: 8 (merge + handoff +
  summary).
- **AD-8 — Ledger as a custom Runner.** `LedgerRunner(Runner)` overrides
  `run()` to time the run, then append one JSON line
  (`ts, request_id, agent, ms, findings, model, tokens`) to `ledger.jsonl`.
  Instantiated once at startup as `LEDGER_RUNNER`; agent definitions never
  mention it (FR-11). Switching off = don't register it.
- **AD-9 — One trace per review.** `RunConfig(workflow_name="Code Review
  Desk", group_id=request_id)`; tracing export key set once at startup from
  `OPENAI_API_KEY`; missing key → `set_tracing_disabled(True)` + one
  sentence, app still runs (FR-13, NFR-1).
- **AD-10 — Event stream.** `pipeline.run_review(diff, context)` is an async
  generator of typed events (`ReviewStarted`, `ReviewerStarted`,
  `FindingsLanded`, `Merged`, `RemediationOffered`, `GuardrailRefused`,
  `PartialReview`, `ReviewComplete`) consumed identically by `cli.py` and
  `app.py` — the UI streams what actually happened, in order (FR-12).

## Task List

### Phase A: Foundations (parallel pair)
- [ ] Task 1: model router (`src/model_config.py` + tests)
- [ ] Task 2: intake (`src/intake.py` + tests, sample rulesets + diffs)

### Checkpoint A
- [ ] pytest green; router failover unit test proves switch; two-file diff → two chunks

### Phase B: Review core (sequential)
- [ ] Task 3: Finding, base reviewer, clones, dynamic instructions,
      forced ruleset tool, concurrent fan-out, turn ceiling
      (`src/review.py` + tests)

### Checkpoint B
- [ ] Concurrent vs sequential wall-clock evidence; typed list output; prompts differ per context

### Phase C: Specialists (sequential)
- [ ] Task 4: Desk + Merge as_tool + Remediation handoff + output guardrail
      (`src/specialists.py` + tests)

### Checkpoint C
- [ ] Merge dedupes/orders; handoff fires on critical security; planted key → refusal

### Phase D: Observability (parallel pair with Task 6 later; Task 5 first)
- [ ] Task 5: hooks, LedgerRunner, tracing, footer assembly
      (`src/observe.py`, `src/pipeline.py`, `cli.py` + tests)

### Checkpoint D
- [ ] One three-file review → ledger lines per run; footer shows real tokens; FR-7 override evidence

### Phase E: UI
- [ ] Task 6: Chainlit app + theme/CSS (`app.py`, `public/styles.css`,
      `.chainlit/config.toml`) consuming the pipeline event stream

### Checkpoint E
- [ ] App boots; manual paste-review works end to end

### Phase F: Verification & ship
- [ ] Task 7: DoD evidence runs (CLI), integrated-browser UI pass with
      screenshots, README, final push

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| `gemini-2.5-flash` 404 for this key | Reviews fail at default model | Router treats model-not-found as failover (verified by smoke test) |
| Gemini strict-schema quirks with `list[Finding]` root | Structured output errors | SDK wraps list root in `{"response": [...]}` — expected; keep schema simple; verify schema in test |
| `tool_choice="required"` blocks final answer | Reviewer loops | SDK `reset_tool_choice=True` (default) clears it after first call; verified in SDK source |
| Rate limits during evidence runs | Flaky demos | Router chain + cooldowns; review again on fallback model |
| Chainlit session-state races | Cross-talk between messages | Serialize per-session review with an asyncio lock |
| Trace export blocked (key/organization) | FR-13 unverifiable | Capture trace IDs + export success logs; report dashboard link and any error verbatim |

## Open Questions

None.
