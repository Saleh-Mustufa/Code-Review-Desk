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

## Task 1: Model router

**Description:** Create `src/model_config.py`, the only place model concerns
live: explicit Gemini `AsyncOpenAI` client factory, rate-limit table, chain
building from a user-settable priority model, a `FailoverModel` implementing
the SDK `Model` interface that transparently switches down the chain on
rate-limit/quota errors (and model-not-found), per-model usage tracking and
cooldowns, and helpers for run-level overrides. Plus unit tests.

**Acceptance criteria:**
- [ ] `get_model(name)` / `model_for_run(override=None)` return SDK `Model` objects over an explicit Gemini client (no global default client)
- [ ] Rate-limit table encoded verbatim (15 RPM: gemini-3.5-flash-lite, gemini-3.1-flash-lite; 5 RPM: gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-3-flash, gemini-2.5-flash)
- [ ] Priority model from env `PRIORITY_MODEL`, default gemini-2.5-flash
- [ ] Unit test simulates a 429 and asserts the switch to the next chain model; cooldown + usage tracked
- [ ] No other module hardcodes a model name

**Verification:** `.venv/Scripts/python -m pytest tests/test_model_config.py -q`

**Dependencies:** None

**Files likely touched:** `src/model_config.py`, `tests/test_model_config.py`

**Estimated scope:** Small (2 files)

## Task 2: Diff intake + context + rulesets

**Description:** Create `src/intake.py`: async diff entry, `split_diff()`
into per-file chunks before any model call, `DiffError` for empty/malformed
input with a message (never a traceback), the `ReviewContext` dataclass
exactly as specified, and `load_ruleset()` reading `data/rulesets/<id>.json`
with failures converted to sentences. Add sample diffs (clean two-file;
three-file with planted secret + critical security issue) and ruleset files.

**Acceptance criteria:**
- [ ] Two-file diff → exactly two chunks; three-file → three
- [ ] Empty/malformed diff → `DiffError` with a friendly message, no traceback
- [ ] `ReviewContext(repo, language, ruleset_id, strictness="normal")` matches the PDF shape
- [ ] `load_ruleset()` returns sentence-error on missing file
- [ ] Example diffs + default.json/strict.json rulesets exist

**Verification:** `.venv/Scripts/python -m pytest tests/test_intake.py -q`

**Dependencies:** None

**Files likely touched:** `src/intake.py`, `data/rulesets/*.json`, `examples/*.diff`, `tests/test_intake.py`

**Estimated scope:** Small (5 files, mostly fixtures)

### Checkpoint A: pytest green; FR-1 chunking + router failover evidence

### Phase B: Review core (sequential)

## Task 3: Reviewers + concurrency

**Description:** Create `src/review.py`: `Finding` pydantic model; base
reviewer Agent with `output_type=list[Finding]`, dynamic instructions
assembled from ruleset + language (terser under strict), forced ruleset tool
(`tool_choice="required"`, `reset_tool_choice=True`), bounded model settings;
three clones (SecurityReviewer, TestsReviewer, StyleReviewer);
`run_reviewers_concurrently()` awaiting the group with `asyncio.gather`;
sequential comparison helper for wall-clock evidence; turn ceiling 4 with
SDK `error_handlers` max_turns catch producing a partial review.

**Acceptance criteria:**
- [ ] Reviewer `output_type=list[Finding]`; `final_output` is a plain list
- [ ] Two contexts produce two visibly different resolved prompts; prompt printable before any model call
- [ ] Generated schema shows the `{"response": [...]}` wrapper; tool schema has no context wrapper parameter
- [ ] Three reviewers launched together; concurrent wall clock ≈ slowest (fake-model test)
- [ ] Turn ceiling caught → partial review message, not a raise

**Verification:** `.venv/Scripts/python -m pytest tests/test_review.py -q`

**Dependencies:** Task 1, Task 2

**Files likely touched:** `src/review.py`, `tests/test_review.py`

**Estimated scope:** Medium (2 files)

### Checkpoint B: concurrent ≈ slowest reviewer, not sum (both numbers shown)

### Phase C: Specialists (sequential)

## Task 4: Merge, Remediation, guardrail

**Description:** Create `src/specialists.py`: Desk orchestrator agent with
`merge_specialist.as_tool()` (dedupe + severity order; Desk keeps the
conversation) and `handoff(remediation_specialist, input_type=Escalation)`
(typed input naming the triggering finding); secret-quoting output guardrail
attached to reviewers + Desk + Remediation; pipeline-level catch of the
tripwire rendering a refusal.

**Acceptance criteria:**
- [ ] Merge specialist exposed via `as_tool`; dedupes overlapping findings and orders by severity
- [ ] Remediation reached by handoff with typed `Escalation` input on critical security findings
- [ ] Output guardrail refuses credential-shaped output; clean output passes; catching line identifiable
- [ ] Desk turn ceiling 8

**Verification:** `.venv/Scripts/python -m pytest tests/test_specialists.py -q`

**Dependencies:** Task 3

**Files likely touched:** `src/specialists.py`, `tests/test_specialists.py`

**Estimated scope:** Medium (2 files)

### Checkpoint C: merge + handoff fire on the right diffs; planted key → refusal

### Phase D: Observability (sequential)

## Task 5: Hooks, ledger, tracing, pipeline, CLI

**Description:** Create `src/observe.py` (run-level hooks recording
per-reviewer latency + real usage from run context; agent-level hooks on
exactly one reviewer; `LedgerRunner` appending one JSON line per run to
`ledger.jsonl`, registered once at startup; tracing setup exported under
`OPENAI_API_KEY`, disabled-with-a-sentence if missing), `src/pipeline.py`
(`ReviewReport` + footer renderer; `run_review()` async event stream;
sequential/concurrent comparison), and `cli.py` (async entry with
`--model-override`, `--strict`).

**Acceptance criteria:**
- [ ] Footer rows carry latency ms + real token counts from run context (not estimated)
- [ ] One three-file review → one ledger line per run; removing registration = only change to switch off
- [ ] One review = one trace (workflow_name, group_id=request_id)
- [ ] FR-7: same reviewer object re-run under `RunConfig(model=...)` override, no agent edits
- [ ] CLI async entry; malformed diff → message not traceback

**Verification:** `.venv/Scripts/python -m pytest tests/test_observe.py tests/test_pipeline.py -q`

**Dependencies:** Tasks 1-4

**Files likely touched:** `src/observe.py`, `src/pipeline.py`, `cli.py`, `tests/test_observe.py`, `tests/test_pipeline.py`

**Estimated scope:** Medium (5 files)

### Checkpoint D: ledger line per run; footer rows carry real token counts

### Phase E: UI (sequential)

## Task 6: Chainlit app

**Description:** Create `app.py` consuming the pipeline event stream: paste
diff → status messages → findings streamed as they land with severity-styled
cards → latency/token footer → header (repo/language/strictness); session
state (context + last report) reused on a second diff; handler awaits the
async pipeline; malformed/empty diff → friendly message; custom theme via
`public/styles.css` + `.chainlit/config.toml`.

**Acceptance criteria:**
- [ ] Findings appear progressively (per reviewer), not one lump
- [ ] Severity cards styled (critical/major/minor colours + badges)
- [ ] Footer renders latency + tokens; header renders settings
- [ ] Session-state context reuse on second diff
- [ ] Empty/malformed diff → friendly message, no crash

**Verification:** `chainlit run app.py` boots; browser pass in Task 7

**Dependencies:** Task 5

**Files likely touched:** `app.py`, `public/styles.css`, `.chainlit/config.toml`

**Estimated scope:** Medium (3 files)

### Checkpoint E: end-to-end paste-review works locally

### Phase F: Verification & ship (sequential)

## Task 7: DoD evidence + browser pass + ship

**Description:** Execute all 10 DoD items with recorded evidence; fix
defects; run the mandatory integrated-browser UI pass with screenshots;
write README; commit and push everything.

**Acceptance criteria:**
- [ ] All 10 DoD checks executed with evidence in `evidence/`
- [ ] Browser pass screenshots: progressive findings, styled cards, footer, refusal, handoff, session reuse, malformed diff
- [ ] README complete; all commits pushed

**Verification:** evidence transcripts + screenshots; `git log`; pushed remote

**Dependencies:** Task 6

**Files likely touched:** `evidence/*`, `README.md`

**Estimated scope:** Medium

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
