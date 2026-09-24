# Task List: Code Review Desk

Ordered by dependency; each task independently verifiable; each names the
requirement it serves. Checkpoints between phases.

## Phase A: Foundations

- [ ] Task 1: Model router
  - Acceptance: `src/model_config.py` exposes `get_model(name)` /
    `model_for_run()` returning SDK `Model` objects over an explicit Gemini
    client; rate-limit table encoded verbatim (15 RPM: gemini-3.5-flash-lite,
    gemini-3.1-flash-lite; 5 RPM: gemini-3.8-flash, gemini-3.7-flash,
    gemini-3.6-flash, gemini-3.5-flash, gemini-3-flash, gemini-2.5-flash);
    priority model from env with default gemini-2.5-flash; simulated 429 →
    automatic switch with cooldown + usage tracking; no other module
    hardcodes a model name.
  - Verify: `.venv/Scripts/python -m pytest tests/test_model_config.py -q`
  - Files: `src/model_config.py`, `tests/test_model_config.py`
  - Serves: FR-1 (model on agent), FR-7 (run-level swap), router requirement

- [ ] Task 2: Diff intake + context + rulesets
  - Acceptance: `split_diff(text)` → list of per-file chunks before any model
    call; empty/malformed diff → `DiffError` with message (no traceback);
    `ReviewContext` dataclass exactly as specified; `load_ruleset(id)` reads
    `data/rulesets/<id>.json`, failures become sentences; sample diffs
    (clean two-file, three-file with planted secret + critical security
    issue) and ruleset files exist.
  - Verify: `.venv/Scripts/python -m pytest tests/test_intake.py -q`
  - Files: `src/intake.py`, `data/rulesets/*.json`, `examples/*.diff`,
    `tests/test_intake.py`
  - Serves: FR-1, FR-2, FR-9 (failing-tool sentence), NFR-4

## Checkpoint A: pytest green; evidence for FR-1 chunking + router failover

## Phase B: Review core

- [ ] Task 3: Reviewers + concurrency
  - Acceptance: `Finding` pydantic model; base reviewer with
    `output_type=list[Finding]`, dynamic instructions from ruleset+language
    (terser under strict), forced ruleset tool (`tool_choice="required"`),
    bounded model settings; three clones; `run_reviewers_concurrently()`
    awaits the group (`asyncio.gather`); sequential comparison helper; turn
    ceiling 4 with partial-review catch; schema wrapper visible in test.
  - Verify: `.venv/Scripts/python -m pytest tests/test_review.py -q` + live
    wall-clock evidence script
  - Files: `src/review.py`, `tests/test_review.py`
  - Serves: FR-3, FR-4, FR-5, FR-9

## Checkpoint B: concurrent ≈ slowest reviewer, not sum (both numbers shown)

## Phase C: Specialists

- [ ] Task 4: Merge, Remediation, guardrail
  - Acceptance: Desk agent with `merge_specialist.as_tool()` (dedupe +
    severity order; Desk keeps conversation) and
    `handoff(remediation_specialist, input_type=Escalation)` (typed input
    naming the triggering finding); output guardrail refusing credential-
    shaped text planted in the diff, attached to reviewers + Desk +
    Remediation; pipeline catches tripwire → refusal message; Desk ceiling 8.
  - Verify: `.venv/Scripts/python -m pytest tests/test_specialists.py -q`
  - Files: `src/specialists.py`, `tests/test_specialists.py`
  - Serves: FR-6, FR-8

## Checkpoint C: merge fires on any multi-reviewer overlap; handoff fires on critical security; planted key → refusal

## Phase D: Observability

- [ ] Task 5: Hooks, ledger, tracing, pipeline, CLI
  - Acceptance: run-level `RunHooks` recording per-reviewer latency +
    real usage from run context; agent-level hooks on exactly one reviewer;
    `LedgerRunner` (one JSON line per run to `ledger.jsonl`, registered once
    at startup, unmentioned by agents); tracing exported under
    `OPENAI_API_KEY` with one trace per review (`workflow_name`,
    `group_id=request_id`), disabled-with-a-sentence when key missing;
    `ReviewReport` + footer renderer; `pipeline.run_review()` async event
    stream; `cli.py` async entry with `--model-override` (FR-7).
  - Verify: `.venv/Scripts/python -m pytest tests/test_observe.py
    tests/test_pipeline.py -q`; CLI evidence run
  - Files: `src/observe.py`, `src/pipeline.py`, `cli.py`,
    `tests/test_observe.py`, `tests/test_pipeline.py`
  - Serves: FR-7, FR-9 (ceiling catch), FR-10, FR-11, FR-13, NFR-3

## Checkpoint D: ledger line per run; footer rows carry real token counts

## Phase E: UI

- [ ] Task 6: Chainlit app
  - Acceptance: paste diff → status messages → findings streamed as they
    land (per-reviewer), severity-styled cards, latency/token footer, header
    (repo/language/strictness), session state (context + last report) reused
    on second diff, handler awaits async pipeline, malformed/empty diff →
    friendly message; custom theme CSS. Additive to FR-12 behaviour.
  - Verify: app boots (`chainlit run app.py`); browser pass in Task 7
  - Files: `app.py`, `public/styles.css`, `.chainlit/config.toml`
  - Serves: FR-12 + UI enhancements

## Checkpoint E: end-to-end paste-review works locally

## Phase F: Verification & ship

- [ ] Task 7: DoD evidence + browser pass + ship
  - Acceptance: all 10 DoD items executed with recorded evidence; defects
    fixed; README with run instructions; all work committed and pushed.
  - Verify: evidence transcripts + screenshots in `evidence/`
  - Files: `evidence/`, `README.md`
  - Serves: every FR (final gate)
