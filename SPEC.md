# Spec: Code Review Desk

> Spec-driven build. This document is the binding authority for WHAT the Desk
> does; `tasks/plan.md` argues HOW; `tasks/todo.md` is the ordered task list.
> Committed before the first line of source code (provenance gate, NFR-5).

## Assumptions I'm making (surfaced before any spec content)

1. All model traffic goes to **Google Gemini** through its OpenAI-compatible
   endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`),
   driven by `GEMINI_API_KEY`; the OpenAI key is used **only** to export
   traces (FR-13). Verified by smoke call during environment setup.
2. The router's rate-limit table (model names and RPM tiers) is encoded
   verbatim as given, even though some listed models may not resolve for this
   key; **model-not-found is treated as a failover trigger** alongside rate
   limits so the chain still completes (observed: `gemini-2.5-flash` returns
   404 "no longer available to new users" for this key).
3. The project runs on Windows 10 / Git Bash / Python 3.14 with a venv.
4. Working branch is `main` (the mission explicitly directs commits and
   pushes to `main`), so implementation proceeds on `main` by consent of the
   brief that ordered it.
5. "Diff from a path" (FR-1) is the CLI path input; the Chainlit UI accepts a
   pasted diff (FR-12) through the same intake module.

## Objective

Build **Code Review Desk**: an agentic code-review pipeline. A unified diff
goes in; three reviewer agents — **security**, **tests**, **style** — read it
**at the same time**, each tuned differently; their findings merge into one
structured, typed report. A critical security finding hands the conversation
to a **Remediation** agent that proposes a fix. Nothing leaves the Desk that
quotes a secret it found in the diff. Findings appear in the UI **as they
land**, not in one lump at the end, and the whole review is a **single trace**
in which you can name the slowest reviewer.

Users: a developer who wants a structured second opinion on a change. Success:
every numbered requirement below is demonstrable on demand (the DoD table at
the bottom), including the browser-tested UI pass.

## Tech Stack

- **Agent framework:** OpenAI Agents SDK for Python (`openai-agents` 0.22.3,
  verified signatures against installed source) — agents, cloning, `as_tool`,
  handoffs (typed input), output guardrails, run/agent hooks, `RunConfig`,
  `error_handlers`, tracing.
- **Models:** Gemini via the SDK's model-configuration layer
  (`OpenAIChatCompletionsModel` over an explicit `AsyncOpenAI` client with
  Gemini's OpenAI-compatible base URL). **No global default client is ever
  set.** All models flow through the router (`src/model_config.py`).
- **UI:** Chainlit 2.11 with custom theme/CSS, severity-styled cards, footer,
  status messages.
- **Types:** pydantic v2 (`Finding`), dataclasses (`ReviewContext`), typed
  structures crossing every boundary.
- **Env:** Python 3.14, venv + `requirements.txt`, `python-dotenv`, pytest +
  pytest-asyncio.

## Router requirement (beyond the base PDF)

`src/model_config.py` is the **single source of truth** for model concerns:

- **Priority model:** user-settable (env `PRIORITY_MODEL`), default
  `gemini-2.5-flash` (the PDF's FR-1 model).
- **Rate-limit table (encoded verbatim):**
  - **15 RPM:** `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`
  - **5 RPM:** `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`,
    `gemini-3.5-flash`, `gemini-3-flash`, `gemini-2.5-flash`
- **Automatic failover:** on any limit hit — RPM, TPM, RPD, quota/429 (and
  model-not-found, see Assumptions) — the router switches to the next model in
  its chain, tracks per-model usage, applies cooldowns, retries
  transparently. Agents never see the switch.
- **Chain:** priority model first, then remaining table models in table order
  (15 RPM tier, then 5 RPM tier), duplicates removed.
- Every agent/reviewer obtains its model only through the router; zero
  hardcoded model names elsewhere. The router also satisfies FR-7 by exposing
  models for run-level `RunConfig(model=...)` overrides.
- Unit test simulates a rate-limit error and asserts the switch happens.

## Commands

```
Setup:    python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
Test:     .venv/Scripts/python -m pytest -q
UI:       .venv/Scripts/chainlit run app.py --port 8000
CLI demo: .venv/Scripts/python cli.py --diff examples/three_file.diff --repo demo --language python --ruleset default
CLI strict + cheaper re-run (FR-7):
          .venv/Scripts/python cli.py --diff examples/three_file.diff --repo demo --language python --ruleset default --strict --model-override gemini-3.5-flash-lite
```

## Project Structure

```
app.py               → Chainlit entry (UI module)
cli.py               → async CLI entry (evidence runs)
src/
  model_config.py    → model router: client factory, chain, failover, usage
  intake.py          → diff parsing/splitting, ReviewContext, ruleset loader
  review.py          → Finding, base reviewer, clones, fan-out, ceilings
  specialists.py     → Desk, Merge (as_tool), Remediation (handoff), guardrail
  observe.py         → hooks, LedgerRunner, tracing setup
  pipeline.py        → orchestration, ReviewReport, event stream
data/rulesets/       → default.json, strict.json (ruleset files)
public/styles.css    → custom Chainlit theme additions
tests/               → pytest unit tests (per module)
examples/            → sample diffs (clean, multi-file, planted-secret)
tasks/               → plan.md, todo.md (process artifacts)
.env / .env.example  → secrets (gitignored) / template
ledger.jsonl         → runtime output (gitignored)
```

## Code Style

Typed, small, sentence-returning. One real snippet showing the house style
for tools — failures become sentences the model can use, never exceptions
(NFR-4):

```python
@function_tool
async def read_ruleset(ctx: RunContextWrapper[ReviewContext]) -> str:
    """Load the repository review ruleset named by the review context."""
    try:
        return load_ruleset(ctx.context.ruleset_id)
    except FileNotFoundError:
        return (
            f"Error: ruleset '{ctx.context.ruleset_id}' could not be found. "
            "Continue the review using your built-in judgement and say so."
        )
```

Conventions: `snake_case` modules/functions; agents named in PascalCase
(`SecurityReviewer`); every agent declares bounded `ModelSettings`
(temperature + max_tokens — NFR-2); context objects flow through
`RunContextWrapper` and never appear in prompt text (FR-2); secrets only in
`.env` (NFR-1); async entry points (`await`) end to end (FR-1, FR-12).

## Testing Strategy

- **Framework:** pytest + pytest-asyncio; tests in `tests/`, one file per
  module (`test_model_config.py`, `test_intake.py`, `test_review.py`, …).
- **Levels:** unit tests for pure logic (diff splitting, router failover,
  guardrail detection, report rendering); integration tests with a fake model
  (monkeypatched `Model`) for fan-out concurrency, merge tool wiring, turn
  ceiling, ledger lines; live smoke against the real Gemini endpoint for the
  DoD evidence runs (marked, run manually, not in the default suite).
- **Coverage expectation:** every FR has at least one automated test or an
  explicit DoD evidence script; the UI additionally gets the mandatory
  integrated-browser black-box pass with screenshots.

## Boundaries

- **Always:** split the diff before any model call; pass `ReviewContext` to
  every run; run the three reviewers concurrently (`asyncio.gather`);
  bound every agent's model settings; catch tool/guardrail/turn-ceiling
  failures and convert them to messages or partial reports; append exactly
  one ledger line per run; commit per checkpoint and push milestones.
- **Ask first:** adding a dependency; changing the rate-limit table or
  priority model default; changing the ledger schema.
- **Never:** commit secrets or let them enter git history, the ledger, or any
  report; set a global default OpenAI client; route model traffic to OpenAI;
  put `ReviewContext` content into prompt text; let a tool raise into the
  runner; run the reviewers sequentially; cut FR-5, FR-8, or the browser UI
  pass.

## What this project deliberately will NOT do

1. **No git hosting integration** — no GitHub/PR fetching; input is a
   pasted or path-located unified diff, nothing else.
2. **No report persistence or diff-queueing** — beyond `ledger.jsonl` lines
   and Chainlit session state, nothing is stored; no concurrent review of
   multiple diffs.
3. **No auto-apply of fixes** — Remediation proposes patches as text; the
   Desk never writes to the user's tree.

## Requirements (in my own words)

- **FR-1 Intake.** Diff read from a path; split into per-file chunks before
  any model sees it; async entry point; empty/malformed diff → friendly
  message, never a traceback; no global default client anywhere.
- **FR-2 Context.** `ReviewContext(repo, language, ruleset_id, strictness)`
  dataclass passed to every run; tools read it through the
  `RunContextWrapper`; the generated tool schema has no wrapper parameter;
  grepping prompts finds no repository name.
- **FR-3 Typed findings.** `Finding(file, line, severity:
  critical|major|minor, message)`; reviewer `output_type=list[Finding]`;
  `final_output` is a plain Python list; the SDK's schema wrapper
  (`{"response": [...]}`) is expected and explainable.
- **FR-4 Per-run instructions.** Reviewer system prompt assembled at request
  time (dynamic instructions) from ruleset + language; terser under
  `strictness="strict"`; two contexts → two visibly different prompts,
  printable before any model call.
- **FR-5 Concurrent fan-out.** Security/tests/style are clones of one base
  reviewer (same object, different instructions/model settings), launched
  together and awaited as a group; wall clock ≈ slowest reviewer, not the
  sum; both numbers demonstrable.
- **FR-6 Specialists.** Merge specialist via `as_tool` (dedupe + severity
  order; Desk keeps the conversation); Remediation specialist via handoff on
  critical security findings; both fire on the right kind of diff.
  **Why tool vs transfer:** merging is a *service the Desk consumes* — the
  Desk must stay in charge of the report and its conversation, so the merge
  is a function call whose result comes back. Remediation is a *change of
  who speaks* — once a critical security hole exists, the user should be
  talking to the fixer, so control genuinely transfers.
- **FR-7 Run-level model swap.** Same reviewer object re-run under a
  different model via `RunConfig(model=...)`; no agent definition edited.
- **FR-8 Output guardrail.** Inspects the finished output; anything shaped
  like a credential found in the diff (API key/token/password) → refusal,
  never an echo; the tripwire is caught and reported, not crashed; a clean
  diff passes untouched; the catching line is identifiable.
- **FR-9 Controls.** The ruleset-consulting reviewer is forced to call the
  tool (`tool_choice="required"` with `reset_tool_choice=True`); tool
  failures return sentences, never raise into the runner; every review runs
  under a turn ceiling (reviewers: 4 turns — the minimum is 2, forced ruleset
  call + final answer, so 4 gives loop headroom; Desk: 8 — merge + optional
  handoff + summary) that raises `MaxTurnsExceeded`, is caught (SDK
  `error_handlers["max_turns"]`), and is reported as a **partial review**.
- **FR-10 Hooks.** Run-level hooks record per-reviewer latency and real
  token usage (from `RunContextWrapper.usage` — requests, input/output
  tokens; never estimated) into the report footer; agent-level hooks attach
  to exactly one reviewer (they see per-event detail — tool calls and LLM
  starts/stops — that run-level hooks aggregate away).
- **FR-11 Ledger.** Custom `LedgerRunner` appends one JSON line per run to
  `ledger.jsonl`, registered once at startup; no agent definition mentions
  it; removing the registration is the only change needed to switch it off.
- **FR-12 Chainlit.** Paste a diff → findings stream as they land; session
  state holds context + last report; a second diff reuses the context; the
  handler awaits the async pipeline.
- **FR-13 One trace.** Tracing exported under `OPENAI_API_KEY`
  (`set_tracing_export_api_key`); one review = one trace (`workflow_name`,
  `group_id=request_id`) with overlapping reviewer spans; the slowest
  reviewer is nameable from the trace and the footer.
- **NFR-1..5.** Secrets only in `.env` (gitignored before first commit;
  missing `GEMINI_API_KEY` → one clear sentence at startup, missing
  `OPENAI_API_KEY` → tracing disabled, app still runs); bounded model
  settings per agent; every review traceable and every run ledgered; tools
  never raise; `git log` proves spec-before-code.

## Success Criteria (mapping to DoD evidence)

| # | Requirement | Check |
|---|---|---|
| 1 | Spec preceded code | `git log` order, spec artifacts first |
| 2 | FR-1 | Two-file diff → two chunks before any model call |
| 3 | FR-2 | Tool schema has no wrapper; grep prompts for repo name |
| 4 | FR-3 | `final_output` iterable; criticals counted in one expression |
| 5 | FR-5 | Concurrent vs sequential wall clocks, side by side |
| 6 | FR-6 | Merge tool fires; remediation handoff fires on critical security |
| 7 | FR-7 | Same agent object, two models, no edits |
| 8 | FR-8 | Planted key → refusal, not a report; clean diff passes |
| 9 | FR-10 | Footer rows with real token counts from run context |
| 10 | FR-11 | One three-file review → ledger line per run |
| 11 | FR-12 + UI | Browser pass: progressive findings, cards, footer, refusal, handoff, session reuse, malformed-diff message |
| 12 | FR-13 | One trace, overlapping reviewer spans, slowest named |
| 13 | Router | Simulated 429 → automatic switch; review still completes |
| 14 | Provenance | Everything pushed to GitHub at milestones |

## Open Questions

None blocking. (The provider's deprecation of `gemini-2.5-flash` for this
key is resolved by Assumption 2 — the router fails over transparently and the
priority model remains the configured default.)
