# Capability Map: Code Review Desk

One request — "build the Code Review Desk" — bundles several independently
testable capabilities. Per spec-driven-development Phase 0, this map fixes
module boundaries, dependency direction, and build order before any module
spec is written. Module ids are stable and never renamed.

| Module id | Responsibility | Depends on |
|---|---|---|
| `model-config` | The only place model concerns live: Gemini client factory, model chain + rate-limit table, priority model, automatic failover (429/quota/TPM/RPD/model-not-found), per-model usage tracking and cooldowns, run-level model override (FR-7). No hardcoded model names anywhere else. | — |
| `intake` | Read a unified diff from a path (async entry), split into per-file chunks **before any model call**, reject empty/malformed diffs with a message (never a traceback); `ReviewContext` dataclass; ruleset file loading. | — |
| `review` | `Finding` pydantic model; one base reviewer agent (dynamic instructions from ruleset + language + strictness, forced ruleset tool, `output_type=list[Finding]`, bounded model settings); three clones (security/tests/style); concurrent fan-out via `asyncio.gather`; per-review turn ceiling with partial-review reporting. | `model-config`, `intake` |
| `specialists` | Desk orchestrator agent that keeps the conversation: Merge specialist exposed via `as_tool` (dedupe + severity order); Remediation specialist reached by typed-input **handoff** on critical security findings; output guardrail that refuses anything quoting a secret found in the diff. | `review` |
| `observe` | Run-level hooks (per-reviewer latency + real token usage from run context), agent-level hooks on exactly one reviewer, custom `LedgerRunner` appending one JSON line per run to `ledger.jsonl` (registered once at startup, no agent mentions it), tracing setup exported under `OPENAI_API_KEY`. | — |
| `pipeline` | Orchestration: intake → concurrent fan-out → Desk (merge tool / remediation handoff) → guardrail → typed `ReviewReport` with per-reviewer footer; async event stream for progressive UI consumption; CLI entry (`cli.py`). | `intake`, `review`, `specialists`, `observe`, `model-config` |
| `ui` | Chainlit app: paste a diff, findings stream as they land, severity-styled cards, latency/token footer, status messages, header (repo/language/strictness), session-state context reuse, custom theme/CSS. | `pipeline` |

Build order: `model-config` ∥ `intake` → `review` → `specialists` →
`observe` ∥ `ui` (after `pipeline` exists) → integration.

Pairwise parallel dispatches planned (≤ 2 concurrent, no shared state):
`model-config` ∥ `intake`; `observe` ∥ `ui`.

Module specs live in `SPEC.md` (single cohesive spec with per-capability
success criteria); the task list lives in `tasks/todo.md`, the plan in
`tasks/plan.md`.
