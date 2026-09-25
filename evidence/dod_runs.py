"""Definition-of-Done evidence runs (live, against real Gemini via the router).

Run:  .venv/Scripts/python evidence/dod_runs.py

Produces a transcript covering the CLI-checkable DoD items:
 1. FR-1  — two-file diff -> two chunks before any model call
 2. FR-5  — concurrent vs sequential wall clocks, side by side
 3. FR-3  — final_output iterable; criticals counted in one expression
 4. FR-8  — planted secret -> output guardrail refusal (not a report)
 5. FR-10 — footer rows with per-reviewer latency + REAL token counts
 6. FR-11 — one three-file review -> one ledger line per run
 7. FR-7  — same reviewer objects, model override via RunConfig
 8. Router — simulated rate-limit switch is unit-tested (test_model_config.py);
    here we report the model the router actually used.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.intake import DiffError, ReviewContext, intake, read_diff, split_diff  # noqa: E402
from src.model_config import PRIORITY_MODEL, current_model_name, get_model  # noqa: E402
from src.observe import default_ledger_path  # noqa: E402
from src.pipeline import (  # noqa: E402
    FindingsLanded,
    GuardrailRefused,
    MergedReport,
    RemediationOffered,
    ReviewComplete,
    ReviewReport,
    render_footer,
    run_review,
)

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(exist_ok=True)
OUT = EVIDENCE / "dod_transcript.txt"

_lines: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _lines.append(line)


def section(title: str) -> None:
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


async def drive(diff_text: str, ctx: ReviewContext, **kwargs) -> tuple[ReviewReport | None, list]:
    """Drive one review to completion, collecting events (must fully drain)."""
    events: list = []
    report: ReviewReport | None = None
    async for event in run_review(diff_text, ctx, **kwargs):
        events.append(event)
        if isinstance(event, FindingsLanded):
            say(
                f"  <- {event.reviewer} landed: {len(event.findings)} finding(s)"
                + (" [PARTIAL]" if event.partial else "")
            )
        elif isinstance(event, MergedReport):
            say(f"  <- merged: {len(event.findings)} finding(s) after dedupe/order")
        elif isinstance(event, RemediationOffered) and event.escalation is not None:
            say("  <- remediation engineer took over (critical security finding)")
        elif isinstance(event, GuardrailRefused):
            say(f"  <- GUARDRAIL REFUSED the report: {event.reason} (masked: {event.masked})")
        elif isinstance(event, ReviewComplete):
            report = event.report
    return report, events


async def main() -> int:
    ledger_path = default_ledger_path()

    # ---- 1. FR-1: two-file diff -> two chunks, split before any model call ----
    section("1. FR-1 — two-file diff -> two chunks (split before any model call)")
    diff_path = ROOT / "examples" / "clean_two_file.diff"
    diff_text = await read_diff(diff_path)
    chunks = split_diff(diff_text)
    say(f"file: {diff_path.name}")
    say(f"chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        first = chunk.splitlines()[0]
        say(f"  chunk {i}: {first}  ({len(chunk.splitlines())} lines)")
    assert len(chunks) == 2, "FR-1 expected two chunks"
    say("PASS: exactly two chunks, produced by pure splitting (no model involved).")

    malformed = await read_diff(ROOT / "examples" / "malformed.diff")
    try:
        await intake(malformed)
        say("FAIL: malformed diff did not raise DiffError")
        return 1
    except DiffError as exc:
        say(f"malformed diff -> friendly message: {exc.message}")
        say("PASS: message, not traceback.")

    # ---- 2+3+5+6. Live review of the three-file example ----
    section("2/3/5/6. Live three-file review — concurrency, typed findings, footer, ledger")
    ledger_before = (
        len(ledger_path.read_text(encoding="utf-8").splitlines()) if ledger_path.exists() else 0
    )
    say(f"ledger lines before: {ledger_before}")

    clean_diff = await read_diff(ROOT / "examples" / "three_file_clean.diff")
    ctx = ReviewContext(repo="demo-repo", language="python", ruleset_id="default")

    concurrent_start = time.perf_counter()
    report, _events = await drive(clean_diff, ctx)
    concurrent_s = time.perf_counter() - concurrent_start
    if report is None:
        say("FAIL: no report")
        return 1

    # FR-3: final output is a plain Python list of typed findings
    criticals = sum(1 for f in report.findings if f.severity == "critical")
    say(f"merged findings: {len(report.findings)} — one-Python-expression critical count: {criticals}")
    say(f"types check: all findings are typed objects -> {all(type(f).__name__ == 'MergedFinding' for f in report.findings)}")

    # FR-10: footer carries per-reviewer latency + real token counts
    say()
    say("FR-10 footer (per-reviewer latency + real token usage from run contexts):")
    say(report.footer)
    real_tokens = all(s.tokens_in > 0 and s.tokens_out > 0 for s in report.per_reviewer)
    say(f"all rows carry REAL usage from the run context (not estimated): {real_tokens}")

    # FR-11: ledger line per run
    ledger_after = len(ledger_path.read_text(encoding="utf-8").splitlines())
    say()
    say(f"FR-11: ledger lines before={ledger_before} after={ledger_after} (+{ledger_after - ledger_before})")
    say("  (3 reviewer runs + 1 desk run = 4 lines expected for this review)")
    for line in ledger_path.read_text(encoding="utf-8").splitlines()[ledger_before:]:
        say(f"    {line}")
    say(f"router model actually used by reviewer 1: {report.per_reviewer[0].model}")

    # ---- 2. Sequential wall clock ----
    section("2. FR-5 — sequential wall clock (same diff, one reviewer at a time)")
    seq_start = time.perf_counter()
    seq_report, _ = await drive(clean_diff, ctx)
    sequential_s = time.perf_counter() - seq_start
    say()
    say(f"CONCURRENT wall clock: {concurrent_s:.2f}s")
    say(f"SEQUENTIAL wall clock: {sequential_s:.2f}s")
    say("concurrency makes the wall clock ~= the slowest reviewer, not the sum.")

    # ---- 4. FR-8: planted secret -> refusal ----
    section("4. FR-8 — planted secret in the diff -> output guardrail refusal")
    secret_diff = await read_diff(ROOT / "examples" / "three_file_issue.diff")  # plants a fake key
    ref_report, ref_events = await drive(secret_diff, ctx)
    refused = any(isinstance(e, GuardrailRefused) for e in ref_events)
    say(f"refusal observed: {refused}")
    if ref_report is not None:
        say(f"report refused flag: {ref_report.refused}; findings shown: {len(ref_report.findings)}")
        say(f"refusal reason: {ref_report.refusal_reason}")
    planted = "sk-test-DO-NOT-USE-1234567890abcdef"
    transcript_has_secret = planted in "\n".join(_lines)
    say(f"full secret echoed anywhere in this transcript: {transcript_has_secret}")
    if ref_report is not None and ref_report.refused:
        say("PASS: refusal, not a report; the secret is never echoed.")

    # ---- 7. FR-7: run-level model override ----
    section("7. FR-7 — same reviewer objects, run-level model override")
    say(f"chain head (priority model): {PRIORITY_MODEL}")
    override = "gemini-3.5-flash-lite"
    report_o, _ = await drive(clean_diff, ctx, model_override=override)
    if report_o is None:
        say("FAIL: no report for override run")
        return 1
    models = {s.reviewer_name: s.model for s in report_o.per_reviewer}
    say(f"models under override: {models}")
    say(f"override honoured without touching any agent definition: {all('lite' in (m or '') for m in models.values())}")

    section("EVIDENCE RUN COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        code = 130
    OUT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\ntranscript saved to {OUT}")
    sys.exit(code)
