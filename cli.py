"""Code Review Desk — async command-line entry (FR-1).

Reads a unified diff from a path, runs the review pipeline, and prints
progressive status lines plus the final severity-styled report with its
measurements footer. Every failure mode is one friendly sentence and a clean
exit — never a traceback (NFR-4):

- missing/unreadable/malformed diff → intake's ``DiffError.message``;
- missing ``GEMINI_API_KEY`` → one clear sentence at startup;
- missing ``OPENAI_API_KEY`` → tracing disabled (from :func:`setup_tracing`),
  the review still runs;
- anything else → one sentence, non-zero exit code.

FR-5 evidence lives here too: ``--sequential`` runs the fan-out one reviewer
at a time, and ``--concurrent-timing`` runs BOTH modes and prints the two wall
clocks side by side. FR-7 is the ``--model-override`` flag: the same reviewer
objects re-run under another model via ``RunConfig(model=...)``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator, Sequence

from dotenv import load_dotenv

from src.intake import DiffError, ReviewContext, read_diff
from src.observe import setup_tracing
from src.pipeline import (
    FindingsLanded,
    GuardrailRefused,
    PartialReview,
    RemediationOffered,
    ReviewComplete,
    ReviewReport,
    ReviewerStarted,
    ReviewStarted,
    render_report_markdown,
    run_review,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface: one flag per knob, defaults that just work."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Code Review Desk — review a unified diff with three concurrent reviewers.",
    )
    parser.add_argument("--diff", required=True, help="path to a unified diff file")
    parser.add_argument("--repo", default="local", help="repository label for the report")
    parser.add_argument("--language", default="generic", help="language under review")
    parser.add_argument("--ruleset", default="default", help="ruleset id under data/rulesets/")
    parser.add_argument(
        "--strict", action="store_true", help="review in strict mode (terser prompts)"
    )
    parser.add_argument(
        "--model-override",
        default=None,
        metavar="NAME",
        help="run-level model override via RunConfig(model=...), e.g. a lighter model from the router chain (FR-7)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="run the reviewers one by one (FR-5 wall-clock comparison demo)",
    )
    parser.add_argument(
        "--concurrent-timing",
        action="store_true",
        help="run both concurrent and sequential modes and print both wall clocks",
    )
    return parser


def _consume_event(event: object) -> None:
    """Print one progressive status line for a pipeline event (FR-12 CLI side)."""
    if isinstance(event, ReviewStarted):
        print(
            f"[{event.request_id}] reviewing {event.n_chunks} file chunk(s) "
            f"({event.context.strictness} mode)…"
        )
    elif isinstance(event, ReviewerStarted):
        print(f"  → {event.reviewer} started")
    elif isinstance(event, FindingsLanded):
        state = "partial" if event.partial else "done"
        print(
            f"  ← {event.reviewer} landed ({state}): "
            f"{len(event.findings)} finding(s) in {event.latency_ms or 0:.0f} ms"
        )
    elif isinstance(event, PartialReview):
        print(f"  ! partial review — {event.reason}")
    elif isinstance(event, GuardrailRefused):
        print("  ! secret guardrail refused the report (nothing will be echoed)")
    elif isinstance(event, RemediationOffered):
        if event.escalation is not None:
            print("  → remediation engineer consulted for a critical security finding")


async def _drive_review(
    diff_text: str, ctx: ReviewContext, *, args: argparse.Namespace
) -> ReviewReport | None:
    """Run one review, printing events as they land; return the final report."""
    report: ReviewReport | None = None
    stream: AsyncIterator[object] = run_review(
        diff_text,
        ctx,
        model_override=args.model_override,
        mode="sequential" if args.sequential else "concurrent",
    )
    async for event in stream:
        _consume_event(event)
        if isinstance(event, ReviewComplete):
            report = event.report
    return report


async def main(argv: Sequence[str] | None = None) -> int:
    """Async CLI entry. Returns the process exit code; never dumps a traceback."""
    load_dotenv()
    args = build_parser().parse_args(argv)

    # Tracing first: missing OPENAI_API_KEY only costs us traces, not the review.
    tracing_note = setup_tracing()
    if tracing_note:
        print(tracing_note)

    # The desk cannot call any model without the Gemini key — one clear sentence.
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print(
            "GEMINI_API_KEY is not set — add it to the project .env file "
            "(never commit it) and run the CLI again."
        )
        return 1

    try:
        diff_text = await read_diff(args.diff)
    except DiffError as exc:
        print(exc.message)
        return 1

    ctx = ReviewContext(
        repo=args.repo,
        language=args.language,
        ruleset_id=args.ruleset,
        strictness="strict" if args.strict else "normal",
    )

    if args.concurrent_timing:
        return await _concurrent_timing_demo(diff_text, ctx, args)

    try:
        report = await _drive_review(diff_text, ctx, args=args)
        if report is None:
            print("The review ended without a report — please try again.")
            return 1
        # The render sits inside the same guard: a report that cannot be
        # rendered is one friendly sentence, never a traceback (NFR-4).
        markdown = render_report_markdown(report)
    except DiffError as exc:
        print(exc.message)
        return 1
    except Exception as exc:  # noqa: BLE001 - one sentence, never a traceback
        print(
            f"The review could not be completed ({type(exc).__name__}: {exc}). "
            "Check your configuration and try again."
        )
        return 1

    print()
    print(markdown)
    return 0


async def _concurrent_timing_demo(
    diff_text: str, ctx: ReviewContext, args: argparse.Namespace
) -> int:
    """FR-5 evidence: both modes, two wall clocks, side by side."""
    clocks: dict[str, float] = {}
    for label, sequential in (("concurrent", False), ("sequential", True)):
        print(f"--- {label} run ---")
        mode_args = argparse.Namespace(**{**vars(args), "sequential": sequential})
        started = time.perf_counter()
        try:
            report = await _drive_review(diff_text, ctx, args=mode_args)
        except DiffError as exc:
            print(exc.message)
            return 1
        except Exception as exc:  # noqa: BLE001 - one sentence, never a traceback
            print(
                f"The {label} run could not be completed "
                f"({type(exc).__name__}: {exc}). Check your configuration and try again."
            )
            return 1
        clocks[label] = time.perf_counter() - started
        print(f"{label} wall clock: {clocks[label]:.2f}s")
        if report is not None:
            print(f"{label} merged findings: {len(report.findings)}")
    print()
    print(
        f"Concurrent: {clocks['concurrent']:.2f}s | Sequential: {clocks['sequential']:.2f}s "
        "— concurrency makes the wall clock ≈ the slowest reviewer, not the sum."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
