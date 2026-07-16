"""Application entry point (``uv run python -m src``)."""

from __future__ import annotations
import sys
import time
from typing import Any
from .cli import parse_args
from .decoder import StepFn
from .io_utils import (
    InputError,
    load_function_definitions,
    load_test_prompts,
    write_json,
)
from .pipeline import Pipeline

_TTY = sys.stderr.isatty()


def _ansi(code: str) -> str:
    """Return an ANSI code only when stderr is a real terminal."""
    return code if _TTY else ""


_DIM = _ansi("\033[2m")
_BOLD = _ansi("\033[1m")
_CYAN = _ansi("\033[36m")
_GREEN = _ansi("\033[32m")
_YELLOW = _ansi("\033[33m")
_MAGENTA = _ansi("\033[35m")
_RED = _ansi("\033[31m")
_RESET = _ansi("\033[0m")

_PHASE_COLOR = {
    "NAME": _CYAN,
    "PARAM_KEY": _YELLOW,
    "VALUE_NUMBER": _GREEN,
    "VALUE_STRING": _GREEN,
    "VALUE_BOOLEAN": _GREEN,
    "DONE": _MAGENTA,
}


def _fmt_time(seconds: float) -> str:
    """Format a duration as '7.0s' or '2m 27s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(round(seconds - minutes * 60))
    return f"{minutes}m {rest:02d}s"


def _truncate(text: str, width: int) -> str:
    """Trim text to width characters, ending with an ellipsis if cut."""
    text = text.replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def make_printer() -> StepFn:
    """Build the per-step trace printer used in verbose mode."""

    def printer(
        step: int,
        phase: str,
        legal: int,
        vocab: int,
        chose: str,
        sofar: str,
    ) -> None:
        color = _PHASE_COLOR.get(phase, _DIM)
        bar = "\u258f" * min(legal, 20)
        print(
            f"  {step:>3}  {color}{phase:<13}{_RESET} "
            f"{_DIM}{legal:>3}/{vocab}{_RESET} "
            f"{_DIM}{bar:<20}{_RESET} "
            f"{_BOLD}{chose!r}{_RESET}",
            file=sys.stderr,
        )

    return printer


def main(argv: list[str] | None = None) -> int:
    """Run the full read -> generate -> write pipeline.

    Args:
        argv: Optional explicit argument list (used in tests).

    Returns:
        Exit code: ``0`` on success, ``1`` on a handled error.
    """
    args = parse_args(argv)

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "Loading model (first run downloads ~1.5 GB, then caches)...",
        flush=True,
    )
    load_start = time.perf_counter()
    try:
        pipeline = Pipeline(functions)
    except Exception as exc:
        print(f"error: could not initialise pipeline: {exc}", file=sys.stderr)
        return 1
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {_fmt_time(load_time)}", flush=True)

    on_step = make_printer() if args.verbose else None

    total = len(prompts)
    print(
        f"\n{_BOLD}Processing {total} prompt(s){_RESET} "
        f"{_DIM}\u2192 {args.output}{_RESET}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    succeeded = 0
    gen_start = time.perf_counter()
    for i, entry in enumerate(prompts, start=1):
        step_start = time.perf_counter()
        if args.verbose:
            line = "\u2500" * 60
            print(
                f"\n{_DIM}{line}{_RESET}\n"
                f"{_BOLD}[{i}/{total}]{_RESET} {entry.prompt}\n"
                f"{_DIM}{line}{_RESET}",
                file=sys.stderr,
            )
        counter = f"{i:>{len(str(total))}}/{total}"
        prompt_col = _truncate(entry.prompt, 38)
        try:
            call = pipeline.run(entry.prompt, on_step=on_step)
            results.append(call.model_dump())
            succeeded += 1
            elapsed = time.perf_counter() - step_start
            print(
                f"  {_DIM}{counter}{_RESET}  {prompt_col:<38}  "
                f"{_GREEN}\u2192{_RESET} {_BOLD}{call.name:<30}{_RESET}"
                f"{_DIM}{_fmt_time(elapsed):>8}{_RESET}",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - step_start
            print(
                f"  {_DIM}{counter}{_RESET}  {prompt_col:<38}  "
                f"{_RED}\u2717 failed{_RESET}"
                f"{_DIM}{_fmt_time(elapsed):>8}{_RESET}",
                flush=True,
            )
            print(
                f"warning: could not process prompt "
                f"{entry.prompt!r}: {exc}",
                file=sys.stderr,
            )
    gen_time = time.perf_counter() - gen_start

    try:
        write_json(args.output, results)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{_BOLD}Done \u2014 {succeeded}/{total} succeeded{_RESET}\n"
        f"{_DIM}Model load {_fmt_time(load_time)}  \u00b7  "
        f"Generation {_fmt_time(gen_time)}  \u00b7  "
        f"Total {_fmt_time(load_time + gen_time)}{_RESET}"
    )
    print(f"Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
