"""Application entry point (``uv run python -m src``)."""

from __future__ import annotations
import sys
import time
from typing import Any
from .cli import parse_args
from .io_utils import (
    InputError,
    load_function_definitions,
    load_test_prompts,
    write_json,
)
from .pipeline import Pipeline


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
    print(
        f"Model loaded in {time.perf_counter() - load_start:.1f}s", flush=True
    )

    total = len(prompts)
    results: list[dict[str, Any]] = []
    for i, entry in enumerate(prompts, start=1):
        step_start = time.perf_counter()
        print(f"[{i}/{total}] {entry.prompt!r} ... ", end="", flush=True)
        try:
            call = pipeline.run(entry.prompt)
            results.append(call.model_dump())
            elapsed = time.perf_counter() - step_start
            print(f"-> {call.name} ({elapsed:.1f}s)", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - step_start
            print(f"failed ({elapsed:.1f}s)", flush=True)
            print(
                f"warning: could not process prompt "
                f"{entry.prompt!r}: {exc}",
                file=sys.stderr,
            )

    try:
        write_json(args.output, results)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
