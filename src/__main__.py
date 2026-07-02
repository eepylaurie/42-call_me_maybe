"""Application entry point (``uv run python -m src``)."""

from __future__ import annotations
import sys
from typing import Any
from .cli import parse_args
from .io_utils import (
    InputError,
    load_function_definitions,
    load_test_prompts,
    write_json,
)
from .pipeline import generate_function_call


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

    results: list[dict[str, Any]] = []
    for entry in prompts:
        try:
            call = generate_function_call(entry.prompt, functions)
            results.append(call.model_dump())
        except Exception as exc:
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
