"""Command-line argument parsing."""

from __future__ import annotations
import argparse
from pathlib import Path

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument list. Defaults to ``sys.argv``
            when ``None``; passing a list makes this testable.

    Returns:
        The parsed arguments namespace with ``functions_definition``,
        ``input`` and ``output`` :class:`~pathlib.Path` attributes.
    """
    parser = argparse.ArgumentParser(
        prog="src",
        description="Translate prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the results JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the constrained-decoding trace to stderr.",
    )
    return parser.parse_args(argv)
