"""Reading and writing the project's JSON files."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from .models import FunctionDefinition, TestPrompt


class InputError(Exception):
    """Raised when an input file is missing, unreadable, or invalid."""


def load_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content (list, dict, or scalar).

    Raises:
        InputError: If the file is missing or not valid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise InputError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as pretty-printed JSON, creating parent dirs.

    Args:
        path: Destination file path.
        data: JSON-serialisable content.

    Raises:
        InputError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise InputError(f"Could not write {path}: {exc}") from exc


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function-definitions file.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        A list of validated :class:`FunctionDefinition` objects.

    Raises:
        InputError: If the file is invalid or has the wrong shape.
    """
    raw = load_json(path)
    if not isinstance(raw, list):
        raise InputError(
            f"{path}: expected a JSON array of function definitions."
        )
    try:
        return [FunctionDefinition.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise InputError(
            f"{path}: invalid function definition:\n{exc}"
        ) from exc


def load_test_prompts(path: Path) -> list[TestPrompt]:
    """Load and validate the prompts file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        A list of validated :class:`TestPrompt` objects.

    Raises:
        InputError: If the file is invalid or has the wrong shape.
    """
    raw = load_json(path)
    if not isinstance(raw, list):
        raise InputError(f"{path}: expected a JSON array of prompts.")
    try:
        return [TestPrompt.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise InputError(f"{path}: invalid prompt entry:\n{exc}") from exc
