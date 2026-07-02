"""Generation pipeline."""

from __future__ import annotations
from typing import Any
from .models import FunctionCall, FunctionDefinition


def _placeholder_value(type_name: str) -> Any:
    """Return a type-appropriate placeholder for a parameter.

    Args:
        type_name: JSON type name from the schema.

    Returns:
        ``0`` for numbers, ``False`` for booleans, ``""`` otherwise.
    """
    if type_name == "number":
        return 0
    if type_name == "boolean":
        return False
    return ""


def generate_function_call(
    prompt: str, functions: list[FunctionDefinition]
) -> FunctionCall:
    """Produce a structured function call for a single prompt.

    Args:
        prompt: The natural-language request.
        functions: The available function definitions.

    Returns:
        A :class:`FunctionCall` for ``prompt``.

    Raises:
        ValueError: If no function definitions are available.
    """
    if not functions:
        raise ValueError("No function definitions were provided.")
    chosen = functions[0]
    params = {
        name: _placeholder_value(spec.type)
        for name, spec in chosen.parameters.items()
    }
    return FunctionCall(prompt=prompt, name=chosen.name, parameters=params)
