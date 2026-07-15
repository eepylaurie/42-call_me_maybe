"""Shared fixtures and fakes for the test suite."""

from __future__ import annotations
from collections.abc import Callable
from src.models import FunctionDefinition, ParameterSpec


def make_functions() -> list[FunctionDefinition]:
    """Return a function set covering number, integer and string types."""
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={
                "a": ParameterSpec(type="number"),
                "b": ParameterSpec(type="number"),
            },
        ),
        FunctionDefinition(
            name="fn_is_even",
            description="Check if an integer is even.",
            parameters={"n": ParameterSpec(type="integer")},
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet a person by name.",
            parameters={"name": ParameterSpec(type="string")},
        ),
    ]


# A fake single-character vocabulary: one token per character. Lets the
# decoder be tested with no LLM at all.
_CHARS = (
    list(' {}",:._-') + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz")
)
_CHARS = list(dict.fromkeys(_CHARS))


class FakeVocab:
    """Stand-in for Vocabulary: token id i maps to a single character."""

    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(_CHARS)

    def token_text(self, token_id: int) -> str:
        """Return the single-character text for a token id."""
        return _CHARS[token_id]


def fake_encode(_text: str) -> list[int]:
    """Return a dummy token id list; the decoder ignores its content."""
    return [0]


def make_logits_fn(
    preferred: str = "",
) -> Callable[[list[int]], list[float]]:
    """Return a logits function that biases the given characters high."""

    def logits_fn(_ids: list[int]) -> list[float]:
        scores = [0.0] * len(_CHARS)
        for ch in preferred:
            if ch in _CHARS:
                scores[_CHARS.index(ch)] = 10.0
        return scores

    return logits_fn


CHARS = _CHARS
