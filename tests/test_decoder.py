"""Tests for the constrained decoder, using a fake model."""

from __future__ import annotations
import json
from typing import cast
import pytest
from src.decoder import ConstrainedDecoder, DecodingError
from src.vocabulary import Vocabulary
from tests.conftest import (
    FakeVocab,
    fake_encode,
    make_functions,
    make_logits_fn,
)


def build(preferred: str = "", max_steps: int = 256) -> ConstrainedDecoder:
    """Build a decoder wired to the fake vocabulary and logits."""
    return ConstrainedDecoder(
        make_functions(),
        cast(Vocabulary, FakeVocab()),
        make_logits_fn(preferred),
        fake_encode,
        max_steps=max_steps,
    )


def test_output_is_always_valid_json_uniform_logits() -> None:
    parsed = json.loads(build().decode("anything"))
    assert set(parsed.keys()) == {"name", "parameters"}


def test_logits_steer_function_choice() -> None:
    assert json.loads(build('g"').decode("x"))["name"] == "fn_greet"


def test_decode_stops_at_completion_not_eos() -> None:
    assert build('g"').decode("x").rstrip().endswith("}}")


def test_max_steps_guard_raises() -> None:
    with pytest.raises(DecodingError):
        build("g", max_steps=40).decode("x")
