"""Tests for the character-level constraint machine."""

from __future__ import annotations
import json
import pytest
from src.constraints import FunctionCallConstraint
from src.models import FunctionDefinition, ParameterSpec
from tests.conftest import make_functions


def drive(
    functions: list[FunctionDefinition], text: str
) -> FunctionCallConstraint:
    """Feed text char-by-char, asserting each is allowed."""
    machine = FunctionCallConstraint(functions)
    for i, ch in enumerate(text):
        assert ch in machine.allowed_next(), (
            f"char {ch!r} at index {i} rejected; "
            f"allowed={sorted(machine.allowed_next())[:10]}"
        )
        machine.advance(ch)
    return machine


def rejects(functions: list[FunctionDefinition], text: str) -> bool:
    """Return True if the machine rejects the text at some character."""
    machine = FunctionCallConstraint(functions)
    for ch in text:
        if ch not in machine.allowed_next():
            return True
        machine.advance(ch)
    return False


def test_empty_functions_raises() -> None:
    with pytest.raises(ValueError):
        FunctionCallConstraint([])


def test_single_number_call_is_valid_json() -> None:
    out = drive(
        make_functions(),
        '{"name": "fn_add_numbers", "parameters": {"a": 3.0, "b": 5.0}}',
    )
    assert out.is_complete()
    assert json.loads(out._output)["name"] == "fn_add_numbers"


def test_integer_parameter() -> None:
    out = drive(
        make_functions(),
        '{"name": "fn_is_even", "parameters": {"n": 4}}',
    )
    assert out.is_complete()
    assert json.loads(out._output)["parameters"] == {"n": 4}


def test_string_parameter() -> None:
    out = drive(
        make_functions(),
        '{"name": "fn_greet", "parameters": {"name": "sam"}}',
    )
    assert out.is_complete()
    assert json.loads(out._output)["parameters"] == {"name": "sam"}


def test_number_type_requires_decimal() -> None:
    assert rejects(
        make_functions(),
        '{"name": "fn_add_numbers", "parameters": {"a": 3,',
    )


def test_integer_type_forbids_decimal() -> None:
    assert rejects(
        make_functions(),
        '{"name": "fn_is_even", "parameters": {"n": 4.',
    )


def test_illegal_letter_in_number_slot() -> None:
    assert rejects(
        make_functions(),
        '{"name": "fn_add_numbers", "parameters": {"a": x',
    )


def test_escaped_quote_inside_string() -> None:
    fns = [
        FunctionDefinition(
            name="fn_greet",
            parameters={"name": ParameterSpec(type="string")},
        )
    ]
    out = drive(fns, '{"name": "fn_greet", "parameters": {"name": "a\\"b"}}')
    assert out.is_complete()
    assert json.loads(out._output)["parameters"]["name"] == 'a"b'


def test_accepts_does_not_mutate_state() -> None:
    machine = FunctionCallConstraint(make_functions())
    for ch in '{"name": "':
        machine.advance(ch)
    before = machine._output
    machine.accepts("fn_add_numbers")
    machine.accepts("nonsense")
    assert machine._output == before


def test_accepts_rejects_empty_string() -> None:
    assert FunctionCallConstraint(make_functions()).accepts("") is False


def test_boolean_parameter() -> None:
    fns = [
        FunctionDefinition(
            name="fn_set_flag",
            parameters={"enabled": ParameterSpec(type="boolean")},
        )
    ]
    out = drive(
        fns, '{"name": "fn_set_flag", "parameters": {"enabled": true}}'
    )
    assert out.is_complete()
    assert json.loads(out._output)["parameters"] == {"enabled": True}


def test_boolean_rejects_partial_word() -> None:
    fns = [
        FunctionDefinition(
            name="fn_set_flag",
            parameters={"enabled": ParameterSpec(type="boolean")},
        )
    ]
    assert rejects(
        fns, '{"name": "fn_set_flag", "parameters": {"enabled": tru}'
    )
