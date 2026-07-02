"""Constraint state machine for schema-valid function-call JSON."""

from __future__ import annotations
from enum import Enum, auto
from .models import FunctionDefinition


class Phase(Enum):
    """The states the machine can be in, in roughly emitted order."""

    PREFIX = auto()
    NAME = auto()
    AFTER_NAME = auto()
    PARAM_KEY = auto()
    VALUE_NUMBER = auto()
    VALUE_STRING = auto()
    SEPARATOR = auto()
    SUFFIX = auto()
    DONE = auto()


PREFIX_TEXT = '{"name": "'
AFTER_NAME_TEXT = '", "parameters": {'
SUFFIX_TEXT = "}}"

_LITERALS = {
    Phase.PREFIX: PREFIX_TEXT,
    Phase.AFTER_NAME: AFTER_NAME_TEXT,
    Phase.SUFFIX: SUFFIX_TEXT,
}


class FunctionCallConstraint:
    """Tracks which characters keep the output on a valid path."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialise the machine for one prompt.

        Args:
            functions: The available function definitions. The name of
                each is a candidate; the chosen one's parameters drive
                the value phases.

        Raises:
            ValueError: If no functions are available.
        """
        if not functions:
            raise ValueError("No function definitions were provided.")
        self._functions = functions
        self._names = [fn.name for fn in functions]

        self._phase = Phase.PREFIX
        self._output = ""
        self._chosen: FunctionDefinition | None = None
        self._param_index = 0
        self._prev_was_backslash = False
        self._literal_pos = 0
        self._name_so_far = ""

    def _enter(self, phase: Phase) -> None:
        """Move to a new phase and reset the per-literal cursor."""
        self._phase = phase
        self._literal_pos = 0

    def _name_next_chars(self) -> set[str]:
        """Return chars continuing at least one allowed name."""
        pos = len(self._name_so_far)
        chars: set[str] = set()
        for name in self._names:
            if name.startswith(self._name_so_far) and len(name) > pos:
                chars.add(name[pos])
        return chars

    def _function_by_name(self, name: str) -> FunctionDefinition:
        """Return the definition matching an exact function name."""
        for fn in self._functions:
            if fn.name == name:
                return fn
        raise ValueError(f"unknown function name: {name}")

    def allowed_next(self) -> set[str]:
        """Return the set of characters legal as the next character.

        Returns:
            A set of single-character strings. Empty only when the
            machine is complete (nothing more may be emitted).
        """
        if self._phase in _LITERALS:
            literal = _LITERALS[self._phase]
            return {literal[self._literal_pos]}
        if self._phase is Phase.NAME:
            return self._name_next_chars()
        return set()

    def advance(self, char: str) -> None:
        """Commit one accepted character and update the state.

        Args:
            char: A single character previously reported by
                ``allowed_next``.

        Raises:
            ValueError: If ``char`` is not currently allowed.
        """
        if char not in self.allowed_next():
            raise ValueError(
                f"character {char!r} not allowed in phase "
                f"{self._phase.name}"
            )
        self._output += char
        if self._phase in _LITERALS:
            self._advance_literal()
        elif self._phase is Phase.NAME:
            self._advance_name(char)

    def _advance_literal(self) -> None:
        """Move the cursor within a fixed literal and transition."""
        self._literal_pos += 1
        if self._literal_pos >= len(_LITERALS[self._phase]):
            self._on_literal_complete()

    def _advance_name(self, char: str) -> None:
        """Extend the chosen name and finish it once it matches.

        Assumes no allowed name is a prefix of another (true for the
        provided set). If names nested, we would instead wait for the
        closing quote to disambiguate the shorter from the longer.
        """
        self._name_so_far += char
        if self._name_so_far in self._names:
            self._chosen = self._function_by_name(self._name_so_far)
            self._enter(Phase.AFTER_NAME)

    def _on_literal_complete(self) -> None:
        """Transition out of a fixed literal once fully emitted."""
        if self._phase is Phase.PREFIX:
            self._enter(Phase.NAME)
        elif self._phase is Phase.AFTER_NAME:
            self._enter(Phase.PARAM_KEY)
        elif self._phase is Phase.SUFFIX:
            self._enter(Phase.DONE)

    def is_complete(self) -> bool:
        """Return whether a full, valid function call has been emitted.

        Returns:
            ``True`` once the machine has reached :attr:`Phase.DONE`.
        """
        return self._phase is Phase.DONE
