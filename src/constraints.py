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

    def allowed_next(self) -> set[str]:
        """Return the set of characters legal as the next character.

        Returns:
            A set of single-character strings. Empty only when the
            machine is complete (nothing more may be emitted).
        """
        return set()

    def advance(self, char: str) -> None:
        """Commit one accepted character and update the state.

        Args:
            char: A single character previously reported by
                ``allowed_next``.

        Raises:
            ValueError: If ``char`` is not currently allowed.
        """
        self._output += char

    def is_complete(self) -> bool:
        """Return whether a full, valid function call has been emitted.

        Returns:
            ``True`` once the machine has reached :attr:`Phase.DONE`.
        """
        return self._phase is Phase.DONE
