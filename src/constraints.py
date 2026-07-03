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
SEPARATOR_TEXT = ", "
SUFFIX_TEXT = "}}"

_LITERALS = {
    Phase.PREFIX: PREFIX_TEXT,
    Phase.AFTER_NAME: AFTER_NAME_TEXT,
    Phase.SEPARATOR: SEPARATOR_TEXT,
    Phase.SUFFIX: SUFFIX_TEXT,
}

DIGITS = set("0123456789")
PRINTABLE = frozenset(chr(c) for c in range(0x20, 0x7F))
_Snapshot = tuple[
    Phase,
    str,
    "FunctionDefinition | None",
    int,
    bool,
    int,
    str,
    str,
    str,
    bool,
]


class FunctionCallConstraint:
    """Tracks which characters keep the output on a valid path."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialise the machine for one prompt.

        Args:
            functions: The available function definitions. Each name is
                a candidate; the chosen one's parameters drive the
                value phases.

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
        self._key_text = ""
        self._number_so_far = ""
        self._string_opened = False

    def _enter(self, phase: Phase) -> None:
        """Move to a new phase and reset the per-literal cursor."""
        self._phase = phase
        self._literal_pos = 0

    def _params(self) -> list[tuple[str, str]]:
        """Return the chosen function's (key, type) pairs in order."""
        if self._chosen is None:
            raise ValueError("no function chosen yet")
        return [(k, s.type) for k, s in self._chosen.parameters.items()]

    def _more_params(self) -> bool:
        """Whether parameters remain after the current one."""
        return self._param_index < len(self._params()) - 1

    def _number_terminator(self) -> str:
        """The char that ends the current number value."""
        return "," if self._more_params() else "}"

    def _number_complete(self) -> bool:
        """Whether the number typed so far is a valid JSON number."""
        s = self._number_so_far
        body = s[1:] if s.startswith("-") else s
        if body == "":
            return False
        if "." in body:
            intpart, _, frac = body.partition(".")
            return intpart.isdigit() and frac.isdigit()
        return body.isdigit()

    def _name_next_chars(self) -> set[str]:
        """Return chars continuing at least one allowed name."""
        pos = len(self._name_so_far)
        chars: set[str] = set()
        for name in self._names:
            if name.startswith(self._name_so_far) and len(name) > pos:
                chars.add(name[pos])
        return chars

    def _number_next_chars(self) -> set[str]:
        """Return chars that continue or end the current number."""
        s = self._number_so_far
        has_digit = any(c in DIGITS for c in s)
        chars: set[str] = set(DIGITS)
        if s == "":
            chars.add("-")
        if "." not in s and has_digit:
            chars.add(".")
        if self._number_complete():
            chars.add(self._number_terminator())
        return chars

    def _string_next_chars(self) -> set[str]:
        """Return chars legal at the current point of a string value."""
        if not self._string_opened:
            return {'"'}
        if self._prev_was_backslash:
            return set('"\\/bfnrt')
        return set(PRINTABLE)

    def _function_by_name(self, name: str) -> FunctionDefinition:
        """Return the definition matching an exact function name."""
        for fn in self._functions:
            if fn.name == name:
                return fn
        raise ValueError(f"unknown function name: {name}")

    def _snapshot(self) -> _Snapshot:
        """Capture all mutable state so it can be restored."""
        return (
            self._phase,
            self._output,
            self._chosen,
            self._param_index,
            self._prev_was_backslash,
            self._literal_pos,
            self._name_so_far,
            self._key_text,
            self._number_so_far,
            self._string_opened,
        )

    def _restore(self, snap: _Snapshot) -> None:
        """Restore mutable state from a snapshot."""
        (
            self._phase,
            self._output,
            self._chosen,
            self._param_index,
            self._prev_was_backslash,
            self._literal_pos,
            self._name_so_far,
            self._key_text,
            self._number_so_far,
            self._string_opened,
        ) = snap

    def accepts(self, text: str) -> bool:
        """Whether ``text`` could be emitted now, leaving state intact.

        Simulates advancing through every character of ``text`` from the
        current state, then restores the state. Usef by the decoder to
        test whole tokens without committing to them.

        Args:
            text: The candidate token's decoded string.

        Returns:
            ``True`` if every character is legal in sequence.
        """
        if text == "":
            return False
        snap = self._snapshot()
        try:
            for ch in text:
                if ch not in self.allowed_next():
                    return False
                self.advance(ch)
            return True
        finally:
            self._restore(snap)

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
        if self._phase is Phase.PARAM_KEY:
            return {self._key_text[self._literal_pos]}
        if self._phase is Phase.VALUE_NUMBER:
            return self._number_next_chars()
        if self._phase is Phase.VALUE_STRING:
            return self._string_next_chars()
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
        if self._phase in _LITERALS:
            self._output += char
            self._advance_literal()
        elif self._phase is Phase.NAME:
            self._output += char
            self._advance_name(char)
        elif self._phase is Phase.PARAM_KEY:
            self._output += char
            self._advance_param_key()
        elif self._phase is Phase.VALUE_NUMBER:
            self._advance_number(char)
        elif self._phase is Phase.VALUE_STRING:
            self._advance_string(char)

    def is_complete(self) -> bool:
        """Return whether a full, valid function call has been emitted.

        Returns:
            ``True`` once the machine has reached :attr:`Phase.DONE`.
        """
        return self._phase is Phase.DONE

    def _advance_literal(self) -> None:
        """Move the cursor within a fixed literal and transition."""
        self._literal_pos += 1
        if self._literal_pos >= len(_LITERALS[self._phase]):
            self._on_literal_complete()

    def _advance_name(self, char: str) -> None:
        """Extend the chosen name and finish it once it matches.

        Assumes no allowed name is a prefix of another (true for the
        provided set). If names nested, we would instead wait for the
        closing quote to disambiguate.
        """
        self._name_so_far += char
        if self._name_so_far in self._names:
            self._chosen = self._function_by_name(self._name_so_far)
            self._enter(Phase.AFTER_NAME)

    def _advance_param_key(self) -> None:
        """Walk the dynamic '"key": ' literal, then enter the value."""
        self._literal_pos += 1
        if self._literal_pos >= len(self._key_text):
            ptype = self._params()[self._param_index][1]
            if ptype == "number":
                self._enter(Phase.VALUE_NUMBER)
                self._number_so_far = ""
            else:
                self._enter(Phase.VALUE_STRING)
                self._string_opened = False
                self._prev_was_backslash = False

    def _advance_number(self, char: str) -> None:
        """Build the number, or finish it on the terminator.

        A number has no closing delimiter, so ``allowed_next`` offers
        the terminator (',' or '}') once the number is valid. When that
        terminator is chosen we clear the number and re-dispatch the
        character into the following literal, which consumes it.
        """
        if char in DIGITS or char == "-" or char == ".":
            self._number_so_far += char
            self._output += char
            return
        self._number_so_far = ""
        if self._more_params():
            self._enter(Phase.SEPARATOR)
        else:
            self._enter(Phase.SUFFIX)
        self.advance(char)

    def _advance_string(self, char: str) -> None:
        """Walk a JSON string value, tracking escapes.

        A closing quote ends the string unless it follows an unescaped
        backslash. The string owns its closing quote, so on close we go
        straight to the following separator or suffix.
        """
        self._output += char
        if not self._string_opened:
            self._string_opened = True
            return
        if self._prev_was_backslash:
            self._prev_was_backslash = False
            return
        if char == "\\":
            self._prev_was_backslash = True
            return
        if char == '"':
            self._string_opened = False
            if self._more_params():
                self._enter(Phase.SEPARATOR)
            else:
                self._enter(Phase.SUFFIX)

    def _on_literal_complete(self) -> None:
        """Transition out of a fixed literal once fully emitted."""
        if self._phase is Phase.PREFIX:
            self._enter(Phase.NAME)
        elif self._phase is Phase.AFTER_NAME:
            self._start_parameters()
        elif self._phase is Phase.SEPARATOR:
            self._param_index += 1
            self._enter_param_key()
        elif self._phase is Phase.SUFFIX:
            self._enter(Phase.DONE)

    def _start_parameters(self) -> None:
        """Enter the first parameter, or skip to the end if none."""
        if not self._params():
            self._enter(Phase.SUFFIX)
        else:
            self._param_index = 0
            self._enter_param_key()

    def _enter_param_key(self) -> None:
        """Set up the '"key": ' literal for the current parameter."""
        key = self._params()[self._param_index][0]
        self._key_text = f'"{key}": '
        self._phase = Phase.PARAM_KEY
        self._literal_pos = 0
