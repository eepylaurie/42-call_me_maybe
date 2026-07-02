"""Vocabulary: map between token IDs and their real text."""

from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _byte_to_char() -> dict[int, str]:
    """Build GPT-2's byte -> printable-unicode mapping.

    Returns:
        A dict from each byte value (0-255) to the single character
        used to represent it in ``vocab.json``.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    byte_to_uni = {b: chr(b) for b in printable}
    next_code = 0
    for b in range(256):
        if b not in byte_to_uni:
            byte_to_uni[b] = chr(256 + next_code)
            next_code += 1
    return byte_to_uni


class Vocabulary:
    """Two-way mapping between token IDs and their real text."""

    def __init__(self, vocab_path: str | Path) -> None:
        """Load the vocabulary file and build the lookup tables.

        Args:
            vocab_path: Path to the model's ``vocab.json``.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file is not a valid token->id mapping.
        """
        with Path(vocab_path).open("r", encoding="utf-8") as handle:
            raw: dict[str, int] = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("vocab.json must be a token->id object.")

        byte_to_uni = _byte_to_char()
        self._char_to_byte = {c: b for b, c in byte_to_uni.items()}

        self._id_to_encoded: dict[int, str] = {
            tid: tok for tok, tid in raw.items()
        }

    def token_text(self, token_id: int) -> str:
        """Return the real text a token contributes.

        Args:
            token_id: The token's integer ID.

        Returns:
            The decoded string (e.g. id 279 -> " the").

        Raises:
            KeyError: If the token ID is not in the vocabulary.
        """
        encoded = self._id_to_encoded[token_id]
        data = bytes(self._char_to_byte[ch] for ch in encoded)
        return data.decode("utf-8", errors="replace")

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self._id_to_encoded)
