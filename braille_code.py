"""English six-dot identities with the project's explicit LEGO A alias.

This is an alphabet/period subset, not a full UEB translator.
Alphabet identities are standard; the project also reads dots 1+6 as A.
The period is the only supported punctuation mark.
Dot order is column-major: left 1,2,3; right 4,5,6, viewed from the reading side.
Source: Unified English Braille Manual, NZ edition, lessons 1-3 and 24.
https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf
"""

from __future__ import annotations

LETTER_DOTS = {
    "A": (1,),
    "B": (1, 2),
    "C": (1, 4),
    "D": (1, 4, 5),
    "E": (1, 5),
    "F": (1, 2, 4),
    "G": (1, 2, 4, 5),
    "H": (1, 2, 5),
    "I": (2, 4),
    "J": (2, 4, 5),
    "K": (1, 3),
    "L": (1, 2, 3),
    "M": (1, 3, 4),
    "N": (1, 3, 4, 5),
    "O": (1, 3, 5),
    "P": (1, 2, 3, 4),
    "Q": (1, 2, 3, 4, 5),
    "R": (1, 2, 3, 5),
    "S": (2, 3, 4),
    "T": (2, 3, 4, 5),
    "U": (1, 3, 6),
    "V": (1, 2, 3, 6),
    "W": (2, 4, 5, 6),
    "X": (1, 3, 4, 6),
    "Y": (1, 3, 4, 5, 6),
    "Z": (1, 3, 5, 6),
}
PUNCTUATION_DOTS = {".": (2, 5, 6)}
# User-selected LEGO convention, not standard English Braille.
PROJECT_LETTER_ALIASES = {"A": (1, 6)}
OUTPUT_CHARACTERS = frozenset(LETTER_DOTS) | frozenset(PUNCTUATION_DOTS)
DECODER_ID = "lego_english_a16_period_v4"


def dots_to_mask(dots):
    dots = tuple(dots)
    if any(type(d) is not int or not 1 <= d <= 6 for d in dots) or len(
        set(dots)
    ) != len(dots):
        raise ValueError("Raised dots must be unique integers from 1 to 6")
    return sum(1 << (d - 1) for d in dots)


MASK_TO_LETTER = {dots_to_mask(d): c for c, d in LETTER_DOTS.items()}
MASK_TO_PUNCTUATION = {dots_to_mask(d): c for c, d in PUNCTUATION_DOTS.items()}
MASK_TO_PROJECT_LETTER = {dots_to_mask(d): c for c, d in PROJECT_LETTER_ALIASES.items()}


def decode_mask(mask):
    if type(mask) is not int or not 0 <= mask < 64:
        raise ValueError("A six-dot mask must be an integer in [0,63]")
    dots = [d for d in range(1, 7) if mask & (1 << (d - 1))]
    project_letter = MASK_TO_PROJECT_LETTER.get(mask)
    letter = project_letter or MASK_TO_LETTER.get(mask)
    punctuation = MASK_TO_PUNCTUATION.get(mask)
    return {
        "mask": mask,
        "raised_dots": dots,
        "unicode": chr(0x2800 + mask),
        "letter_candidate": letter,
        "punctuation_candidate": punctuation,
        "mapping_source": "project_alias" if project_letter else "standard"
        if letter or punctuation or mask == 0 else None,
        "kind": "letter"
        if letter
        else "punctuation"
        if punctuation
        else "blank"
        if mask == 0
        else "unsupported",
    }


def decode_sequence(masks, mode="letters"):
    """Return visible tokens and text; None preserves an unread cell as '?'.

    Only uppercase alphabet identities and periods are recognized. The mode
    argument retains compatibility for callers explicitly selecting "letters".
    Blank, unread and unsupported cells are visible unknowns; no cell is skipped.
    '?' is an unread/unsupported placeholder, never recognized punctuation.
    """
    if mode != "letters":
        raise ValueError("Only mode='letters' is supported (A-Z and period)")
    text, tokens = [], []
    for mask in masks:
        if mask is None:
            text.append("?")
            tokens.append({"value": "?", "mode": "unknown", "reason": "unread_cell"})
            continue
        cell = decode_mask(mask)
        value, reason = "?", "unsupported_pattern"
        if cell["kind"] == "blank":
            reason = "blank_cell"
        elif cell["punctuation_candidate"] is not None:
            value = cell["punctuation_candidate"]
            reason = "punctuation"
        elif cell["letter_candidate"] is not None:
            value = cell["letter_candidate"]
            reason = "project_alias" if cell["mapping_source"] == "project_alias" else "alphabet_identity"
        tokens.append(
            {
                "value": value,
                "mode": "unknown" if value == "?" else "letter",
                "reason": reason,
            }
        )
        text.append(value)
    return {
        "text": "".join(text),
        "tokens": tokens,
        "complete": all(t["value"] != "?" for t in tokens),
    }


def dictionary_document():
    return {
        "language": "English Braille with project-specific LEGO alias",
        "decoder_id": DECODER_ID,
        "scope": "Uppercase A-Z and period only; not a full UEB translator",
        "dot_numbering": [[1, 4], [2, 5], [3, 6]],
        "bit_order": "bit 0 = dot 1, ..., bit 5 = dot 6",
        "letters": LETTER_DOTS,
        "punctuation": PUNCTUATION_DOTS,
        "project_letter_aliases": PROJECT_LETTER_ALIASES,
        "project_alias_rule": "Dots 1+6 additionally decode as A by user request; standard A (dot 1) remains supported.",
        "unsupported_rule": "Blank, unread and unsupported cells, including Braille control indicators, produce '?' with diagnostics; digits are never decoded.",
        "sources": [
            "https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf",
            "https://www.afb.org/diagrams-braille-cells",
        ],
    }
