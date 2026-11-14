"""Standard English six-dot letter identities and integer-number context.

This is an uncontracted alphabet/number subset, not a full UEB translator.
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
DIGIT_DOTS = {
    digit: LETTER_DOTS[letter]
    for digit, letter in zip("1234567890", "ABCDEFGHIJ", strict=True)
}
INDICATOR_DOTS = {"number": (3, 4, 5, 6), "capital": (6,), "grade1": (5, 6)}


def dots_to_mask(dots):
    dots = tuple(dots)
    if any(type(d) is not int or not 1 <= d <= 6 for d in dots) or len(
        set(dots)
    ) != len(dots):
        raise ValueError("Raised dots must be unique integers from 1 to 6")
    return sum(1 << (d - 1) for d in dots)


MASK_TO_LETTER = {dots_to_mask(d): c for c, d in LETTER_DOTS.items()}
MASK_TO_DIGIT = {dots_to_mask(d): c for c, d in DIGIT_DOTS.items()}
MASK_TO_INDICATOR = {dots_to_mask(d): c for c, d in INDICATOR_DOTS.items()}
NUMBER_MASK = dots_to_mask(INDICATOR_DOTS["number"])


def decode_mask(mask):
    if type(mask) is not int or not 0 <= mask < 64:
        raise ValueError("A six-dot mask must be an integer in [0,63]")
    dots = [d for d in range(1, 7) if mask & (1 << (d - 1))]
    letter = MASK_TO_LETTER.get(mask)
    digit = MASK_TO_DIGIT.get(mask)
    indicator = MASK_TO_INDICATOR.get(mask)
    return {
        "mask": mask,
        "raised_dots": dots,
        "unicode": chr(0x2800 + mask),
        "letter_candidate": letter,
        "number_candidate": digit,
        "indicator": indicator,
        "kind": "letter"
        if letter
        else "indicator"
        if indicator
        else "blank"
        if mask == 0
        else "unsupported",
        "requires_number_context": digit is not None,
    }


def decode_sequence(masks, mode="letters"):
    """Return visible tokens and text; None preserves an unread cell as '?'.

    letters: alphabet identities; numbers: explicitly selected digit identities.
    auto: a number indicator begins an integer run, blank/grade-1 ends it.
    Punctuation, contractions and complete UEB capitalization rules are outside
    this prototype. Alphabet identities are displayed in uppercase in all modes.
    """
    if mode not in {"letters", "numbers", "auto"}:
        raise ValueError("mode must be letters, numbers or auto")
    numeric = mode == "numbers"
    text, tokens = [], []
    for mask in masks:
        if mask is None:
            text.append("?")
            tokens.append({"value": "?", "mode": "unknown", "reason": "unread_cell"})
            # Unknown cells could be indicators: do not preserve numeric context.
            numeric = mode == "numbers"
            continue
        cell = decode_mask(mask)
        value, reason = "?", "unsupported_in_mode"
        if cell["indicator"] == "number":
            if mode == "auto":
                numeric, value = True, ""
            else:
                value = "#"
            reason = "number_indicator"
        elif cell["indicator"] in {"grade1", "capital"}:
            if mode == "auto":
                numeric, value = False, ""
            else:
                value = "<GRADE1>" if cell["indicator"] == "grade1" else "<CAP>"
            reason = "indicator"
        elif cell["kind"] == "blank":
            value, reason = " ", "explicit_blank"
            numeric = mode == "numbers"
        elif numeric and cell["number_candidate"] is not None:
            value, reason = cell["number_candidate"], "numeric_context"
        elif mode != "numbers" and cell["letter_candidate"] is not None:
            value, reason = cell["letter_candidate"], "alphabet_identity"
            numeric = False
        tokens.append(
            {
                "value": value,
                "mode": "number" if numeric else "letter",
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
        "language": "Standard English Braille",
        "scope": "Uncontracted alphabet and integer-number subset",
        "dot_numbering": [[1, 4], [2, 5], [3, 6]],
        "bit_order": "bit 0 = dot 1, ..., bit 5 = dot 6",
        "letters": LETTER_DOTS,
        "digits_in_numeric_context": DIGIT_DOTS,
        "indicators": INDICATOR_DOTS,
        "number_rule": "A-J share their patterns with 1-9,0; a number indicator or explicitly selected numeric mode is required to disambiguate",
        "sources": [
            "https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf",
            "https://www.afb.org/diagrams-braille-cells",
        ],
    }
