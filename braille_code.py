"""English six-dot identities with the project's explicit LEGO A alias.

This is an alphabet/number/period subset, not a full UEB translator.
Alphabet and digit identities are standard; the project also reads dots 1+6 as A.
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
DIGIT_DOTS = {
    digit: LETTER_DOTS[letter]
    for digit, letter in zip("1234567890", "ABCDEFGHIJ", strict=True)
}
INDICATOR_DOTS = {"number": (3, 4, 5, 6), "capital": (6,), "grade1": (5, 6)}
PUNCTUATION_DOTS = {".": (2, 5, 6)}
INDICATOR_SYMBOLS = {"number": "#", "capital": "<CAP>", "grade1": "<GRADE1>"}
# User-selected LEGO convention, not standard English Braille or digit 1.
PROJECT_LETTER_ALIASES = {"A": (1, 6)}
DECODER_ID = "lego_english_a16_period_v3"


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
MASK_TO_PUNCTUATION = {dots_to_mask(d): c for c, d in PUNCTUATION_DOTS.items()}
MASK_TO_PROJECT_LETTER = {dots_to_mask(d): c for c, d in PROJECT_LETTER_ALIASES.items()}
NUMBER_MASK = dots_to_mask(INDICATOR_DOTS["number"])


def decode_mask(mask):
    if type(mask) is not int or not 0 <= mask < 64:
        raise ValueError("A six-dot mask must be an integer in [0,63]")
    dots = [d for d in range(1, 7) if mask & (1 << (d - 1))]
    project_letter = MASK_TO_PROJECT_LETTER.get(mask)
    letter = project_letter or MASK_TO_LETTER.get(mask)
    digit = MASK_TO_DIGIT.get(mask)
    indicator = MASK_TO_INDICATOR.get(mask)
    punctuation = MASK_TO_PUNCTUATION.get(mask)
    return {
        "mask": mask,
        "raised_dots": dots,
        "unicode": chr(0x2800 + mask),
        "letter_candidate": letter,
        "number_candidate": digit,
        "punctuation_candidate": punctuation,
        "indicator": indicator,
        "mapping_source": "project_alias" if project_letter else "standard"
        if letter or digit or indicator or punctuation or mask == 0 else None,
        "kind": "letter"
        if letter
        else "indicator"
        if indicator
        else "punctuation"
        if punctuation
        else "blank"
        if mask == 0
        else "unsupported",
        "requires_number_context": digit is not None,
    }


def decode_sequence(masks, mode="letters"):
    """Return visible tokens and text; None preserves an unread cell as '?'.

    letters: alphabet identities; numbers: explicitly selected digit identities.
    auto: a number indicator starts numeric context and a period retains it.
    Blank, capital/grade-1 indicators and unread/unsupported cells end it.
    The period is the only punctuation supported; contractions and full UEB
    capitalization are outside this prototype. Letters display uppercase.
    '?' is an unread/unsupported placeholder, never recognized punctuation.
    The custom A16 alias is a letter only; standard digit 1 still uses dot 1.
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
                value = INDICATOR_SYMBOLS["number"]
            reason = "number_indicator"
        elif cell["indicator"] in {"grade1", "capital"}:
            if mode == "auto":
                numeric, value = False, ""
            else:
                value = INDICATOR_SYMBOLS[cell["indicator"]]
            reason = "indicator"
        elif cell["kind"] == "blank":
            value, reason = " ", "explicit_blank"
            numeric = mode == "numbers"
        elif cell["punctuation_candidate"] is not None:
            value = cell["punctuation_candidate"]
            reason = "decimal_point" if numeric and value == "." else "punctuation"
        elif numeric and cell["number_candidate"] is not None:
            value, reason = cell["number_candidate"], "numeric_context"
        elif mode != "numbers" and cell["letter_candidate"] is not None:
            value = cell["letter_candidate"]
            reason = "project_alias" if cell["mapping_source"] == "project_alias" else "alphabet_identity"
            numeric = False
        else:
            # An unsupported cell may interrupt a number. Never carry inferred
            # numeric context through it; explicitly selected number mode stays.
            numeric = mode == "numbers"
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
        "complete": all(t["reason"] not in {"unread_cell", "unsupported_in_mode"} for t in tokens),
    }


def dictionary_document():
    return {
        "language": "English Braille with project-specific LEGO alias",
        "decoder_id": DECODER_ID,
        "scope": "Alphabet, numbers, period and indicators; not a full UEB translator",
        "dot_numbering": [[1, 4], [2, 5], [3, 6]],
        "bit_order": "bit 0 = dot 1, ..., bit 5 = dot 6",
        "letters": LETTER_DOTS,
        "digits_in_numeric_context": DIGIT_DOTS,
        "indicators": INDICATOR_DOTS,
        "indicator_display_symbols": INDICATOR_SYMBOLS,
        "punctuation": PUNCTUATION_DOTS,
        "project_letter_aliases": PROJECT_LETTER_ALIASES,
        "project_alias_rule": "Dots 1+6 additionally decode as A by user request; standard A (dot 1) remains supported. This alias does not change digit 1.",
        "number_rule": "A-J share their patterns with 1-9,0; a number indicator or explicitly selected numeric mode is required to disambiguate",
        "sources": [
            "https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf",
            "https://www.afb.org/diagrams-braille-cells",
        ],
    }
