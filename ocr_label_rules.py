"""Convert printed LEGO OCR labels to A-Z or period; never output numbers."""

import re


def normalize_label(text):
    """Accept one letter with an optional digit suffix, or a period.

    User-selected OCR aliases: 0 -> O, JO -> J, and leading 1 -> I.
    Thus A1 -> A, J0 -> J and 19 -> I. Unsupported labels stay unread;
    no expected word, brick position or Braille pattern influences the result.
    """
    text = "".join(str(text).split()).upper()
    text = {"0": "O", "JO": "J0"}.get(text, text)
    if text.startswith("1"):
        text = "I" + text[1:]
    match = re.fullmatch(r"([A-Z])(?:[0-9])?|\.", text)
    return (match.group(1) or ".") if match else None
