"""Render recognition diagnostics without changing recognition results."""

import cv2
import numpy as np

from braille_geometry import CENTERS


def previews(image, result, crops):
    annotated, cards = image.copy(), []
    for cell, crop in zip(result["cells"], crops, strict=True):
        poly = np.int32(cell["face_polygon"])
        cv2.polylines(annotated, [poly], True, (255, 80, 0), 2)
        for i, (point, bit) in enumerate(
            zip(cell["dot_centers_image"], cell["bits"], strict=True)
        ):
            color = (
                (30, 180, 30)
                if bit is True
                else (0, 130, 255)
                if bit is None
                else (160, 160, 160)
            )
            cv2.circle(annotated, tuple(np.int32(point)), 7, color, 2)
        x, y = poly[3]
        cv2.putText(
            annotated,
            cell["value"],
            (int(x), int(y) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        card = np.full((220, 170, 3), 250, np.uint8)
        card[10:130, 45:125] = crop
        for i, (point, bit) in enumerate(zip(CENTERS, cell["bits"], strict=True)):
            color = (
                (30, 160, 30)
                if bit is True
                else (0, 130, 255)
                if bit is None
                else (160, 160, 160)
            )
            center = tuple(np.int32(point + [45, 10]))
            cv2.circle(card, center, 14, color, 1)
            cv2.putText(
                card,
                str(i + 1),
                (center[0] - 3, center[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )
        cv2.putText(
            card,
            f"{cell['row_id']}/{cell['slot_index'] + 1}",
            (5, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 0),
            1,
        )
        dot_text = "".join(str(d) for d in (cell["raised_dots"] or [])) or "?"
        cv2.putText(
            card,
            f"dots {dot_text} => {cell['value']}",
            (5, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 0),
            1,
        )
        candidate = cell["number_candidate"]
        cv2.putText(
            card,
            f"number: {candidate if candidate is not None else '-'}",
            (5, 197),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 0),
            1,
        )
        cards.append(card)
    blank = np.full_like(cards[0], 250)
    sheet = np.vstack(
        [
            np.hstack((cards[i : i + 7] + [blank] * 7)[:7])
            for i in range(0, len(cards), 7)
        ]
    )
    return annotated, sheet
