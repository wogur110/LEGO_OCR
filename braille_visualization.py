"""Render recognition diagnostics without changing recognition results."""

import cv2
import numpy as np

from braille_geometry import CENTERS


def previews(image, result, crops):
    if result.get("method") == "ocr":
        return ocr_previews(image, result, crops)
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
        status = "accepted" if cell["accepted"] else "unread/unsupported"
        cv2.putText(
            card,
            status,
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


def ocr_previews(image, result, crops):
    """Show printed-label crops and raw OCR in the input image's coordinates."""
    annotated, cards = image.copy(), []
    for cell, crop in zip(result["cells"], crops, strict=True):
        face_polygon = np.int32(cell["face_polygon"])
        label_polygon = np.int32(cell["label_polygon"])
        color = (30, 160, 30) if cell["accepted"] else (0, 130, 255)
        cv2.polylines(annotated, [face_polygon], True, (255, 80, 0), 2)
        cv2.polylines(annotated, [label_polygon], True, color, 2)
        x, y = label_polygon.min(axis=0)
        cv2.putText(
            annotated, cell["value"], (int(x), max(18, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        card = np.full((220, 220, 3), 250, np.uint8)
        enlarged = cv2.resize(crop, (152, 98), interpolation=cv2.INTER_CUBIC)
        card[10:108, 34:186] = enlarged
        raw_text = cell.get("raw_ocr_text") or "<empty>"
        raw_score = cell.get("raw_ocr_score")
        score_text = "n/a" if raw_score is None else f"{raw_score:.3f}"
        lines = [
            f"{cell['row_id']}/{cell['slot_index'] + 1} => {cell['value']}",
            f"raw: {raw_text}",
            f"OCR score: {score_text}",
            "period shape" if cell.get("source") == "printed_period_shape"
            else "accepted" if cell["accepted"] else "unread/unsupported",
        ]
        for line, y_position in zip(lines, (130, 153, 176, 199), strict=True):
            font_scale = min(0.45, 200 / max(1, cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1
            )[0][0]))
            cv2.putText(
                card, line, (8, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 1,
            )
        cards.append(card)
    if not cards:
        return annotated, np.full((220, 220, 3), 250, np.uint8)
    blank = np.full_like(cards[0], 250)
    sheet = np.vstack([
        np.hstack((cards[index:index + 6] + [blank] * 6)[:6])
        for index in range(0, len(cards), 6)
    ])
    return annotated, sheet
