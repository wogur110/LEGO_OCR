"""Calibration validation and coordinate transforms for LEGO brick faces."""

import re

import cv2
import numpy as np

CENTERS = np.float32([[20, 20], [20, 60], [20, 100], [60, 20], [60, 60], [60, 100]])
DESTINATION = np.float32([[0, 0], [79, 0], [79, 159], [0, 159]])


def validate_layout(layout, image_size):
    if not isinstance(layout, dict):
        raise TypeError("Layout must be a JSON object")
    if layout.get("version") != 1 or layout.get("image_size") != list(image_size):
        raise ValueError("Layout version must be 1 and image_size must match the image")
    cells = layout.get("cells")
    if not isinstance(cells, list) or not 1 <= len(cells) <= 200:
        raise ValueError("Layout needs between 1 and 200 cells")
    seen = set()
    w, h = image_size
    for cell in cells:
        row, index = cell.get("row_id"), cell.get("slot_index")
        if not isinstance(row, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", row):
            raise ValueError(
                "row_id must contain letters, numbers, underscores or hyphens"
            )
        if type(index) is not int or index < 0 or (row, index) in seen:
            raise ValueError(
                "Each cell needs a unique row_id/slot_index, with a nonnegative integer slot_index"
            )
        seen.add((row, index))
        q = np.asarray(cell.get("quad"), np.float32)
        if q.shape != (4, 2) or not np.isfinite(q).all():
            raise ValueError("Each quad needs finite TL,TR,BR,BL coordinates")
        if (q < 0).any() or (q[:, 0] >= w).any() or (q[:, 1] >= h).any():
            raise ValueError("All cell corners must be inside the image")
        if not cv2.isContourConvex(q) or cv2.contourArea(q, oriented=True) < 32:
            raise ValueError(
                "Corners must form a convex TL,TR,BR,BL quad with nonzero area"
            )
    for row in {r for r, _ in seen}:
        indices = sorted(i for r, i in seen if r == row)
        if indices != list(range(len(indices))):
            raise ValueError("slot_index must be contiguous from zero in each row")


def project(points, transform):
    return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), transform)[0]
