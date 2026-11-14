"""OpenCV stud-pattern reader for calibrated LEGO 2x4 top faces; no OCR.

A face is warped to 80x160 pixels. Only its upper 80x120 stud region enters
circle detection; the bottom printed-label quarter is never examined.
"""

from __future__ import annotations

import copy
import json
import re
from time import perf_counter

import cv2
import numpy as np

from braille_code import decode_mask, decode_sequence

CENTERS = np.float32([[20, 20], [20, 60], [20, 100], [60, 20], [60, 60], [60, 100]])
DESTINATION = np.float32([[0, 0], [79, 0], [79, 159], [0, 159]])
ANGLES = np.arange(48) * np.pi / 24


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


def detect_dots(crop, clahe):
    """Six ternary decisions. Geometry scores are diagnostics, not probabilities."""
    if crop.shape[:2] != (120, 80):
        raise ValueError("Stud crop must be 80 pixels wide and 120 pixels high")
    raw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(
        cv2.Laplacian(cv2.GaussianBlur(raw, (3, 3), 0.5), cv2.CV_32F).var()
    )
    if sharpness < 8 or float(raw.std()) < 3:
        return {
            "bits": [None] * 6,
            "sharpness": sharpness,
            "circles": [],
            "dot_evidence": [],
            "reason": "low_detail",
        }
    enhanced = cv2.GaussianBlur(clahe.apply(raw), (5, 5), 0.7)
    edges = cv2.Canny(enhanced, 30, 60)
    distance_to_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    evidence = [{"dot": i + 1, "strict": None, "weak": None} for i in range(6)]
    circles_output = []
    for pass_name, votes in [("strict", 12), ("weak", 10)]:
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=22,
            param1=60,
            param2=votes,
            minRadius=8,
            maxRadius=17,
        )
        if circles is None:
            continue
        for x, y, radius in circles[0]:
            distances = np.linalg.norm(CENTERS - [x, y], axis=1)
            index = int(distances.argmin())
            if distances[index] >= 18:
                continue
            xx = np.round(x + radius * np.cos(ANGLES)).astype(int)
            yy = np.round(y + radius * np.sin(ANGLES)).astype(int)
            inside = (xx >= 0) & (xx < 80) & (yy >= 0) & (yy < 120)
            if int(inside.sum()) < 24:
                continue
            coverage = float((distance_to_edge[yy[inside], xx[inside]] <= 2).mean())
            cx, cy = CENTERS[index].astype(int)
            patch = raw[cy - 15 : cy + 16, cx - 15 : cx + 16]
            contrast = float(np.percentile(patch, 95) - np.percentile(patch, 5))
            item = {
                "center": [float(x), float(y)],
                "radius": float(radius),
                "dot": index + 1,
                "edge_coverage": coverage,
                "local_contrast": contrast,
                "center_error": float(distances[index]),
                "pass": pass_name,
            }
            circles_output.append(item)
            threshold = 0.5 if pass_name == "strict" else 0.6
            if coverage >= threshold and contrast >= 15:
                old = evidence[index][pass_name]
                if old is None or coverage > old["edge_coverage"]:
                    evidence[index][pass_name] = item
    bits = [
        True if e["strict"] is not None else None if e["weak"] is not None else False
        for e in evidence
    ]
    return {
        "bits": bits,
        "sharpness": sharpness,
        "circles": circles_output,
        "dot_evidence": evidence,
        "reason": "ambiguous_dot" if any(b is None for b in bits) else "ok",
    }


class BrailleReader:
    def __init__(self, layout, mode="letters"):
        if not isinstance(layout, dict) or len(layout.get("image_size", [])) != 2:
            raise ValueError("Layout needs image_size [width,height]")
        validate_layout(layout, tuple(layout["image_size"]))
        if mode not in {"letters", "numbers", "auto"}:
            raise ValueError("mode must be letters, numbers or auto")
        self.layout, self.mode = copy.deepcopy(layout), mode
        self.clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))

    def read(self, image):
        start = perf_counter()
        h, w = image.shape[:2]
        validate_layout(self.layout, (w, h))
        prepared, crops = [], []
        for cell in self.layout["cells"]:
            q = np.float32(cell["quad"])
            transform = cv2.getPerspectiveTransform(q, DESTINATION)
            inverse = np.linalg.inv(transform)
            # Warp only the three stud rows: no pixels from the label quarter.
            crop = cv2.warpPerspective(
                image, transform, (80, 120), flags=cv2.INTER_CUBIC
            )
            source_diameter = 0.3 * min(
                np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
            )
            prepared.append((cell, q, inverse, source_diameter))
            crops.append(crop)
        warped = perf_counter()
        detections = []
        for (cell, q, inverse, diameter), crop in zip(prepared, crops, strict=True):
            observation = (
                detect_dots(crop, self.clahe)
                if diameter >= 8
                else {
                    "bits": [None] * 6,
                    "sharpness": None,
                    "circles": [],
                    "dot_evidence": [],
                    "reason": "too_small",
                }
            )
            bits = observation["bits"]
            mask = (
                None
                if any(b is None for b in bits)
                else sum(1 << i for i, b in enumerate(bits) if b)
            )
            decoded = (
                decode_mask(mask)
                if mask is not None
                else {
                    "mask": None,
                    "raised_dots": None,
                    "unicode": None,
                    "letter_candidate": None,
                    "number_candidate": None,
                    "indicator": None,
                    "kind": "uncertain",
                    "requires_number_context": False,
                }
            )
            accepted = decoded["kind"] in {"letter", "indicator"}
            reason = observation["reason"]
            if mask == 0:
                reason = "no_studs_or_blank_cell"
            elif mask is not None and not accepted:
                reason = "unsupported_pattern"
            points = project(CENTERS, inverse)
            detections.append(
                {
                    "row_id": cell["row_id"],
                    "slot_index": cell["slot_index"],
                    "face_polygon": q.tolist(),
                    "dot_centers_image": points.tolist(),
                    "estimated_source_stud_diameter_px": float(diameter),
                    **observation,
                    **decoded,
                    "accepted": accepted,
                    "reason": reason,
                }
            )
        detected = perf_counter()
        rows = []
        for row_id in dict.fromkeys(c["row_id"] for c in self.layout["cells"]):
            cells = sorted(
                (d for d in detections if d["row_id"] == row_id),
                key=lambda d: d["slot_index"],
            )
            sequence = decode_sequence(
                [d["mask"] if d["accepted"] else None for d in cells], self.mode
            )
            for item, token in zip(cells, sequence["tokens"], strict=True):
                item["value"] = token["value"]
                item["interpretation_reason"] = token["reason"]
            rows.append({"row_id": row_id, **sequence})
        payload = {
            "image_size": [w, h],
            "mode": self.mode,
            "rows": rows,
            "cells": detections,
            "coordinate_system": "Input-image pixels, origin top-left; polygons are calibrated top faces",
            "bit_order": "dots 1,2,3 down left column; 4,5,6 down right column",
        }
        json.dumps(payload, allow_nan=False)
        done = perf_counter()
        payload["timing_ms"] = {
            "validation_and_rectification": (warped - start) * 1000,
            "stud_detection_and_lookup": (detected - warped) * 1000,
            "assembly_and_serialization": (done - detected) * 1000,
            "image_to_result": (done - start) * 1000,
        }
        return payload, crops


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
