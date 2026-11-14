"""Read calibrated LEGO faces through rectification, stud detection and decoding.

The nominal upper three stud rows are used; calibration must keep printed marks
outside that crop. Geometry and preview exports remain available for callers.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from braille_code import DECODER_ID, decode_mask, decode_sequence
from braille_geometry import CENTERS, DESTINATION, project, validate_layout
from braille_visualization import previews as previews
from stud_detector import ALGORITHM_DESCRIPTION, ALGORITHM_ID
from stud_detector import detect_dots as detect_dots


@dataclass(frozen=True)
class _PreparedFace:
    calibration: dict
    polygon: np.ndarray
    inverse_transform: np.ndarray
    source_stud_diameter: float


def _decode_bits(bits):
    if any(bit is None for bit in bits):
        return {
            "mask": None,
            "raised_dots": None,
            "unicode": None,
            "letter_candidate": None,
            "number_candidate": None,
            "punctuation_candidate": None,
            "indicator": None,
            "mapping_source": None,
            "kind": "uncertain",
            "requires_number_context": False,
        }
    mask = sum(1 << i for i, bit in enumerate(bits) if bit)
    return decode_mask(mask)


class BrailleReader:
    def __init__(self, layout, mode="letters"):
        if not isinstance(layout, dict) or len(layout.get("image_size", [])) != 2:
            raise ValueError("Layout needs image_size [width,height]")
        validate_layout(layout, tuple(layout["image_size"]))
        if mode not in {"letters", "numbers", "auto"}:
            raise ValueError("mode must be letters, numbers or auto")
        self.layout, self.mode = copy.deepcopy(layout), mode
        self.clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))

    def _rectify_faces(self, image):
        prepared, crops = [], []
        for cell in self.layout["cells"]:
            polygon = np.float32(cell["quad"])
            transform = cv2.getPerspectiveTransform(polygon, DESTINATION)
            inverse = np.linalg.inv(transform)
            # The bottom label quarter is outside the nominal stud crop.
            crop = cv2.warpPerspective(
                image, transform, (80, 120), flags=cv2.INTER_CUBIC
            )
            diameter = 0.3 * min(
                np.linalg.norm(polygon[1] - polygon[0]),
                np.linalg.norm(polygon[2] - polygon[3]),
            )
            prepared.append(_PreparedFace(cell, polygon, inverse, diameter))
            crops.append(crop)
        return prepared, crops

    def _read_cell(self, face, crop):
        observation = (
            detect_dots(crop, self.clahe)
            if face.source_stud_diameter >= 8
            else {
                "bits": [None] * 6,
                "sharpness": None,
                "circles": [],
                "dot_evidence": [],
                "reason": "too_small",
            }
        )
        decoded = _decode_bits(observation["bits"])
        accepted = decoded["kind"] in {"letter", "indicator", "punctuation"}
        reason = observation["reason"]
        if decoded["mask"] == 0:
            reason = "no_studs_or_blank_cell"
        elif decoded["mask"] is not None and not accepted:
            reason = "unsupported_pattern"
        return {
            "row_id": face.calibration["row_id"],
            "slot_index": face.calibration["slot_index"],
            "face_polygon": face.polygon.tolist(),
            "dot_centers_image": project(CENTERS, face.inverse_transform).tolist(),
            "estimated_source_stud_diameter_px": float(face.source_stud_diameter),
            **observation,
            **decoded,
            "accepted": accepted,
            "reason": reason,
        }

    def _assemble_rows(self, detections):
        rows = []
        for row_id in dict.fromkeys(c["row_id"] for c in self.layout["cells"]):
            cells = sorted(
                (cell for cell in detections if cell["row_id"] == row_id),
                key=lambda cell: cell["slot_index"],
            )
            sequence = decode_sequence(
                [cell["mask"] if cell["accepted"] else None for cell in cells],
                self.mode,
            )
            for item, token in zip(cells, sequence["tokens"], strict=True):
                item["value"] = token["value"]
                item["interpretation_reason"] = token["reason"]
            rows.append({"row_id": row_id, **sequence})
        return rows

    def read(self, image):
        start = perf_counter()
        height, width = image.shape[:2]
        validate_layout(self.layout, (width, height))
        prepared, crops = self._rectify_faces(image)
        warped = perf_counter()
        detections = [
            self._read_cell(face, crop)
            for face, crop in zip(prepared, crops, strict=True)
        ]
        detected = perf_counter()
        payload = {
            "algorithm_id": ALGORITHM_ID,
            "algorithm": ALGORITHM_DESCRIPTION,
            "decoder_id": DECODER_ID,
            "image_size": [width, height],
            "mode": self.mode,
            "rows": self._assemble_rows(detections),
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
