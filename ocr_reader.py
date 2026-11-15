"""Read printed LEGO labels from calibrated full-face regions using local OCR.

One model is initialized per reader and reused by read(). No automatic text
detector, six-dot decoding, reference labels or expected words are used.
"""

from __future__ import annotations

import copy
import json
from time import perf_counter

import cv2
import numpy as np

from braille_geometry import DESTINATION, project, validate_layout
from ocr_label_rules import normalize_label
from period_detector import detect_period

ALGORITHM_ID = "calibrated_ppocrv4_v1"
DECODER_ID = "lego_ocr_letters_period_v1"
LABEL_CORNERS = np.float32([[2, 110], [77, 110], [77, 158], [2, 158]])
LABEL_SIZE = (76, 49)
LABEL_DESTINATION = np.float32([[0, 0], [75, 0], [75, 48], [0, 48]])
MIN_OCR_SCORE = 0.5


class CalibratedOCRReader:
    def __init__(self, layout, *, threads=4, engine=None):
        if not isinstance(layout, dict) or len(layout.get("image_size", [])) != 2:
            raise ValueError("Layout needs image_size [width,height]")
        validate_layout(layout, tuple(layout["image_size"]))
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.layout = copy.deepcopy(layout)
        self.dictionary = {
            "decoder_id": DECODER_ID,
            "output_characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ.",
            "aliases": {"0": "O", "JO": "J", "leading_1": "I"},
            "suffix_rule": "One trailing digit is discarded after the letter",
            "min_ocr_score": MIN_OCR_SCORE,
        }
        self.regions = []
        for cell in self.layout["cells"]:
            inverse = cv2.getPerspectiveTransform(DESTINATION, np.float32(cell["quad"]))
            polygon = project(LABEL_CORNERS, inverse)
            transform = cv2.getPerspectiveTransform(polygon, LABEL_DESTINATION)
            self.regions.append((polygon, transform))
        if engine is None:
            # Optional dependency: importing or running the six-dot path never
            # imports RapidOCR or loads its models.
            from rapidocr_onnxruntime import RapidOCR

            engine = RapidOCR(
                intra_op_num_threads=threads,
                inter_op_num_threads=1,
                use_cls=False,
                rec_batch_num=6,
                rec_img_shape=[3, 48, 96],
            )
        self.engine = engine

    @staticmethod
    def _interpret(text, score, crop):
        score = float(score)
        score = score if np.isfinite(score) else None
        candidate = normalize_label(text)
        needs_period_check = (
            score is None or score < MIN_OCR_SCORE or candidate in (None, ".")
        )
        period = detect_period(crop) if needs_period_check else {
            "accepted": False, "reason": "valid_letter_ocr"
        }
        if period["accepted"]:
            value, accepted = ".", True
            source, reason, accepted_score = (
                "printed_period_shape", "isolated_filled_printed_dot", None
            )
        else:
            accepted = (
                score is not None and score >= MIN_OCR_SCORE and candidate is not None
            )
            value = candidate if accepted else "?"
            source, accepted_score = "ocr", score
            reason = (
                "recognized" if accepted
                else "low_ocr_score" if score is None or score < MIN_OCR_SCORE
                else "unsupported_label"
            )
        return {
            "value": value,
            "accepted": accepted,
            "reason": reason,
            "letter_candidate": value if accepted and value != "." else None,
            "punctuation_candidate": "." if accepted and value == "." else None,
            "raw_ocr_text": str(text),
            "raw_ocr_score": score,
            "score": accepted_score,
            "source": source,
            "period_diagnostics": period,
        }

    def read(self, image):
        """Return (result, label_crops); all positions use encoded-image pixels."""
        start = perf_counter()
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
        ):
            raise ValueError("Expected a decoded uint8 BGR image")
        height, width = image.shape[:2]
        validate_layout(self.layout, (width, height))
        crops = [
            cv2.warpPerspective(image, matrix, LABEL_SIZE, flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
            for _, matrix in self.regions
        ]
        warped = perf_counter()
        predictions, _ = self.engine.text_rec(crops)
        if predictions is None or len(predictions) != len(crops):
            raise ValueError("OCR must return exactly one prediction per calibrated label")
        recognized = perf_counter()
        cells = []
        for cell, (polygon, _), crop, (text, score) in zip(
            self.layout["cells"], self.regions, crops, predictions, strict=True
        ):
            cells.append({
                "row_id": cell["row_id"],
                "slot_index": cell["slot_index"],
                "face_polygon": np.float32(cell["quad"]).tolist(),
                "label_polygon": polygon.tolist(),
                **self._interpret(text, score, crop),
            })
        rows = []
        for row_id in dict.fromkeys(cell["row_id"] for cell in self.layout["cells"]):
            row = sorted((cell for cell in cells if cell["row_id"] == row_id),
                         key=lambda cell: cell["slot_index"])
            rows.append({
                "row_id": row_id,
                "text": "".join(cell["value"] for cell in row),
                "tokens": [{"value": cell["value"],
                            "mode": "letter" if cell["accepted"] else "unknown",
                            "reason": cell["reason"]} for cell in row],
                "complete": all(cell["accepted"] for cell in row),
            })
        result = {
            "method": "ocr",
            "algorithm_id": ALGORITHM_ID,
            "decoder_id": DECODER_ID,
            "image_size": [width, height],
            "mode": "letters",
            "rows": rows,
            "cells": cells,
            "coordinate_system": "Input-image pixels, origin top-left; calibrated face and printed-label polygons",
        }
        json.dumps(result, allow_nan=False)
        done = perf_counter()
        result["timing_ms"] = {
            "validation_and_rectification": (warped - start) * 1000,
            "ocr_recognition": (recognized - warped) * 1000,
            "normalization_periods_and_serialization": (done - recognized) * 1000,
            "image_to_result": (done - start) * 1000,
        }
        return result, crops
