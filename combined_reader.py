"""Parallel, per-brick OCR-first recognition with six-dot fallback."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import perf_counter

import numpy as np

from braille_code import DECODER_ID as BRAILLE_DECODER_ID, OUTPUT_CHARACTERS
from braille_geometry import validate_layout
from braille_reader import BrailleReader
from ocr_reader import CalibratedOCRReader, DECODER_ID as OCR_DECODER_ID

ALGORITHM_ID = "parallel_ocr_braille_v1"


def _finite_score(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


class CombinedReader:
    """Reuse one reader across frames; close it when the capture session ends.

    Calls are serialized to protect each backend's mutable state. The input is
    copied because six-dot work may still be stopping after an OCR-only return.
    Outstanding work is canceled and drained before another pair is submitted,
    keeping at most one frame in flight. That cleanup is included in latency.
    """

    def __init__(self, layout, *, threads=4, ocr_confidence=0.85, ocr_engine=None):
        if not isinstance(layout, dict) or len(layout.get("image_size", [])) != 2:
            raise ValueError("Layout needs image_size [width,height]")
        validate_layout(layout, tuple(layout["image_size"]))
        threshold = _finite_score(ocr_confidence)
        if threshold is None or not 0 <= threshold <= 1:
            raise ValueError("ocr_confidence must be between 0 and 1")
        self.layout = copy.deepcopy(layout)
        self.ocr_confidence = threshold
        self._keys = {(c["row_id"], c["slot_index"]) for c in layout["cells"]}
        self._ocr = CalibratedOCRReader(
            self.layout, threads=threads, engine=ocr_engine, min_score=threshold
        )
        self._braille = BrailleReader(self.layout)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lego")
        self._lock = Lock()
        self._pending = None
        self._closed = False

    def _cancel_pending(self):
        if self._pending is not None:
            ocr, braille, stop = self._pending
            stop.set()
            ocr.cancel()
            braille.cancel()

    def _drain_pending(self):
        self._cancel_pending()
        if self._pending is not None:
            for future in self._pending[:2]:
                try:
                    future.result()
                except Exception:
                    # Unused results/errors cannot invalidate accepted OCR.
                    pass
            self._pending = None

    def _collect(self, future):
        try:
            result, _ = future.result()
            cells = result["cells"]
            indexed = {(c["row_id"], c["slot_index"]): c for c in cells}
            if len(indexed) != len(cells) or set(indexed) != self._keys:
                raise ValueError("Reader must return each calibrated cell exactly once")
            return indexed, None
        except Exception as error:
            return {}, f"{type(error).__name__}: {error}"

    def _ocr_accepted(self, cell):
        if (not cell.get("accepted") or not isinstance(cell.get("value"), str)
                or cell["value"] not in OUTPUT_CHARACTERS):
            return False
        if cell.get("source") == "printed_period_shape":
            return cell["value"] == "." and bool(
                cell.get("period_diagnostics", {}).get("accepted")
            )
        score = _finite_score(cell.get("score"))
        return score is not None and score >= self.ocr_confidence

    def _fuse(self, ocr, braille, errors):
        cells = []
        for geometry in self.layout["cells"]:
            key = geometry["row_id"], geometry["slot_index"]
            printed, dots = ocr.get(key, {}), braille.get(key, {})
            if self._ocr_accepted(printed):
                chosen, source = printed, "ocr"
            elif (dots.get("accepted") and isinstance(dots.get("value"), str)
                  and dots["value"] in OUTPUT_CHARACTERS):
                chosen, source = dots, "braille"
            else:
                chosen, source = {"value": "?", "reason": "both_unreadable"}, "unknown"
            polygon = np.asarray(geometry["quad"], dtype=float)
            cells.append({
                "row_id": key[0], "slot_index": key[1],
                "value": chosen["value"], "accepted": source != "unknown",
                "source": source, "reason": chosen.get("reason", "recognized"),
                "face_polygon": polygon.tolist(),
                "bbox": np.concatenate((polygon.min(axis=0), polygon.max(axis=0))).tolist(),
                "center": polygon.mean(axis=0).tolist(),
                "ocr": {
                    "value": printed.get("value", "?"),
                    "accepted": self._ocr_accepted(printed),
                    "score": _finite_score(printed.get("score")),
                    "raw_text": printed.get("raw_ocr_text"),
                    "reason": printed.get("reason", "backend_failed"),
                    "source": printed.get("source"),
                },
                "braille": {
                    "value": dots.get("value", "?"),
                    "accepted": bool(dots.get("accepted")),
                    "bits": dots.get("bits"), "raised_dots": dots.get("raised_dots"),
                    "reason": dots.get("reason"),
                } if dots else None,
            })
        rows = []
        for row_id in dict.fromkeys(c["row_id"] for c in cells):
            row = sorted((c for c in cells if c["row_id"] == row_id),
                         key=lambda c: c["slot_index"])
            boxes = np.array([c["bbox"] for c in row])
            rows.append({
                "row_id": row_id, "text": "".join(c["value"] for c in row),
                "complete": all(c["accepted"] for c in row),
                "bbox": np.concatenate((boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0))).tolist(),
            })
        complete = all(row["complete"] for row in rows)
        return {
            "method": "combined", "algorithm_id": ALGORITHM_ID,
            "decoder_ids": {"ocr": OCR_DECODER_ID, "braille": BRAILLE_DECODER_ID},
            "image_size": list(self.layout["image_size"]), "mode": "letters",
            "ocr_min_score": self.ocr_confidence,
            "complete": complete, "needs_review": not complete,
            "rows": rows, "cells": cells, "backend_errors": errors,
            "coordinate_system": "Input-image pixels; bbox=[left,top,right,bottom]; calibrated full faces",
        }

    def read(self, image):
        """Return OCR immediately when complete; otherwise fill its gaps with dots."""
        started = perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("CombinedReader is closed")
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3):
                raise ValueError("Expected a decoded uint8 BGR image")
            validate_layout(self.layout, tuple(image.shape[1::-1]))
            self._drain_pending()
            frame = image.copy()
            frame.flags.writeable = False
            prepared = perf_counter()
            stop = Event()
            ocr_future = self._executor.submit(self._ocr.read, frame)
            braille_future = self._executor.submit(self._braille.read, frame, cancel_event=stop)
            self._pending = (ocr_future, braille_future, stop)
            try:
                ocr, ocr_error = self._collect(ocr_future)
                ocr_ready = perf_counter()
                braille, braille_error = {}, None
                used_fallback = not all(self._ocr_accepted(ocr.get(key, {})) for key in self._keys)
                if used_fallback:
                    braille, braille_error = self._collect(braille_future)
                    self._pending = None
                else:
                    self._cancel_pending()
                decided = perf_counter()
                errors = {name: error for name, error in
                          (("ocr", ocr_error), ("braille", braille_error)) if error}
                result = self._fuse(ocr, braille, errors)
                result["used_braille_fallback"] = used_fallback
                json.dumps(result, allow_nan=False)
                finished = perf_counter()
                result["timing_ms"] = {
                    "preparation_and_previous_cleanup": (prepared - started) * 1000,
                    "parallel_until_ocr": (ocr_ready - prepared) * 1000,
                    "fallback_wait": (decided - ocr_ready) * 1000,
                    "fusion_and_serialization": (finished - decided) * 1000,
                    "image_to_result": (finished - started) * 1000,
                }
                return result
            finally:
                self._cancel_pending()

    def close(self):
        with self._lock:
            if not self._closed:
                self._drain_pending()
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("CombinedReader is closed")
        return self

    def __exit__(self, *exc):
        self.close()
