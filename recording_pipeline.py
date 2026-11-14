"""One-frame END / WRITE_AGAIN / RESET word capture in calibrated board slots.

No START, frame history, or temporal voting is used. A board snapshot is separate
from saved word history. Coordinates are board-reference pixels, not 3D poses.
"""

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from board_alignment import BoardAligner, project
from braille_reader import BrailleReader


class WordRecorder:
    def __init__(
        self,
        empty_reference,
        layout,
        storage_path,
        board_id="simulation-board",
        reader_factory=None,
    ):
        self.reader_factory = reader_factory or BrailleReader
        self.layout = copy.deepcopy(layout)
        self.storage_path = Path(storage_path)
        if len({c["row_id"] for c in layout["cells"]}) != 1:
            raise ValueError("The capture prototype accepts one calibrated row")
        self.aligner = BoardAligner(empty_reference, layout, method="aruco")
        reference_reader = self.reader_factory(layout)
        reference_result, self.empty_crops = reference_reader.read(empty_reference)
        method = reference_result.get("method", "six_dot")
        calibration = hashlib.sha256(
            json.dumps(layout, sort_keys=True).encode()
            + empty_reference.tobytes()
            + json.dumps(
                getattr(reference_reader, "dictionary", None), sort_keys=True
            ).encode()
        ).hexdigest()
        self._data = {
            "version": 2,
            "board_id": board_id,
            "recognition_method": method,
            "calibration_id": calibration,
            "generation": str(uuid4()),
            "words": [],
            "board_state": [],
            "receipts": {},
            "rewrite_pending": None,
            "baseline_ready": True,
        }
        if self.storage_path.exists():
            stored = json.loads(self.storage_path.read_text())
            if stored.get("board_id") != board_id:
                raise ValueError("Storage belongs to a different board")
            if stored.get("version") == 1:
                # Old START/END storage contains no reliable current-board snapshot.
                # Preserve its words; the first readable END establishes a baseline.
                self._data["words"] = stored["words"]
                self._data["baseline_ready"] = not bool(stored["words"])
            elif stored.get("version") == 2:
                if (
                    stored.get("calibration_id") != calibration
                    or stored.get("recognition_method") != method
                ):
                    raise ValueError("Storage uses a different calibration or reader")
                self._data = stored
            else:
                raise ValueError("Unsupported storage schema")
        self.analysis_count = 0

    @property
    def words(self):
        return self._data["words"]

    @property
    def board_state(self):
        return self._data["board_state"]

    @property
    def generation(self):
        return self._data["generation"]

    @property
    def rewrite_pending(self):
        return self._data["rewrite_pending"]

    def _persist(self, data):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_path.with_name(self.storage_path.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.storage_path)

    def _commit(self, data):
        self._persist(data)
        self._data = data

    def _observe(self, frame):
        self.analysis_count += 1
        homography, alignment = self.aligner.align(frame)
        if homography is None:
            return None, alignment
        adjusted = copy.deepcopy(self.layout)
        adjusted["image_size"] = [frame.shape[1], frame.shape[0]]
        inverse = np.linalg.inv(homography)
        for cell in adjusted["cells"]:
            cell["quad"] = project(cell["quad"], inverse).tolist()
        try:
            result, crops = self.reader_factory(adjusted).read(frame)
        except ValueError as error:
            return None, {
                "status": "lost",
                "reason": "slots_outside_frame",
                "detail": str(error),
            }
        observations = []
        for cell, crop, empty, calibrated in zip(
            result["cells"], crops, self.empty_crops, self.layout["cells"], strict=True
        ):
            difference = float(
                np.abs(crop.astype(np.float32) - empty.astype(np.float32)).mean()
            )
            # Keep a rejection band so borderline background changes cannot silently
            # drop one letter from an otherwise accepted word.
            presence = (
                "present"
                if difference > 25
                else "empty"
                if difference <= 15
                else "uncertain"
            )
            observations.append(
                {
                    "slot": calibrated["slot_index"],
                    "presence": presence,
                    "appearance_change": difference,
                    "letter": cell["letter_candidate"]
                    if presence == "present"
                    else None,
                    "bits": cell.get("bits"),
                    "reason": cell["reason"],
                    "board_polygon": calibrated["quad"],
                    "image_polygon": cell["face_polygon"],
                    "sharpness": cell.get("sharpness"),
                }
            )
        return observations, alignment

    def _finish(
        self, result, event_id, signal, started, analysis_started=None, data=None
    ):
        result["generation"] = self.generation
        result.setdefault("chatgpt_payload", None)
        result["frames_analyzed"] = int(analysis_started is not None)
        if analysis_started is not None:
            json.dumps(result, allow_nan=False)
            result["analysis_latency_ms"] = (perf_counter() - analysis_started) * 1000
        data = copy.deepcopy(self._data) if data is None else data
        data["receipts"][event_id] = {
            "signal": signal,
            "status": result["status"],
            "reason": result.get("reason"),
            "word_id": result.get("record", {}).get("word_id"),
            "removed_word_id": result.get("removed_word_id"),
        }
        self._commit(data)
        result["rewrite_pending"] = self.rewrite_pending is not None
        json.dumps(result, allow_nan=False)
        result["latency_ms"] = (perf_counter() - started) * 1000
        return result

    def _check_event(self, signal, event_id, generation):
        if generation != self.generation:
            return {
                "status": "ignored",
                "reason": "stale_generation",
                "frames_analyzed": 0,
                "chatgpt_payload": None,
            }
        if not isinstance(event_id, str) or not event_id.strip():
            return {
                "status": "rejected",
                "reason": "event_id_required",
                "frames_analyzed": 0,
                "chatgpt_payload": None,
            }
        receipt = self._data["receipts"].get(event_id)
        if receipt is not None:
            if receipt["signal"] != signal:
                return {
                    "status": "rejected",
                    "reason": "event_id_reused_for_different_signal",
                    "frames_analyzed": 0,
                    "chatgpt_payload": None,
                }
            record = next(
                (w for w in self.words if w["word_id"] == receipt["word_id"]), None
            )
            return {
                "status": "already_saved"
                if receipt["status"] in {"saved", "replaced"}
                else "already_processed",
                "original_status": receipt["status"],
                "record": copy.deepcopy(record),
                "chatgpt_payload": None,
                "frames_analyzed": 0,
                "generation": self.generation,
                "rewrite_pending": self.rewrite_pending is not None,
            }
        return None

    def end(self, frame, *, event_id, generation):
        """Analyze exactly one final frame and commit its newly observed letters."""
        started = perf_counter()
        signal = "END"
        guard = self._check_event(signal, event_id, generation)
        if guard is not None:
            return guard
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) < 2
        ):
            return {
                "status": "rejected",
                "reason": "decoded_bgr_frame_required",
                "frames_analyzed": 0,
                "chatgpt_payload": None,
            }
        analysis_started = perf_counter()
        observations, alignment = self._observe(frame)
        common = {"alignment": alignment, "observations": observations or []}
        if self.rewrite_pending is not None:
            common["removed_word_id"] = self.rewrite_pending["removed_word_id"]
        if observations is None:
            return self._finish(
                {**common, "status": "needs_review", "reason": "tracking_lost"},
                event_id,
                signal,
                started,
                analysis_started,
            )
        uncertain = [
            o["slot"]
            for o in observations
            if o["presence"] == "uncertain"
            or (o["presence"] == "present" and o["letter"] is None)
        ]
        if uncertain:
            return self._finish(
                {
                    **common,
                    "status": "needs_review",
                    "reason": "uncertain_presence_or_letter",
                    "uncertain_slots": uncertain,
                },
                event_id,
                signal,
                started,
                analysis_started,
            )
        current = {o["slot"]: o for o in observations if o["presence"] == "present"}
        data = copy.deepcopy(self._data)
        if not data["baseline_ready"]:
            owners = {}
            for word in self.words:
                for letter in word["letters"]:
                    owners[(letter["slot"], letter["letter"])] = word["word_id"]
            data["board_state"] = [
                {
                    "slot": slot,
                    "letter": o["letter"],
                    "word_id": owners.get((slot, o["letter"])),
                }
                for slot, o in sorted(current.items())
            ]
            data["baseline_ready"] = True
            return self._finish(
                {
                    **common,
                    "status": "baseline_initialized",
                    "reason": "legacy_words_preserved",
                },
                event_id,
                signal,
                started,
                analysis_started,
                data,
            )
        previous = {s["slot"]: s for s in self.board_state}
        added = [
            o
            for slot, o in current.items()
            if slot not in previous or previous[slot]["letter"] != o["letter"]
        ]
        added.sort(key=lambda o: float(np.mean(o["board_polygon"], axis=0)[0]))
        common["new_slots"] = [o["slot"] for o in added]
        common["removed_slots"] = sorted(set(previous) - set(current))
        common["unchanged_slots"] = sorted(set(current) - set(common["new_slots"]))
        if not added and self.rewrite_pending is not None:
            return self._finish(
                {
                    **common,
                    "status": "needs_review",
                    "reason": "no_replacement_letters",
                },
                event_id,
                signal,
                started,
                analysis_started,
            )
        record = None
        if added:
            identifier = str(uuid4())
            letters = [
                {
                    "track_id": identifier + ":" + str(o["slot"]),
                    "slot": o["slot"],
                    "letter": o["letter"],
                    "board_polygon": o["board_polygon"],
                    "image_polygon": o["image_polygon"],
                }
                for o in added
            ]
            points = np.float32([p for c in letters for p in c["board_polygon"]])
            image_points = np.float32([p for c in letters for p in c["image_polygon"]])
            record = {
                "word_id": identifier,
                "event_id": event_id,
                "generation": self.generation,
                "word": "".join(c["letter"] for c in letters),
                "letters": letters,
                "board_id": data["board_id"],
                "recognition_method": data["recognition_method"],
                "coordinate_system": "board_reference_pixels",
                "box_xyxy": [
                    *points.min(axis=0).tolist(),
                    *points.max(axis=0).tolist(),
                ],
                "image_box_xyxy": [
                    *image_points.min(axis=0).tolist(),
                    *image_points.max(axis=0).tolist(),
                ],
                "position_source": "calibrated placement slots",
            }
            data["words"].append(record)
            data["rewrite_pending"] = None
        data["board_state"] = [
            {
                "slot": slot,
                "letter": o["letter"],
                "word_id": record["word_id"]
                if slot in common["new_slots"]
                else previous[slot].get("word_id"),
            }
            for slot, o in sorted(current.items())
        ]
        result = {**common, "status": "no_new_letters"}
        if record is not None:
            result.update(
                status="replaced" if self.rewrite_pending is not None else "saved",
                record=record,
                chatgpt_payload=record["word"],
            )
        return self._finish(result, event_id, signal, started, analysis_started, data)

    def write_again(self, *, event_id, generation):
        """Undo the latest word only; defer all image analysis to the next END."""
        started = perf_counter()
        signal = "WRITE_AGAIN"
        guard = self._check_event(signal, event_id, generation)
        if guard is not None:
            return guard
        if self.rewrite_pending is not None:
            return self._finish(
                {
                    "status": "already_pending",
                    "removed_word_id": self.rewrite_pending["removed_word_id"],
                },
                event_id,
                signal,
                started,
            )
        if not self._data["baseline_ready"]:
            return self._finish(
                {
                    "status": "needs_review",
                    "reason": "legacy_storage_needs_baseline_end",
                },
                event_id,
                signal,
                started,
            )
        if not self.words:
            return self._finish(
                {"status": "nothing_to_rewrite"}, event_id, signal, started
            )
        data = copy.deepcopy(self._data)
        removed = data["words"].pop()
        released = [
            s["slot"]
            for s in data["board_state"]
            if s.get("word_id") == removed["word_id"]
        ]
        data["board_state"] = [
            s for s in data["board_state"] if s.get("word_id") != removed["word_id"]
        ]
        data["rewrite_pending"] = {
            "removed_word_id": removed["word_id"],
            "released_slots": released,
        }
        return self._finish(
            {"status": "removed", "removed_word_id": removed["word_id"]},
            event_id,
            signal,
            started,
            data=data,
        )

    def reset(self, *, generation):
        if generation != self.generation:
            return {
                "status": "ignored",
                "reason": "stale_generation",
                "frames_analyzed": 0,
                "chatgpt_payload": None,
            }
        data = copy.deepcopy(self._data)
        data.update(
            words=[],
            board_state=[],
            receipts={},
            rewrite_pending=None,
            baseline_ready=True,
            generation=str(uuid4()),
        )
        self._commit(data)
        return {
            "status": "reset",
            "stored_words": 0,
            "generation": self.generation,
            "frames_analyzed": 0,
            "chatgpt_payload": None,
        }

    def handle_signal(self, signal, frame=None, *, event_id=None, generation=None):
        """Synchronous adapter. A Unity transport must preserve capture/signal order."""
        signal = signal.upper()
        if signal == "END":
            return self.end(frame, event_id=event_id, generation=generation)
        if signal == "WRITE_AGAIN":
            return self.write_again(event_id=event_id, generation=generation)
        if signal == "RESET":
            return self.reset(generation=generation)
        return {
            "status": "ignored",
            "reason": "unsupported_signal",
            "frames_analyzed": 0,
            "chatgpt_payload": None,
        }
