"""Project aliases must survive six-dot reading and one-frame word storage.

The rendered fixtures have explicit physical dots, independent of the runtime
dictionary. They check integration, not real-camera recognition accuracy.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from braille_code import decode_mask, decode_sequence
from braille_reader import BrailleReader
from recording_pipeline import WordRecorder


def two_cell_scene(first_dots=(1, 6)):
    reference = np.full((180, 210, 3), 40, dtype=np.uint8)
    layout = {
        "version": 1,
        "image_size": [210, 180],
        "cells": [
            {
                "row_id": "word",
                "slot_index": slot,
                "quad": [[x, 10], [x + 79, 10], [x + 79, 169], [x, 169]],
            }
            for slot, x in enumerate((10, 120))
        ],
    }
    frame = reference.copy()
    centers = ((20, 20), (20, 60), (20, 100), (60, 20), (60, 60), (60, 100))
    for x, dots in ((10, first_dots), (120, (2, 5, 6))):
        face = np.full((160, 80, 3), 160, dtype=np.uint8)
        for dot, (cx, cy) in enumerate(centers, start=1):
            if dot in dots:
                cv2.circle(face, (cx, cy), 12, (95, 95, 95), -1, cv2.LINE_AA)
                cv2.circle(face, (cx - 2, cy - 2), 10, (205, 205, 205), -1, cv2.LINE_AA)
        frame[10:170, x : x + 80] = face
    return reference, layout, frame


class ProjectDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)

    def test_custom_a_alias_does_not_redefine_standard_digit_one(self):
        self.assertEqual(decode_mask(33)["letter_candidate"], "A")
        self.assertIsNone(decode_mask(33)["number_candidate"])
        self.assertEqual(decode_mask(1)["letter_candidate"], "A")
        self.assertEqual(decode_mask(1)["number_candidate"], "1")
        self.assertEqual(decode_sequence([33], "numbers")["text"], "?")
        self.assertEqual(decode_sequence([1], "numbers")["text"], "1")

    def test_real_reader_accepts_custom_and_standard_a_followed_by_period(self):
        for dots in ((1, 6), (1,)):
            with self.subTest(a_dots=dots):
                _, layout, frame = two_cell_scene(dots)
                result, _ = BrailleReader(layout).read(frame)
                self.assertEqual(result["rows"][0]["text"], "A.")
                self.assertTrue(result["rows"][0]["complete"])
                self.assertEqual(result["cells"][0]["raised_dots"], list(dots))
                period = result["cells"][1]
                self.assertEqual(period["raised_dots"], [2, 5, 6])
                self.assertTrue(period["accepted"])
                self.assertEqual(period["kind"], "punctuation")
                self.assertIsNone(period["letter_candidate"])
                self.assertEqual(period["value"], ".")

    def test_end_preserves_period_in_storage_and_chatgpt_payload(self):
        reference, layout, frame = two_cell_scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.json"
            recorder = WordRecorder(reference, layout, path, alignment_mode="none")
            result = recorder.end(frame, event_id="a_period", generation=recorder.generation)
            self.assertEqual(result["status"], "saved")
            self.assertEqual(result["frames_analyzed"], 1)
            self.assertEqual(result["record"]["word"], "A.")
            self.assertEqual(result["chatgpt_payload"], "A.")
            self.assertEqual([cell["letter"] for cell in result["record"]["letters"]], ["A", "."])
            self.assertEqual(result["record"]["box_xyxy"], [10.0, 10.0, 199.0, 169.0])
            self.assertEqual(json.loads(path.read_text())["words"][0]["word"], "A.")

            restored = WordRecorder(reference, layout, path, alignment_mode="none")
            repeated = restored.end(frame, event_id="same_frame", generation=restored.generation)
            self.assertEqual(repeated["status"], "no_new_letters")
            self.assertEqual([word["word"] for word in restored.words], ["A."])

    def test_recognized_question_mark_is_complete_and_storable(self):
        reference, layout, frame = two_cell_scene((2, 3, 6))
        result, _ = BrailleReader(layout).read(frame)
        self.assertEqual(result["rows"][0]["text"], "?.")
        self.assertTrue(result["rows"][0]["complete"])
        self.assertEqual(result["cells"][0]["kind"], "punctuation")
        self.assertTrue(result["cells"][0]["accepted"])
        with tempfile.TemporaryDirectory() as directory:
            recorder = WordRecorder(reference, layout, Path(directory) / "words.json", alignment_mode="none")
            saved = recorder.end(frame, event_id="question", generation=recorder.generation)
            self.assertEqual(saved["status"], "saved")
            self.assertEqual(saved["record"]["word"], "?.")
            self.assertEqual(saved["chatgpt_payload"], "?.")


if __name__ == "__main__":
    unittest.main()
