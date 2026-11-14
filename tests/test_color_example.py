import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from color_reader import ColorReader, classify_color, extract_color
from recording_pipeline import WordRecorder
from simulate_recording_dataset import make_reference


class ColorExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.dictionary = json.loads(
            (Path(__file__).resolve().parents[1] / "color_dictionary.json").read_text()
        )

    def test_assignment_is_in_requested_order(self):
        self.assertEqual(
            "".join(e["letter"] for e in self.dictionary["entries"]), "JUDYFOXRAIN"
        )
        self.assertEqual(self.dictionary["entries"][8]["color_name"], "dark teal")
        self.assertEqual(self.dictionary["entries"][9]["color_name"], "near-black")

    def test_flat_color_without_any_studs_is_sufficient(self):
        for entry in self.dictionary["entries"]:
            crop = np.empty((160, 80, 3), np.uint8)
            crop[:] = entry["swatch_rgb"][::-1]
            result = classify_color(extract_color(crop), self.dictionary)
            self.assertEqual(result["letter_candidate"], entry["letter"])

    def test_unknown_and_between_dark_colors_are_rejected(self):
        entries = self.dictionary["entries"]
        middle = (np.asarray(entries[8]["lab"]) + entries[9]["lab"]) / 2
        self.assertIsNone(classify_color(middle, self.dictionary)["letter_candidate"])
        magenta = np.full((160, 80, 3), (255, 0, 255), np.uint8)
        self.assertIsNone(
            classify_color(extract_color(magenta), self.dictionary)["letter_candidate"]
        )

    def test_recording_color_mode_never_calls_stud_detection(self):
        reference, layout = make_reference()
        frame = reference.copy()
        frame[220:380, 230:310] = self.dictionary["entries"][0]["swatch_rgb"][::-1]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "braille_reader.detect_dots",
                side_effect=AssertionError("Stud detector must not run"),
            ),
        ):
            recorder = WordRecorder(
                reference,
                layout,
                Path(temporary) / "words.json",
                reader_factory=partial(ColorReader, dictionary=self.dictionary),
            )
            result = recorder.end(
                frame, event_id="flat-color", generation=recorder.generation
            )
            self.assertEqual(result["record"]["word"], "J")
            self.assertEqual(recorder.analysis_count, 1)
