import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from recording_pipeline import WordRecorder
from simulate_recording_dataset import atlas_from_photos, compose_frame, make_reference

ROOT = Path(__file__).resolve().parents[1]


class NoAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.atlas, _ = atlas_from_photos(ROOT)

    def test_marker_free_signal_flow_never_constructs_an_aligner(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "recording_pipeline.BoardAligner",
                side_effect=AssertionError("Alignment ran"),
            ),
            patch(
                "cv2.aruco.ArucoDetector",
                side_effect=AssertionError("Marker detection ran"),
            ),
            patch(
                "cv2.aruco.getPredefinedDictionary",
                side_effect=AssertionError("Markers requested"),
            ),
        ):
            reference, layout = make_reference(with_markers=False)
            self.assertTrue(np.all(reference == 80))
            recorder = WordRecorder(
                reference, layout, Path(temp) / "storage.json", alignment_mode="none"
            )
            frame, _ = compose_frame(
                reference, layout, self.atlas, dict(enumerate("JUDY"))
            )
            result = recorder.end(
                frame, event_id="first", generation=recorder.generation
            )
            self.assertEqual(result["record"]["word"], "JUDY")
            self.assertEqual(result["alignment"]["status"], "disabled")
            self.assertEqual(
                result["record"]["coordinate_system"], "calibrated_image_pixels"
            )
            self.assertIsNone(recorder.aligner)
            for observation, cell in zip(
                result["observations"], layout["cells"], strict=True
            ):
                self.assertEqual(observation["image_polygon"], cell["quad"])
            self.assertEqual(
                recorder.write_again(event_id="undo", generation=recorder.generation)[
                    "frames_analyzed"
                ],
                0,
            )
            self.assertEqual(recorder.words, [])
            frame, _ = compose_frame(
                reference, layout, self.atlas, dict(enumerate("TREE"))
            )
            self.assertEqual(
                recorder.end(
                    frame, event_id="corrected", generation=recorder.generation
                )["record"]["word"],
                "TREE",
            )
            recorder.reset(generation=recorder.generation)
            self.assertEqual(json.loads(recorder.storage_path.read_text())["words"], [])

    def test_size_change_is_rejected_without_rescaling_fixed_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            reference, layout = make_reference(with_markers=False)
            recorder = WordRecorder(
                reference, layout, Path(temp) / "storage.json", alignment_mode="none"
            )
            result = recorder.end(
                reference[::2, ::2], event_id="resized", generation=recorder.generation
            )
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["reason"], "fixed_layout_image_size_mismatch")
            self.assertEqual(recorder.words, [])
            self.assertEqual(recorder.board_state, [])

    def test_storage_cannot_mix_alignment_coordinate_systems(self):
        with tempfile.TemporaryDirectory() as temp:
            reference, layout = make_reference()
            path = Path(temp) / "storage.json"
            recorder = WordRecorder(reference, layout, path)
            recorder.reset(generation=recorder.generation)
            with self.assertRaisesRegex(ValueError, "alignment mode"):
                WordRecorder(reference, layout, path, alignment_mode="none")
            stored = json.loads(path.read_text())
            stored.pop("alignment_mode")
            path.write_text(json.dumps(stored))
            loaded = WordRecorder(reference, layout, path)
            self.assertEqual(loaded.alignment_mode, "aruco")
