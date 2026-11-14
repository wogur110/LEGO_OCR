"""Deterministic synthetic stud/texture separation checks.

These fixtures model a ridged monochrome face and visibly shaded circular
studs. They are regression tests, not measured real-camera accuracy.
"""

import unittest

import cv2
import numpy as np

from braille_reader import BrailleReader, detect_dots


def textured_face(dots=(), *, gray=160, direction=45, amplitude=15):
    """Return an 80x120 BGR stud-only face without text or colored cues."""
    yy, xx = np.mgrid[:120, :80]
    angle = np.deg2rad(direction)
    ridges = np.sin(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) / 5)
    noise = np.random.default_rng(17).normal(0, 1.5, (120, 80))
    surface = np.clip(gray + amplitude * ridges + noise, 0, 255).astype(np.uint8)
    # Left column is dots 1,2,3; right column is dots 4,5,6.
    for dot, (cx, cy) in enumerate(
        [(20, 20), (20, 60), (20, 100), (60, 20), (60, 60), (60, 100)], start=1
    ):
        if dot not in dots:
            continue
        cv2.circle(surface, (cx, cy), 12, gray - 65, -1, lineType=cv2.LINE_AA)
        cv2.circle(
            surface, (cx - 2, cy - 2), 10, min(gray + 45, 255), -1,
            lineType=cv2.LINE_AA,
        )
    return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)


class StudTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)

    def detect(self, face):
        return detect_dots(face, cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)))

    def assert_dots(self, face, dots):
        self.assertEqual(self.detect(face)["bits"], [i in dots for i in range(1, 7)])

    def test_parallel_surface_ridges_do_not_invent_studs(self):
        for direction in (0, 45, 90):
            with self.subTest(direction=direction):
                result = self.detect(textured_face(direction=direction))
                # Rejecting the crop as uncertain is allowed; invented studs are not.
                self.assertNotIn(True, result["bits"])

    def test_visible_studs_survive_surface_ridges(self):
        for dots in ((2, 4, 5), (1, 3, 4, 6)):
            for direction in (0, 45, 90):
                with self.subTest(dots=dots, direction=direction):
                    self.assert_dots(textured_face(dots, direction=direction), dots)

    def test_same_geometry_is_preserved_across_gray_backgrounds(self):
        dots = (2, 4, 5)
        for gray in (95, 160, 210):
            with self.subTest(gray=gray):
                self.assert_dots(textured_face(dots, gray=gray), dots)

    def test_smooth_background_studs_are_not_lost(self):
        for gray in (95, 160, 210):
            with self.subTest(gray=gray):
                self.assert_dots(textured_face((1,), gray=gray, amplitude=0), (1,))

    def test_entire_printed_label_quarter_is_excluded(self):
        layout = {
            "version": 1,
            "image_size": [100, 180],
            "cells": [{
                "row_id": "cell", "slot_index": 0,
                "quad": [[10, 10], [89, 10], [89, 169], [10, 169]],
            }],
        }
        first = np.full((180, 100, 3), 160, dtype=np.uint8)
        first[10:130, 10:90] = textured_face((2, 4, 5))
        second = first.copy()
        # The prior test starts at y=140, leaving ten label rows untested.
        second[130:170, 10:90] = np.random.default_rng(7).integers(
            0, 256, (40, 80, 3), dtype=np.uint8
        )
        a, crops_a = BrailleReader(layout).read(first)
        b, crops_b = BrailleReader(layout).read(second)
        np.testing.assert_array_equal(crops_a[0], crops_b[0])
        self.assertEqual(a["cells"][0]["bits"], b["cells"][0]["bits"])


if __name__ == "__main__":
    unittest.main()
