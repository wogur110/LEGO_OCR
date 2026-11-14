import copy
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from braille_code import (
    DIGIT_DOTS,
    LETTER_DOTS,
    NUMBER_MASK,
    decode_mask,
    decode_sequence,
    dictionary_document,
    dots_to_mask,
)
from braille_reader import BrailleReader, detect_dots, validate_layout


class BrailleCodeTests(unittest.TestCase):
    def test_exported_dictionary_matches_runtime(self):
        stored = json.loads(
            (
                Path(__file__).resolve().parents[1] / "braille_dictionary.json"
            ).read_text()
        )
        self.assertEqual(stored, json.loads(json.dumps(dictionary_document())))

    def test_dictionary_matches_standard_unicode_alphabet(self):
        alphabet = "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵"
        self.assertEqual(len(set(map(dots_to_mask, LETTER_DOTS.values()))), 26)
        for letter, char in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", alphabet, strict=True):
            cell = decode_mask(ord(char) - 0x2800)
            self.assertEqual(cell["letter_candidate"], letter)
            self.assertEqual(dots_to_mask(LETTER_DOTS[letter]), ord(char) - 0x2800)

    def test_letter_digit_ambiguity_is_explicit(self):
        for letter, digit in zip("ABCDEFGHIJ", "1234567890", strict=True):
            self.assertEqual(LETTER_DOTS[letter], DIGIT_DOTS[digit])
            cell = decode_mask(dots_to_mask(LETTER_DOTS[letter]))
            self.assertEqual(cell["letter_candidate"], letter)
            self.assertEqual(cell["number_candidate"], digit)
            self.assertTrue(cell["requires_number_context"])
        self.assertIsNone(
            decode_mask(dots_to_mask(LETTER_DOTS["K"]))["number_candidate"]
        )

    def test_numeric_indicator_context_and_explicit_numeric_mode(self):
        sequence = [NUMBER_MASK, 3, 26, 3, 11]  # number indicator + B J B F
        self.assertEqual(decode_sequence(sequence, "auto")["text"], "2026")
        self.assertEqual(decode_sequence(sequence, "letters")["text"], "#BJBF")
        masks = [dots_to_mask(LETTER_DOTS[c]) for c in "ABCDEFGHIJ"]
        self.assertEqual(decode_sequence(masks, "numbers")["text"], "1234567890")
        self.assertEqual(decode_sequence([NUMBER_MASK, 1, 0, 1], "auto")["text"], "1 A")
        self.assertEqual(decode_sequence([NUMBER_MASK, 1, 48, 1], "auto")["text"], "1A")

    def test_unknown_cells_and_non_alphabet_patterns_are_not_guessed(self):
        self.assertEqual(decode_mask(8)["kind"], "unsupported")  # dot 4 is outside this subset.
        self.assertEqual(decode_sequence([13, 21, None, 29])["text"], "MO?N")
        self.assertEqual(
            decode_sequence([NUMBER_MASK, 1, None, 1], "auto")["text"], "1?A"
        )
        for value in [-1, 64, True, 1.5]:
            with self.assertRaises(ValueError):
                decode_mask(value)
        for dots in [[0], [7], [1, 1], [True]]:
            with self.assertRaises(ValueError):
                dots_to_mask(dots)

    def test_project_a_alias_does_not_change_standard_a_or_digit_one(self):
        alias = decode_mask(33)
        self.assertEqual(alias["raised_dots"], [1, 6])
        self.assertEqual(alias["letter_candidate"], "A")
        self.assertEqual(alias["mapping_source"], "project_alias")
        self.assertIsNone(alias["number_candidate"])
        self.assertEqual(decode_sequence([33, 1])["text"], "AA")
        self.assertEqual(decode_sequence([33, 1], "numbers")["text"], "?1")

    def test_period_survives_word_and_decimal_decoding(self):
        period = dots_to_mask([2, 5, 6])
        word = [dots_to_mask(LETTER_DOTS[c]) for c in "TREE"] + [period]
        self.assertEqual(decode_sequence(word)["text"], "TREE.")
        self.assertTrue(decode_sequence(word)["complete"])
        self.assertEqual(decode_mask(period)["kind"], "punctuation")
        self.assertIsNone(decode_mask(period)["letter_candidate"])
        self.assertEqual(decode_sequence([NUMBER_MASK, 1, period, 3], "auto")["text"], "1.2")

    def test_photo_last_row_symbols_and_question_mark_completeness(self):
        # Explicit patterns from the six bricks, independent of the dictionary.
        masks = [dots_to_mask(dots) for dots in [(2, 3, 6), (2, 3, 5), (3,), (2,), (3, 6), (3, 4, 5, 6)]]
        decoded = decode_sequence(masks)
        self.assertEqual(decoded["text"], "?!',-#")
        self.assertTrue(decoded["complete"])
        self.assertEqual(decoded["tokens"][0]["reason"], "punctuation")
        self.assertEqual(decode_mask(masks[-1])["indicator"], "number")
        self.assertFalse(decode_sequence([None])["complete"])
        self.assertFalse(decode_sequence([8])["complete"])
        self.assertTrue(decode_sequence([masks[0]])["complete"])
        self.assertFalse(decode_sequence([masks[0], None])["complete"])

    def test_punctuation_numeric_context(self):
        self.assertEqual(decode_sequence([NUMBER_MASK, 1, 2, 3], "auto")["text"], "1,2")
        for punctuation in [38, 22, 4, 36]:  # question, exclamation, apostrophe, hyphen
            with self.subTest(punctuation=punctuation):
                self.assertEqual(decode_sequence([NUMBER_MASK, 1, punctuation, 3], "auto")["text"][-1], "B")
                self.assertEqual(decode_sequence([1, punctuation, 3], "numbers")["text"][-1], "2")


class StudGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)

    def setUp(self):
        self.layout = {
            "version": 1,
            "image_size": [100, 180],
            "cells": [
                {
                    "row_id": "word",
                    "slot_index": 0,
                    "quad": [[10, 10], [89, 10], [89, 169], [10, 169]],
                }
            ],
        }

    def test_printed_label_quarter_cannot_affect_detection(self):
        first = np.full((180, 100, 3), 180, np.uint8)
        cv2.circle(first, (30, 30), 12, (80, 80, 80), -1)
        cv2.circle(first, (28, 28), 10, (220, 220, 220), -1)
        second = first.copy()
        second[140:169, 10:90] = np.random.default_rng(7).integers(
            0, 256, (29, 80, 3), dtype=np.uint8
        )
        a, crops_a = BrailleReader(self.layout).read(first)
        b, crops_b = BrailleReader(self.layout).read(second)
        np.testing.assert_array_equal(crops_a[0], crops_b[0])
        self.assertEqual(a["cells"][0]["bits"], b["cells"][0]["bits"])

    def test_blank_or_low_detail_image_is_uncertain(self):
        image = np.full((180, 100, 3), 180, np.uint8)
        result, _ = BrailleReader(self.layout).read(image)
        self.assertEqual(result["cells"][0]["bits"], [None] * 6)
        self.assertFalse(result["cells"][0]["accepted"])
        self.assertEqual(result["rows"][0]["text"], "?")

    def test_tiny_studs_are_not_upsampled_into_confident_letters(self):
        layout = copy.deepcopy(self.layout)
        layout["cells"][0]["quad"] = [[10, 10], [30, 10], [30, 50], [10, 50]]
        image = np.random.default_rng(1).integers(0, 256, (180, 100, 3), dtype=np.uint8)
        result, _ = BrailleReader(layout).read(image)
        self.assertEqual(result["cells"][0]["reason"], "too_small")
        self.assertEqual(result["rows"][0]["text"], "?")

    def test_invalid_or_ambiguous_geometry_rejected(self):
        for mutation in ["duplicate", "crossed", "missing_index", "size"]:
            layout = copy.deepcopy(self.layout)
            if mutation == "duplicate":
                layout["cells"] *= 2
            elif mutation == "crossed":
                layout["cells"][0]["quad"] = [[10, 10], [89, 169], [10, 169], [89, 10]]
            elif mutation == "missing_index":
                layout["cells"][0]["slot_index"] = 1
            else:
                layout["image_size"] = [200, 180]
            with self.assertRaises(ValueError):
                validate_layout(layout, (100, 180))
        with self.assertRaises(ValueError):
            detect_dots(np.zeros((160, 80, 3), np.uint8), None)


if __name__ == "__main__":
    unittest.main()
