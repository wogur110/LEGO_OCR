import copy
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from color_reader import DESTINATION, ColorReader
from recording_pipeline import WordRecorder
from simulate_recording_dataset import atlas_from_photos, compose_frame, make_reference

ROOT = Path(__file__).resolve().parents[1]


class RecordingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.reference, cls.layout = make_reference()
        braille_atlas, _ = atlas_from_photos(ROOT)
        dictionary = json.loads((ROOT / "color_dictionary.json").read_text())
        photo = cv2.imread(str(ROOT / "samples/lego_color.jpg"))
        layout = json.loads((ROOT / "layouts/lego_color.json").read_text())
        color_atlas = {}
        for entry in dictionary["entries"]:
            quad = layout["cells"][entry["source_index"]]["quad"]
            color_atlas[entry["letter"]] = cv2.warpPerspective(
                photo,
                cv2.getPerspectiveTransform(np.float32(quad), DESTINATION),
                (80, 160),
                flags=cv2.INTER_CUBIC,
            )
        cls.modes = {
            "six_dot": (None, braille_atlas),
            "color": (partial(ColorReader, dictionary=dictionary), color_atlas),
        }

    def make_recorder(self, mode):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "words.json"
        factory, _ = self.modes[mode]
        return WordRecorder(self.reference, self.layout, path, reader_factory=factory)

    def frame(self, mode, bricks):
        return compose_frame(self.reference, self.layout, self.modes[mode][1], bricks)[
            0
        ]

    def end(self, recorder, mode, bricks, event_id):
        return recorder.end(
            self.frame(mode, bricks), event_id=event_id, generation=recorder.generation
        )

    def test_one_final_frame_without_start_or_history(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                result = self.end(recorder, mode, dict(enumerate("JUDY")), "first")
                self.assertEqual(result["record"]["word"], "JUDY")
                self.assertEqual(result["frames_analyzed"], 1)
                self.assertEqual(recorder.analysis_count, 1)
                self.assertFalse(hasattr(recorder, "start"))
                with patch.object(
                    recorder, "_observe", side_effect=AssertionError("Unexpected CV")
                ):
                    for signal in ["FRAME", "START"]:
                        self.assertEqual(
                            recorder.handle_signal(signal)["status"], "ignored"
                        )

    def test_only_added_letters_and_changed_letters_are_new(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "j")
                result = self.end(recorder, mode, {0: "J", 1: "U"}, "u")
                self.assertEqual(result["record"]["word"], "U")
                self.assertEqual(result["unchanged_slots"], [0])
                result = self.end(recorder, mode, {0: "F", 1: "U"}, "f")
                self.assertEqual(result["record"]["word"], "F")
                self.assertEqual([w["word"] for w in recorder.words], ["J", "U", "F"])

    def test_unchanged_board_and_empty_board_preserve_saved_words(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "first")
                saved = copy.deepcopy(recorder.words)
                same = self.end(recorder, mode, {0: "J"}, "same")
                self.assertEqual(same["status"], "no_new_letters")
                clear = self.end(recorder, mode, {}, "clear")
                self.assertEqual(clear["status"], "no_new_letters")
                self.assertEqual(recorder.words, saved)
                self.assertEqual(recorder.board_state, [])
                self.assertEqual(
                    self.end(recorder, mode, {0: "J"}, "rebuild")["record"]["word"], "J"
                )

    def test_unreadable_slot_blocks_partial_word_and_keeps_board_snapshot(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "first")
                previous = copy.deepcopy(recorder.board_state)
                frame = self.frame(mode, {0: "J", 2: "D"})
                frame[220:380, 340:420] = (255, 0, 255)
                result = recorder.end(
                    frame, event_id="uncertain", generation=recorder.generation
                )
                self.assertEqual(result["status"], "needs_review")
                self.assertIn(1, result["uncertain_slots"])
                self.assertEqual(recorder.board_state, previous)
                self.assertEqual([w["word"] for w in recorder.words], ["J"])

    def test_tracking_loss_does_not_update_words_or_snapshot(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "first")
                previous = copy.deepcopy(recorder.board_state)
                result = recorder.end(
                    np.zeros_like(self.reference),
                    event_id="lost",
                    generation=recorder.generation,
                )
                self.assertEqual(result["reason"], "tracking_lost")
                self.assertEqual(recorder.board_state, previous)
                self.assertEqual([w["word"] for w in recorder.words], ["J"])

    def test_duplicate_end_and_reload_do_not_reanalyze_or_send(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "first")
                factory, _ = self.modes[mode]
                loaded = WordRecorder(
                    self.reference,
                    self.layout,
                    recorder.storage_path,
                    reader_factory=factory,
                )
                with patch.object(
                    loaded, "_observe", side_effect=AssertionError("Duplicate ran CV")
                ):
                    duplicate = loaded.end(
                        None, event_id="first", generation=loaded.generation
                    )
                self.assertEqual(duplicate["status"], "already_saved")
                self.assertIsNone(duplicate["chatgpt_payload"])
                self.assertEqual(duplicate["frames_analyzed"], 0)
                self.assertEqual(loaded.board_state, recorder.board_state)

    def test_write_again_only_undoes_latest_and_next_end_reads_replacement(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, dict(enumerate("JUDY")), "judy")
                first = copy.deepcopy(recorder.words[0])
                self.end(recorder, mode, dict(enumerate("FOX")), "fox")
                with patch.object(
                    recorder,
                    "_observe",
                    side_effect=AssertionError("WRITE_AGAIN ran CV"),
                ):
                    undo = recorder.write_again(
                        event_id="undo", generation=recorder.generation
                    )
                    duplicate = recorder.write_again(
                        event_id="undo", generation=recorder.generation
                    )
                    pending = recorder.write_again(
                        event_id="another-undo", generation=recorder.generation
                    )
                self.assertEqual(undo["status"], "removed")
                self.assertEqual(undo["frames_analyzed"], 0)
                self.assertEqual(duplicate["status"], "already_processed")
                self.assertEqual(pending["status"], "already_pending")
                self.assertEqual(recorder.words, [first])
                self.assertEqual(
                    json.loads(recorder.storage_path.read_text())["words"], [first]
                )
                result = self.end(
                    recorder, mode, dict(enumerate("RAIN")), "replacement"
                )
                self.assertEqual(result["status"], "replaced")
                self.assertEqual(result["record"]["word"], "RAIN")
                self.assertEqual(recorder.words[0], first)
                self.assertIsNone(recorder.rewrite_pending)

    def test_rewrite_rereads_unchanged_latest_letters_but_excludes_older_word(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "j")
                self.end(recorder, mode, {0: "J", 1: "U"}, "u")
                recorder.write_again(event_id="undo", generation=recorder.generation)
                result = self.end(recorder, mode, {0: "J", 1: "U"}, "same-letter")
                self.assertEqual(result["record"]["word"], "U")
                self.assertEqual([w["word"] for w in recorder.words], ["J", "U"])

    def test_unreadable_replacement_stays_removed_and_pending_across_restart(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "j")
                self.end(recorder, mode, {0: "J", 1: "U"}, "u")
                recorder.write_again(event_id="undo", generation=recorder.generation)
                bad = self.frame(mode, {0: "J"})
                bad[220:380, 340:420] = (255, 0, 255)
                self.assertEqual(
                    recorder.end(bad, event_id="bad", generation=recorder.generation)[
                        "status"
                    ],
                    "needs_review",
                )
                factory, _ = self.modes[mode]
                loaded = WordRecorder(
                    self.reference,
                    self.layout,
                    recorder.storage_path,
                    reader_factory=factory,
                )
                self.assertIsNotNone(loaded.rewrite_pending)
                self.assertEqual([w["word"] for w in loaded.words], ["J"])
                self.assertEqual(
                    loaded.write_again(event_id="repeat", generation=loaded.generation)[
                        "status"
                    ],
                    "already_pending",
                )
                self.assertEqual(
                    self.end(loaded, mode, {0: "J", 1: "D"}, "good")["record"]["word"],
                    "D",
                )

    def test_write_again_empty_storage_is_noop(self):
        for mode in self.modes:
            recorder = self.make_recorder(mode)
            with patch.object(
                recorder, "_observe", side_effect=AssertionError("Unexpected CV")
            ):
                result = recorder.write_again(
                    event_id="undo", generation=recorder.generation
                )
            self.assertEqual(result["status"], "nothing_to_rewrite")
            self.assertIsNone(recorder.rewrite_pending)

    def test_reset_invalidates_delayed_end_undo_and_reset(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                old = recorder.generation
                self.end(recorder, mode, {0: "J"}, "j")
                recorder.reset(generation=old)
                self.assertNotEqual(old, recorder.generation)
                self.end(recorder, mode, {0: "U"}, "u")
                with patch.object(
                    recorder,
                    "_observe",
                    side_effect=AssertionError("Stale signal ran CV"),
                ):
                    for signal in ["END", "WRITE_AGAIN", "RESET"]:
                        result = recorder.handle_signal(
                            signal, None, event_id="delayed", generation=old
                        )
                        self.assertEqual(result["status"], "ignored")
                self.assertEqual([w["word"] for w in recorder.words], ["U"])

    def test_failed_disk_write_keeps_memory_unchanged(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "j")
                before = copy.deepcopy(recorder._data)
                for operation in [
                    partial(self.end, recorder, mode, {0: "U"}, "u"),
                    partial(
                        recorder.write_again,
                        event_id="undo",
                        generation=recorder.generation,
                    ),
                    partial(recorder.reset, generation=recorder.generation),
                ]:
                    with (
                        patch.object(
                            recorder, "_persist", side_effect=OSError("disk full")
                        ),
                        self.assertRaises(OSError),
                    ):
                        operation()
                    self.assertEqual(recorder._data, before)
                    self.assertEqual(
                        json.loads(recorder.storage_path.read_text()), before
                    )

    def test_legacy_words_preserved_and_first_end_only_sets_baseline(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "j")
                saved = copy.deepcopy(recorder.words)
                recorder.storage_path.write_text(
                    json.dumps(
                        {"version": 1, "board_id": "simulation-board", "words": saved}
                    )
                )
                factory, _ = self.modes[mode]
                loaded = WordRecorder(
                    self.reference,
                    self.layout,
                    recorder.storage_path,
                    reader_factory=factory,
                )
                self.assertEqual(
                    self.end(loaded, mode, {0: "J"}, "baseline")["status"],
                    "baseline_initialized",
                )
                self.assertEqual(loaded.words, saved)
                self.assertEqual(
                    self.end(loaded, mode, {0: "J", 1: "U"}, "u")["record"]["word"], "U"
                )

    def test_moving_camera_uses_board_coordinates(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                recorder = self.make_recorder(mode)
                self.end(recorder, mode, {0: "J"}, "j")
                frame, _ = compose_frame(
                    self.reference,
                    self.layout,
                    self.modes[mode][1],
                    {0: "J", 1: "U"},
                    camera_index=12,
                )
                result = recorder.end(
                    frame, event_id="moved", generation=recorder.generation
                )
                self.assertEqual(result["record"]["word"], "U")
                self.assertEqual(
                    result["record"]["box_xyxy"], [340.0, 220.0, 419.0, 379.0]
                )

    def test_tree_keeps_two_distinct_e_positions(self):
        recorder = self.make_recorder("six_dot")
        result = self.end(recorder, "six_dot", dict(enumerate("TREE")), "tree")
        self.assertEqual(result["record"]["word"], "TREE")
        letters = result["record"]["letters"]
        self.assertNotEqual(letters[2]["track_id"], letters[3]["track_id"])
