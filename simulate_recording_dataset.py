"""Generate photo-derived LEGO words, signals, motion, blur and occlusion.

Ground truth is stored separately from the signal stream; the reader receives
only the empty reference, geometry, final END images, WRITE_AGAIN and RESET.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from braille_geometry import DESTINATION
from read_braille import write_image

ROOT = Path(__file__).resolve().parent
WORDS = ["JUDY", "FOX", "RAIN", "TREE"]


def make_reference(*, with_markers=True):
    image = np.full((640, 960, 3), 80, np.uint8)
    if with_markers:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        for identifier, (x, y) in enumerate(
            [(32, 32), (864, 32), (32, 544), (864, 544)]
        ):
            image[y - 12 : y + 76, x - 12 : x + 76] = 255
            marker = cv2.aruco.generateImageMarker(dictionary, identifier, 64)
            image[y : y + 64, x : x + 64] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    cells = []
    for i in range(4):
        x, y = 230 + 110 * i, 220
        cells.append(
            {
                "row_id": "word",
                "slot_index": i,
                "quad": [[x, y], [x + 79, y], [x + 79, y + 159], [x, y + 159]],
            }
        )
    return image, {"version": 1, "image_size": [960, 640], "cells": cells}


def atlas_from_photos(root):
    atlas, provenance = {}, {}
    for name, layout_name in [
        ("lego.png", "lego_braille.json"),
        ("lego2.jpg", "lego2_braille_full.json"),
    ]:
        image = cv2.imread(str(root / "samples" / name))
        layout = json.loads((root / "layouts" / layout_name).read_text())
        for letter, cell in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ", layout["cells"][:26], strict=True
        ):
            # The first photo has a nonstandard A1 pattern; use standard A from photo two.
            if letter not in atlas or letter == "A":
                transform = cv2.getPerspectiveTransform(
                    np.float32(cell["quad"]), DESTINATION
                )
                atlas[letter] = cv2.warpPerspective(
                    image, transform, (80, 160), flags=cv2.INTER_CUBIC
                )
                provenance[letter] = {"image": name, "quad": cell["quad"]}
    return atlas, provenance


def camera_transform(index):
    angle = 2.5 * np.sin(index * 0.23)
    scale = 0.97 + 0.015 * np.cos(index * 0.17)
    h = np.vstack([cv2.getRotationMatrix2D((480, 320), angle, scale), [0, 0, 1]])
    h[0, 2] += 15 * np.sin(index * 0.19)
    h[1, 2] += 9 * np.cos(index * 0.13)
    perspective = np.eye(3)
    perspective[2, 0] = 0.008 * np.sin(index * 0.1) / 960
    return perspective @ h


def compose_frame(
    reference, layout, atlas, occupied, *, camera_index=None, disturbance=None
):
    image = reference.copy()
    for slot, letter in occupied.items():
        x, y = layout["cells"][slot]["quad"][0]
        image[y : y + 160, x : x + 80] = atlas[letter]
    if disturbance == "occlusion" and occupied:
        slot = max(occupied)
        x, y = layout["cells"][slot]["quad"][0]
        image[y : y + 160, x : x + 80] = (145, 160, 180)
    if disturbance == "tracking_loss":
        image[:130] = 80
        image[510:] = 80
    transform = np.eye(3) if camera_index is None else camera_transform(camera_index)
    image = cv2.warpPerspective(
        image,
        transform,
        (960, 640),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    if disturbance == "blur":
        kernel = np.zeros((7, 7), np.float32)
        kernel[3, :] = 1 / 7
        image = cv2.filter2D(image, -1, kernel)
    elif disturbance == "gaussian_blur":
        image = cv2.GaussianBlur(image, (0, 0), 2)
    return image, transform


def generate(
    root,
    out,
    *,
    words=None,
    atlas=None,
    provenance=None,
    description=None,
    replacement_word=None,
    with_markers=True,
    camera_motion=True,
):
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    reference, layout = make_reference(with_markers=with_markers)
    words = WORDS if words is None else list(words)
    if not words:
        raise ValueError("Provide at least one word")
    replacement_word = words[-1] if replacement_word is None else replacement_word
    if any(not 1 <= len(w) <= 4 for w in [*words, replacement_word]):
        raise ValueError("Each example word needs one to four letters")
    if atlas is None:
        atlas, provenance = atlas_from_photos(root)
    write_image(out / "reference.png", reference)
    (out / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    events, annotations, expected, complete_boards = [], [], [], []

    def end_frame(word, event_id):
        occupied = dict(enumerate(word))
        image, transform = compose_frame(
            reference,
            layout,
            atlas,
            occupied,
            camera_index=7 + 17 * len(annotations) if camera_motion else None,
        )
        path = f"frames/{len(annotations):04}.png"
        write_image(out / path, image)
        event = {"op": "end", "event_id": event_id, "generation": 0, "image": path}
        events.append(event)
        annotations.append(
            {
                "event_index": len(events) - 1,
                "image": path,
                "occupied_slots": occupied,
                "reference_to_frame": transform.tolist(),
            }
        )
        if word:
            expected.append(
                {
                    "event_index": len(events) - 1,
                    "word": word,
                    "slots": list(range(len(word))),
                    "box_xyxy": [230, 220, 230 + 110 * (len(word) - 1) + 79, 379],
                }
            )
            complete_boards.append(path)
        return event

    for index, word in enumerate(words):
        event = end_frame(word, f"word-{index}")
        events.append(dict(event))  # Transport retry must use zero additional CV calls.
    events.append({"op": "snapshot", "name": "words_before_rewrite"})
    undo = {"op": "write_again", "event_id": "undo-latest", "generation": 0}
    events.extend([undo, dict(undo)])  # No image accompanies either signal.
    events.append({"op": "snapshot", "name": "words_after_write_again"})
    event = end_frame(replacement_word, "replacement")
    events.append(dict(event))
    events.append({"op": "snapshot", "name": "words_before_reset"})
    end_frame("", "board-cleared")  # Observe dismantling without deleting word history.
    events.append({"op": "snapshot", "name": "words_after_board_clear"})
    events.append({"op": "reset", "generation": 0})
    events.append({**event, "event_id": "delayed-old-end"})
    events.append({"op": "reset", "generation": 0})  # Duplicate stale RESET.
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (out / "ground_truth.json").write_text(
        json.dumps(
            {
                "protocol": "end_only_v2",
                "expected_commits": expected,
                "frames": annotations,
                "complete_word_frames": complete_boards,
                "source_cells": provenance,
                "words": words,
                "replacement_word": replacement_word,
                "final_words": [*words[:-1], replacement_word],
                "markers_in_scene": with_markers,
                "camera_motion": camera_motion,
                "simulation": description
                or (
                    "One final photo-derived frame per END, four fixed slots; "
                    + ("planar camera motion" if camera_motion else "stationary camera")
                    + (", ArUco markers" if with_markers else ", no markers")
                    + "; no START or intermediate frames."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "frames": len(annotations),
        "events": len(events),
        "words": words,
        "replacement_word": replacement_word,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--without-markers", action="store_true")
    p.add_argument("--stationary", action="store_true")
    args = p.parse_args()
    output = args.output_dir or ROOT / "datasets" / (
        "end_only_six_dot_no_alignment" if args.without_markers else "end_only_six_dot"
    )
    print(
        json.dumps(
            generate(
                args.project_root,
                output,
                replacement_word="REED",
                with_markers=not args.without_markers,
                camera_motion=not args.stationary,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
