"""Calibrate/test the user-assigned JUDY / FOX / RAIN color example."""

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark_full_images import stats
from color_reader import DESTINATION, ColorReader, extract_color
from read_braille import write_image
from simulate_recording_dataset import generate
from test_recording_pipeline import replay

ROOT = Path(__file__).resolve().parent
LETTERS = "JUDYFOXRAIN"
COLOR_NAMES = [
    "tan",
    "light gray",
    "blue",
    "brown",
    "red",
    "teal",
    "yellow",
    "lime",
    "dark teal",
    "near-black",
    "orange",
]


def calibrate(image, layout):
    dictionary = {
        "version": 1,
        "scope": "Example-only user-assigned color code, not Braille",
        "source_image": "samples/lego_color.jpg",
        "assignment_order": "top row left-to-right, then bottom row left-to-right",
        "words": ["JUDY", "FOX", "RAIN"],
        "feature": "median CIELAB on the middle-brightness subset of an inset face crop",
        "distance": "Euclidean CIELAB (Delta E 76)",
        "minimum_margin_delta_e76": 2.0,
        "entries": [],
    }
    for i, (letter, name, cell) in enumerate(
        zip(LETTERS, COLOR_NAMES, layout["cells"], strict=True)
    ):
        crop = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(np.float32(cell["quad"]), DESTINATION),
            (80, 160),
            flags=cv2.INTER_CUBIC,
        )
        feature = extract_color(crop)
        rgb = cv2.cvtColor(np.float32([[feature]]), cv2.COLOR_LAB2RGB)[0, 0]
        dictionary["entries"].append(
            {
                "letter": letter,
                "color_name": name,
                "lab": feature.tolist(),
                "swatch_rgb": [round(255 * x) for x in rgb],
                "source_index": i,
            }
        )
    for entry in dictionary["entries"]:
        closest = min(
            np.linalg.norm(np.asarray(entry["lab"]) - other["lab"])
            for other in dictionary["entries"]
            if entry is not other
        )
        entry["nearest_other_distance_delta_e76"] = float(closest)
        entry["acceptance_radius_delta_e76"] = float(min(15, closest * 0.45))
    return dictionary


def static_test(image, layout, dictionary, out, runs):
    variants = [
        ("reference", image),
        ("gaussian_sigma_2", cv2.GaussianBlur(image, (0, 0), 2)),
    ]
    kernel = np.zeros((7, 7), np.float32)
    kernel[3, :] = 1 / 7
    variants += [
        ("motion_7px", cv2.filter2D(image, -1, kernel)),
        (
            "brightness_0.75",
            np.clip(image.astype(float) * 0.75, 0, 255).astype(np.uint8),
        ),
        (
            "brightness_1.15",
            np.clip(image.astype(float) * 1.15, 0, 255).astype(np.uint8),
        ),
        (
            "warm_white_balance",
            np.clip(image.astype(float) * [0.85, 1.0, 1.15], 0, 255).astype(np.uint8),
        ),
    ]
    reader = ColorReader(layout, dictionary)
    results = []
    for name, frame in variants:
        for _ in range(3):
            reader.read(frame)
        times = []
        for _ in range(runs):
            start = perf_counter()
            result, _ = reader.read(frame)
            times.append((perf_counter() - start) * 1000)
        correct = sum(
            c["letter_candidate"] == letter
            for c, letter in zip(result["cells"], LETTERS, strict=True)
        )
        wrong = sum(
            c["letter_candidate"] is not None and c["letter_candidate"] != letter
            for c, letter in zip(result["cells"], LETTERS, strict=True)
        )
        results.append(
            {
                "condition": name,
                "correct": correct,
                "wrong": wrong,
                "unknown": len(LETTERS) - correct - wrong,
                "latency_ms": stats(times),
                "samples_ms": times,
                "cells": result["cells"],
            }
        )
        preview = frame.copy()
        for cell in result["cells"]:
            q = np.int32(cell["face_polygon"])
            cv2.polylines(
                preview,
                [q],
                True,
                (0, 255, 0) if cell["accepted"] else (0, 150, 255),
                2,
            )
            cv2.putText(
                preview,
                cell["letter_candidate"] or "?",
                tuple(q[0] + [0, -10]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        write_image(out / f"{name}_annotated.png", preview)
        if name != "reference":
            write_image(out / f"{name}_input.png", frame)
    report = {
        "reference_accuracy_is_calibration_resubstitution": True,
        "cpu_threads": 4,
        "runs": runs,
        "timing_boundary": "Decoded full image through eleven calibrated face warps, color extraction, dictionary decisions and JSON serialization; no marker alignment for these static tests.",
        "results": results,
    }
    (out / "static_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/color_example"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "datasets/end_only_color"
    )
    parser.add_argument(
        "--dictionary", type=Path, default=ROOT / "color_dictionary.json"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Write color prototypes from the assigned reference photo before testing",
    )
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    cv2.setNumThreads(4)
    image = cv2.imread(str(args.project_root / "samples/lego_color.jpg"))
    layout = json.loads((args.project_root / "layouts/lego_color.json").read_text())
    if args.calibrate:
        args.dictionary.write_text(
            json.dumps(calibrate(image, layout), indent=2) + "\n"
        )
    dictionary = json.loads(args.dictionary.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_report = static_test(
        image, layout, dictionary, args.output_dir, args.runs
    )
    atlas, provenance = {}, {}
    for entry in dictionary["entries"]:
        q = layout["cells"][entry["source_index"]]["quad"]
        atlas[entry["letter"]] = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(np.float32(q), DESTINATION),
            (80, 160),
            flags=cv2.INTER_CUBIC,
        )
        provenance[entry["letter"]] = {
            "image": "lego_color.jpg",
            "quad": q,
            "color_name": entry["color_name"],
        }
    generated = generate(
        args.project_root,
        args.dataset_dir,
        words=["JUDY", "FOX", "RAIN"],
        atlas=atlas,
        provenance=provenance,
        description="One final color-only frame per END, four fixed slots and planar camera motion; no START or intermediate frames.",
        replacement_word="FAIR",
    )
    replay_report = replay(
        args.dataset_dir,
        args.output_dir / "end_only",
        reader_factory=partial(ColorReader, dictionary=dictionary),
        recognition_method="example_color",
    )
    print(
        json.dumps(
            {
                "generated": generated,
                "static": [
                    {k: v for k, v in r.items() if k not in ["cells", "samples_ms"]}
                    for r in reference_report["results"]
                ],
                "recording_checks": replay_report["checks"],
            },
            indent=2,
        )
    )

    if not all(replay_report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
