"""Measure six-dot quality with no markers or camera alignment.

Fixed calibrated face crops are retained. Transformations below generate test
images only; their true transforms never update the reader's input geometry.
"""

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from benchmark_braille_robustness import compact, evaluate, motion_blur
from benchmark_full_images import CASES, stats
from board_alignment import project
from braille_reader import BrailleReader, previews
from read_braille import write_image
from recording_pipeline import WordRecorder
from simulate_recording_dataset import (
    WORDS,
    atlas_from_photos,
    compose_frame,
    generate,
    make_reference,
)
from test_recording_pipeline import replay

ROOT = Path(__file__).resolve().parent


def conditions(width, height):
    entries = [{"name": "stationary", "transform": np.eye(3), "blur": None}]
    for axis, shift in [(0, 2), (0, 5), (0, 10), (0, 20), (1, 5)]:
        transform = np.eye(3)
        transform[axis, 2] = shift
        entries.append(
            {
                "name": f"shift_{'x' if axis == 0 else 'y'}_{shift}px",
                "transform": transform,
                "blur": None,
            }
        )
    for angle in [1, 3]:
        transform = np.vstack(
            [cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0), [0, 0, 1]]
        )
        entries.append(
            {"name": f"rotate_{angle}deg", "transform": transform, "blur": None}
        )
    for scale in [0.98, 0.95]:
        transform = np.vstack(
            [cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale), [0, 0, 1]]
        )
        entries.append({"name": f"scale_{scale}", "transform": transform, "blur": None})
    entries += [
        {"name": "gaussian_sigma_1", "transform": np.eye(3), "blur": "gaussian1"},
        {"name": "gaussian_sigma_2", "transform": np.eye(3), "blur": "gaussian2"},
        {"name": "motion_blur_7px", "transform": np.eye(3), "blur": "motion7"},
    ]
    mixed = np.vstack(
        [cv2.getRotationMatrix2D((width / 2, height / 2), 2.5, 0.97), [0, 0, 1]]
    )
    mixed[0, 2] += width * 0.015
    mixed[1, 2] += height * 0.014
    entries.append({"name": "combined_motion", "transform": mixed, "blur": None})
    return entries


def perturb(image, condition):
    transform = condition["transform"]
    if np.array_equal(transform, np.eye(3)):
        frame = image.copy()
    else:
        frame = cv2.warpPerspective(
            image,
            transform,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
    if condition["blur"] == "motion7":
        frame = motion_blur(frame, 7)
    elif condition["blur"] in {"gaussian1", "gaussian2"}:
        frame = cv2.GaussianBlur(
            frame, (0, 0), 1 if condition["blur"] == "gaussian1" else 2
        )
    return frame


def geometry_error(layout, transform):
    errors, out_of_frame = [], []
    width, height = layout["image_size"]
    for cell in layout["cells"]:
        assumed = np.float32(cell["quad"])
        true = project(assumed, transform)
        errors.extend(np.linalg.norm(assumed - true, axis=1).tolist())
        if (
            (true < 0).any()
            or (true[:, 0] >= width).any()
            or (true[:, 1] >= height).any()
        ):
            out_of_frame.append([cell["row_id"], cell["slot_index"]])
    return {"corner_error_px": stats(errors), "partly_out_of_frame_slots": out_of_frame}


def test_photos(root, out, runs):
    reports = []
    for name, layout_name, expected in CASES:
        image = cv2.imread(
            str(root / "samples" / name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        layout = json.loads((root / "layouts" / layout_name).read_text())
        reader = BrailleReader(layout)
        for _ in range(3):
            reader.read(image)
        rows = []
        directory = out / Path(name).stem
        directory.mkdir(parents=True, exist_ok=True)
        for condition in conditions(image.shape[1], image.shape[0]):
            frame = perturb(image, condition)
            times = []
            for _ in range(runs):
                started = perf_counter()
                result, crops = reader.read(frame)
                times.append((perf_counter() - started) * 1000)
            row = {
                "condition": condition["name"],
                "accuracy": evaluate(result, expected),
                "latency_ms": stats(times),
                "samples_ms": times,
                "cells": compact(result),
                **geometry_error(layout, condition["transform"]),
                "true_transform_evaluation_only": condition["transform"].tolist(),
            }
            rows.append(row)
            if condition["name"] in {
                "stationary",
                "shift_x_10px",
                "rotate_3deg",
                "gaussian_sigma_2",
                "combined_motion",
            }:
                annotation, _ = previews(frame, result, crops)
                write_image(
                    directory / f"{condition['name']}_annotated.png", annotation
                )
            print(
                name,
                condition["name"],
                row["accuracy"]["alphabet_correct"],
                "/ 26 letters;",
                row["accuracy"]["alphabet_wrong"],
                "wrong;",
                row["accuracy"]["alphabet_unknown"],
                "unknown",
                flush=True,
            )
        reports.append({"image": name, "cell_count": len(expected), "conditions": rows})
    return reports


def test_words(root, out, runs):
    reference, layout = make_reference(with_markers=False)
    atlas, _ = atlas_from_photos(root)
    reports, thumbnails = [], {}
    selected = {
        "stationary",
        "shift_x_10px",
        "rotate_3deg",
        "gaussian_sigma_2",
        "motion_blur_7px",
        "combined_motion",
    }
    with tempfile.TemporaryDirectory(
        prefix="lego-no-alignment-benchmark-"
    ) as directory:
        recorder = WordRecorder(
            reference, layout, Path(directory) / "words.json", alignment_mode="none"
        )
        warmup, _ = compose_frame(reference, layout, atlas, dict(enumerate(WORDS[0])))
        for i in range(3):
            recorder.reset(generation=recorder.generation)
            recorder.end(warmup, event_id=f"warmup-{i}", generation=recorder.generation)
        for condition in conditions(reference.shape[1], reference.shape[0]):
            cases, analysis_times, total_times, errors = [], [], [], []
            for word in WORDS:
                image, _ = compose_frame(
                    reference, layout, atlas, dict(enumerate(word))
                )
                frame = perturb(image, condition)
                for run in range(runs):
                    # Reset only storage; reference setup is outside the timed region.
                    recorder.reset(generation=recorder.generation)
                    result = recorder.end(
                        frame,
                        event_id=f"{condition['name']}-{word}-{run}",
                        generation=recorder.generation,
                    )
                    analysis_times.append(result["analysis_latency_ms"])
                    total_times.append(result["latency_ms"])
                actual = result.get("record", {}).get("word")
                observations = {o["slot"]: o for o in result.get("observations", [])}
                predicted = [
                    observations.get(i, {}).get("letter") for i in range(len(word))
                ]
                correct = sum(
                    letter == target
                    for letter, target in zip(predicted, word, strict=True)
                )
                wrong = sum(
                    letter is not None and letter != target
                    for letter, target in zip(predicted, word, strict=True)
                )
                unknown = len(word) - correct - wrong
                expected_layout = {**layout, "cells": layout["cells"][: len(word)]}
                geometry = geometry_error(expected_layout, condition["transform"])
                errors.append(geometry["corner_error_px"]["mean"])
                cases.append(
                    {
                        "expected": word,
                        "actual_saved": actual,
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "exact_word": actual == word,
                        "wrong_word_saved": actual is not None and actual != word,
                        "letters": {
                            "correct": correct,
                            "wrong": wrong,
                            "unknown": unknown,
                            "total": len(word),
                        },
                        "predicted_at_expected_slots": predicted,
                        "observations": result.get("observations", []),
                        **geometry,
                    }
                )
                if condition["name"] in selected:
                    thumb = np.full((225, 320, 3), 245, np.uint8)
                    thumb[10:190] = cv2.resize(frame, (320, 180))
                    caption = f"{word} -> {actual or 'REVIEW'}"
                    color = (0, 100, 0) if actual == word else (0, 0, 170)
                    cv2.putText(
                        thumb,
                        caption,
                        (10, 215),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    thumbnails.setdefault(condition["name"], []).append(thumb)
            report = {
                "condition": condition["name"],
                "cases": cases,
                "exact_words": sum(c["exact_word"] for c in cases),
                "total_words": len(cases),
                "wrong_words_saved": sum(c["wrong_word_saved"] for c in cases),
                "rejected_words": sum(c["actual_saved"] is None for c in cases),
                "letters": {
                    key: sum(c["letters"][key] for c in cases)
                    for key in ["correct", "wrong", "unknown", "total"]
                },
                "mean_corner_error_px": float(np.mean(errors)),
                "analysis_latency_ms": stats(analysis_times),
                "end_including_storage_latency_ms": stats(total_times),
                "analysis_samples_ms": analysis_times,
                "end_samples_ms": total_times,
                "true_transform_evaluation_only": condition["transform"].tolist(),
            }
            reports.append(report)
            print(
                "words",
                condition["name"],
                report["exact_words"],
                "/ 4 exact;",
                report["wrong_words_saved"],
                "wrong saved;",
                report["rejected_words"],
                "rejected",
                flush=True,
            )
    strips = []
    for label, thumbs in thumbnails.items():
        heading = np.full((38, 1280, 3), 230, np.uint8)
        cv2.putText(
            heading,
            label,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        strips += [heading, np.hstack(thumbs)]
    write_image(out / "word_conditions.png", np.vstack(strips))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/braille_no_alignment"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "datasets/end_only_six_dot_no_alignment",
    )
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    cv2.setNumThreads(4)
    cv2.setRNGSeed(2026)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "recording_pipeline.BoardAligner",
            side_effect=AssertionError("Camera alignment must never run"),
        ) as aligner_guard,
        patch(
            "cv2.aruco.ArucoDetector",
            side_effect=AssertionError("Marker detection must never run"),
        ) as marker_guard,
        patch(
            "cv2.ORB_create",
            side_effect=AssertionError("Feature alignment must never run"),
        ) as orb_guard,
    ):
        generate(
            args.project_root,
            args.dataset_dir,
            with_markers=False,
            camera_motion=False,
            replacement_word="REED",
        )
        signal_report = replay(
            args.dataset_dir, args.output_dir / "signals", alignment_mode="none"
        )
        photo_report = test_photos(args.project_root, args.output_dir, args.runs)
        word_report = test_words(args.project_root, args.output_dir, args.runs)
        proof = {
            "board_aligner_constructions": aligner_guard.call_count,
            "aruco_detector_constructions": marker_guard.call_count,
            "orb_detector_constructions": orb_guard.call_count,
        }
    report = {
        "recognition_method": "six_dot",
        "alignment_mode": "none",
        "runtime_call_guards": proof,
        "cpu_threads": 4,
        "runs_per_condition": args.runs,
        "accuracy_counts_distinct_frames_not_timing_repeats": True,
        "photo_results": photo_report,
        "word_results": word_report,
        "signal_checks": signal_report["checks"],
        "timing_boundary": "Decoded image through fixed face crops, six-dot recognition and serialization. Word analysis additionally includes presence/snapshot comparison. END total includes atomic storage. Camera acquisition, decoding, reference initialization, simulated motion and evaluation are excluded.",
        "limitations": [
            "Fixed manual face calibration remains; per-face crop normalization remains. No camera image registration, feature tracking, marker localization or automatic brick detector runs.",
            "Two development photos and synthetic photo-derived words, not held-out Quest footage.",
            "No transform or expected word is supplied to the reader. True transforms are used only to generate disturbances and evaluate coordinate errors.",
            "Pixel shifts refer to input-image pixels; relative effect depends on each brick's image size. Synthetic warps do not represent 3D parallax or rolling shutter.",
            "The printed A1 in lego.png visibly has dots 1+6; agreement with its printed A label is not the same metric as correct stud reading.",
            "Recognition can be correct while coordinates are already wrong. Wrong confidently accepted letters can also be saved as words.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "signal_checks": report["signal_checks"],
                "guards": proof,
                "report": str(args.output_dir / "report.json"),
            },
            indent=2,
        )
    )
    if not all(signal_report["checks"].values()) or any(proof.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
