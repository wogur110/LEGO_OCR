"""Reproducible blur and planar camera-motion stress tests; no detector tuning."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark_full_images import CASES, score, stats
from board_alignment import BoardAligner
from braille_reader import BrailleReader, previews
from read_braille import write_image

ROOT = Path(__file__).resolve().parent


def motion_blur(image, length):
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def move_layout(layout, homography):
    moved = copy.deepcopy(layout)
    for cell in moved["cells"]:
        cell["quad"] = cv2.perspectiveTransform(
            np.float32(cell["quad"]).reshape(1, 4, 2), homography
        )[0].tolist()
    return moved


def camera_transform(width, height, index, count):
    phase = 2 * np.pi * index / count
    angle = 5 * np.sin(phase)
    scale = 0.90 + 0.04 * np.cos(phase)
    affine = np.vstack(
        [cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale), [0, 0, 1]]
    )
    affine[0, 2] += 0.025 * width * np.sin(phase + 0.4)
    affine[1, 2] += 0.02 * height * np.cos(phase + 0.7)
    perspective = np.eye(3)
    perspective[2, 0] = 0.035 * np.sin(phase + 0.2) / width
    perspective[2, 1] = 0.025 * np.cos(phase + 0.8) / height
    return perspective @ affine


def compact(result):
    return [
        {
            "row_id": c["row_id"],
            "slot_index": c["slot_index"],
            "bits": c["bits"],
            "raised_dots": c["raised_dots"],
            "letter": c["letter_candidate"],
            "reason": c["reason"],
        }
        for c in result["cells"]
    ]


def evaluate(result, expected):
    scored = score(result, expected)
    scored["wrong_resolved_patterns"] = sum(
        c["raised_dots"] is not None and c["raised_dots"] != [int(d) for d in dots]
        for c, dots in zip(result["cells"], expected, strict=True)
    )
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/braille_robustness"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    if args.runs < 1 or args.frames < 4:
        parser.error("runs must be positive and frames at least four")
    cv2.setNumThreads(4)
    cv2.setRNGSeed(2026)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "cpu_threads": 4,
        "static_measured_runs": args.runs,
        "static_warmups": 2,
        "moving_frames_per_sequence": args.frames,
        "timing_boundary": "Decoded full frame through result JSON serialization. Moving-frame timing includes geometry copying, reader construction, and ORB alignment when selected. Reference ORB initialization excluded. Simulation, evaluation and file I/O excluded.",
        "simulation": "Planar image warp: rotation +/-5 degrees, scale 0.86..0.94, translation +/-2.5% width and +/-2% height, projective coefficients +/-0.035/width and +/-0.025/height. Reflect borders; all evaluated brick corners remain in frame.",
        "limitations": [
            "Synthetic 2D homographies do not simulate 3D parallax, occlusion, changing lighting, or rolling shutter.",
            "Two existing development images; no new held-out real camera data.",
            "Known-transform mode is an oracle for isolating stud detection, not automatic tracking.",
            "ORB mode uses existing BoardAligner; no known transform is given to it.",
            "Frames are read independently; no temporal voting or previous-letter reuse.",
            "lego.png A1 has visible dots 1+6, so alphabet agreement cannot reach 26 without contradicting the dots.",
        ],
        "images": [],
    }
    for name, layout_file, expected in CASES:
        stem = Path(name).stem
        image = cv2.imread(
            str(args.project_root / "samples" / name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        layout = json.loads((args.project_root / "layouts" / layout_file).read_text())
        reader = BrailleReader(layout)
        item = {"image": name, "cells": len(expected), "static": [], "moving": []}
        directory = out / stem
        directory.mkdir(exist_ok=True)
        variants = [("original", image)]
        variants += [
            (f"gaussian_sigma_{sigma}", cv2.GaussianBlur(image, (0, 0), sigma))
            for sigma in [1, 2, 3]
        ]
        variants += [
            (f"motion_{length}px", motion_blur(image, length)) for length in [7, 15]
        ]
        for label, frame in variants:
            for _ in range(2):
                reader.read(frame)
            times = []
            for _ in range(args.runs):
                start = perf_counter()
                result, crops = reader.read(frame)
                times.append((perf_counter() - start) * 1000)
            entry = {
                "condition": label,
                "accuracy": evaluate(result, expected),
                "latency_ms": stats(times),
                "samples_ms": times,
                "cells": compact(result),
            }
            item["static"].append(entry)
            annotation, _ = previews(frame, result, crops)
            write_image(directory / f"{label}_input.png", frame)
            write_image(directory / f"{label}_annotated.png", annotation)
            print(
                stem,
                label,
                entry["accuracy"],
                round(np.mean(times), 1),
                "ms",
                flush=True,
            )
        aligner = BoardAligner(image, layout, method="orb")
        height, width = image.shape[:2]
        for blur_length in [0, 7]:
            sequence_dir = directory / f"moving_blur_{blur_length}"
            sequence_dir.mkdir(exist_ok=True)
            for index in range(args.frames):
                homography = camera_transform(width, height, index, args.frames)
                frame = cv2.warpPerspective(
                    image,
                    homography,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101,
                )
                if blur_length:
                    frame = motion_blur(frame, blur_length)
                # Validate known geometry so cropped-out bricks cannot silently affect accuracy.
                BrailleReader(move_layout(layout, homography))
                write_image(sequence_dir / f"frame_{index:02}.png", frame)
                for mode in [
                    "known_transform",
                    "orb_alignment",
                    "unchanged_coordinates",
                ]:
                    start = perf_counter()
                    alignment = None
                    if mode == "orb_alignment":
                        frame_to_ref, alignment = aligner.align(frame)
                        transform = (
                            np.linalg.inv(frame_to_ref)
                            if frame_to_ref is not None
                            else None
                        )
                    else:
                        transform = (
                            homography if mode == "known_transform" else np.eye(3)
                        )
                    result = None
                    geometry_error = None
                    if transform is not None:
                        try:
                            adjusted = move_layout(layout, transform)
                            result, crops = BrailleReader(adjusted).read(frame)
                        except ValueError as exc:
                            geometry_error = str(exc)
                    elapsed = (perf_counter() - start) * 1000
                    entry = {
                        "blur_pixels": blur_length,
                        "frame": index,
                        "mode": mode,
                        "latency_ms": elapsed,
                        "alignment": alignment,
                        "geometry_error": geometry_error,
                        "ground_truth_reference_to_frame": homography.tolist(),
                        "accuracy": evaluate(result, expected) if result else None,
                        "cells": compact(result) if result else None,
                    }
                    if transform is not None:
                        true = np.float32(
                            [
                                c["quad"]
                                for c in move_layout(layout, homography)["cells"]
                            ]
                        )
                        predicted = np.float32(
                            [c["quad"] for c in move_layout(layout, transform)["cells"]]
                        )
                        entry["mean_corner_error_px"] = float(
                            np.linalg.norm(true - predicted, axis=2).mean()
                        )
                    item["moving"].append(entry)
                    if result is not None and index in [0, args.frames // 2]:
                        annotation, _ = previews(frame, result, crops)
                        write_image(
                            sequence_dir / f"frame_{index:02}_{mode}.png", annotation
                        )
            print(
                stem, "completed moving sequence, motion blur", blur_length, flush=True
            )
        item["moving_summary"] = []
        for blur_length in [0, 7]:
            for mode in ["known_transform", "orb_alignment", "unchanged_coordinates"]:
                entries = [
                    e
                    for e in item["moving"]
                    if e["mode"] == mode and e["blur_pixels"] == blur_length
                ]
                successes = [e for e in entries if e["accuracy"] is not None]
                totals = {
                    key: sum(e["accuracy"][key] for e in successes)
                    for key in [
                        "alphabet_correct",
                        "alphabet_wrong",
                        "alphabet_unknown",
                        "visible_patterns_correct",
                        "wrong_resolved_patterns",
                    ]
                }
                summary = {
                    "blur_pixels": blur_length,
                    "mode": mode,
                    "successful_frames": len(successes),
                    "total_frames": len(entries),
                    "alphabet_total_including_lost_frames": 26 * len(entries),
                    "patterns_total_including_lost_frames": len(expected)
                    * len(entries),
                    "lost_frame_letters": 26 * (len(entries) - len(successes)),
                    **totals,
                    "latency_ms_all_frames": stats([e["latency_ms"] for e in entries]),
                    "latency_ms_successful_frames": stats(
                        [e["latency_ms"] for e in successes]
                    )
                    if successes
                    else None,
                    "mean_corner_error_px": float(
                        np.mean([e["mean_corner_error_px"] for e in successes])
                    )
                    if successes
                    else None,
                }
                item["moving_summary"].append(summary)
                print(stem, json.dumps(summary), flush=True)
        report["images"].append(item)
        (out / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
