"""Benchmark every calibrated Braille brick in both bundled full images.

Manual visible-dot annotations below are evaluation data only, never detector input.
Includes unsupported punctuation patterns in dot accuracy and timing.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from braille_reader import BrailleReader, previews
from read_braille import write_image

ROOT = Path(__file__).resolve().parent
# Independently transcribed visible dots in A-Z order, then the symbol bricks.
ALPHABET_DOTS = [
    "1",
    "12",
    "14",
    "145",
    "15",
    "124",
    "1245",
    "125",
    "24",
    "245",
    "13",
    "123",
    "134",
    "1345",
    "135",
    "1234",
    "12345",
    "1235",
    "234",
    "2345",
    "136",
    "1236",
    "2456",
    "1346",
    "13456",
    "1356",
]
CASES = [
    (
        "lego.png",
        "lego_braille.json",
        ["16", *ALPHABET_DOTS[1:], "6", "256", "236", "235", "3", "2", "36", "3456"],
    ),
    ("lego2.jpg", "lego2_braille_full.json", [*ALPHABET_DOTS, "6", "2", "3", "256"]),
]


def stats(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": min(values),
        "max": max(values),
    }


def score(result, expected):
    cells = result["cells"]
    if len(cells) != len(expected):
        raise ValueError("Ground truth must cover every configured brick")
    exact = sum(
        c["raised_dots"] == [int(d) for d in dots] for c, dots in zip(cells, expected)
    )
    bit_correct = sum(
        b is not None and bool(b) == (str(i + 1) in dots)
        for c, dots in zip(cells, expected)
        for i, b in enumerate(c["bits"])
    )
    correct = sum(
        c["letter_candidate"] == letter
        for c, letter in zip(cells[:26], "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    wrong = sum(
        c["letter_candidate"] is not None and c["letter_candidate"] != letter
        for c, letter in zip(cells[:26], "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    return {
        "alphabet_correct": correct,
        "alphabet_total": 26,
        "alphabet_wrong": wrong,
        "alphabet_unknown": 26 - correct - wrong,
        "visible_patterns_correct": exact,
        "visible_patterns_total": len(cells),
        "dot_bits_correct": int(bit_correct),
        "dot_bits_total": 6 * len(cells),
        "uncertain_cells": sum(c["mask"] is None for c in cells),
        "unsupported_patterns": sum(c["kind"] == "unsupported" for c in cells),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/braille_full_images"
    )
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.threads < 1:
        p.error("runs/threads must be positive, warmup nonnegative")
    cv2.setNumThreads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "default_method": "six-dot Braille",
        "reference_image": "samples/lego.png",
        "cpu_threads": args.threads,
        "runs_per_image_per_boundary": args.runs,
        "warmup": args.warmup,
        "boundaries": {
            "decoded_image": "Full decoded frame to returned result, including all calibrated face warps, stud detection, dictionary lookup, coordinates, row assembly and JSON serialization.",
            "encoded_image": "In-memory JPEG/PNG decoding plus the same full-frame analysis; excludes disk reads.",
        },
        "exclusions": [
            "One-time reader setup",
            "Manual calibration / automatic brick localization",
            "Camera acquisition",
            "Output file writes",
            "Unity/network/rendering",
        ],
        "caveats": [
            "Two development photos, not a held-out accuracy dataset.",
            "lego.png A1 visibly contains dots 1 and 6, unlike standard A (dot 1).",
            "Punctuation dot patterns are measured but the alphabet/number decoder does not translate punctuation.",
        ],
        "images": [],
    }
    for name, layout_name, expected in CASES:
        path = ROOT / "samples" / name
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        image = cv2.imdecode(encoded, flags)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        layout = json.loads((ROOT / "layouts" / layout_name).read_text())
        reader = BrailleReader(layout)
        for _ in range(args.warmup):
            reader.read(image)
        decoded_times = []
        encoded_times = []
        for _ in range(args.runs):
            start = perf_counter()
            result, crops = reader.read(image)
            decoded_times.append((perf_counter() - start) * 1000)
            start = perf_counter()
            fresh = cv2.imdecode(encoded, flags)
            reader.read(fresh)
            encoded_times.append((perf_counter() - start) * 1000)
        summary = {
            "image": name,
            "image_size": result["image_size"],
            "layout": layout_name,
            "cells": len(result["cells"]),
            "accuracy": score(result, expected),
            "decoded_image_ms": stats(decoded_times),
            "encoded_image_ms": stats(encoded_times),
            "decoded_samples_ms": decoded_times,
            "encoded_samples_ms": encoded_times,
            "expected_visible_dots": expected,
            "rows": result["rows"],
        }
        report["images"].append(summary)
        (args.output_dir / f"{path.stem}_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        annotation, sheet = previews(image, result, crops)
        write_image(args.output_dir / f"{path.stem}_annotated.png", annotation)
        write_image(args.output_dir / f"{path.stem}_cells.png", sheet)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if k
                    not in (
                        "rows",
                        "expected_visible_dots",
                        "decoded_samples_ms",
                        "encoded_samples_ms",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
