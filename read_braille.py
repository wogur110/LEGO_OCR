"""Read six-dot LEGO Braille patterns from a photo with calibrated top-face corners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from braille_reader import BrailleReader, previews

ROOT = Path(__file__).resolve().parent


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path, default=ROOT / "samples/lego.png"
    )
    parser.add_argument(
        "--layout",
        type=Path,
        help="Calibrated geometry; bundled photos select their layout automatically",
    )
    parser.add_argument(
        "--mode", choices=["letters", "numbers", "auto"], default="letters"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/braille")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args(argv)
    if args.layout is None:
        layouts = {
            "lego.png": "lego_braille.json",
            "lego2.jpg": "lego2_braille_full.json",
        }
        bundled = ROOT / "samples" / args.image.name
        if (
            args.image.name not in layouts
            or not bundled.is_file()
            or not args.image.is_file()
            or args.image.read_bytes() != bundled.read_bytes()
        ):
            parser.error(
                "Provide --layout for a new photo; bundled geometry applies only to the bundled images"
            )
        args.layout = ROOT / "layouts" / layouts[args.image.name]
    if args.threads < 1 or args.warmup < 0 or args.runs < 1:
        parser.error("threads/runs must be positive and warmup nonnegative")
    cv2.setNumThreads(args.threads)
    image = cv2.imread(
        str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    layout = json.loads(args.layout.read_text())
    reader = BrailleReader(layout, args.mode)
    for _ in range(args.warmup):
        reader.read(image)
    timings = []
    for _ in range(args.runs):
        result, crops = reader.read(image)
        timings.append(result["timing_ms"])
    values = [t["image_to_result"] for t in timings]
    report = {
        "image": str(args.image.resolve()),
        "layout": layout,
        "algorithm": "calibrated six-dot stud detection; no OCR",
        "cpu_threads": args.threads,
        "warmup_calls": args.warmup,
        "measured_calls": args.runs,
        "timing_boundary": "Decoded image through layout validation, rectification, circle detection, dot assignment, dictionary lookup, row assembly and JSON payload serialization; excludes image/file I/O and one-time reader setup",
        "timings": timings,
        "latency_ms": {
            "mean": mean(values),
            "median": median(values),
            "p95": float(np.percentile(values, 95)),
            "min": min(values),
            "max": max(values),
        },
        **result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem + "_braille"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    annotation, sheet = previews(image, result, crops)
    write_image(args.output_dir / f"{stem}_annotated.png", annotation)
    write_image(args.output_dir / f"{stem}_cells.png", sheet)
    crop_dir = args.output_dir / f"{stem}_crops"
    crop_dir.mkdir(exist_ok=True)
    for cell, crop in zip(result["cells"], crops, strict=True):
        write_image(
            crop_dir / f"{cell['row_id']}_{cell['slot_index'] + 1:02}.png", crop
        )
    for row in result["rows"]:
        print(row["row_id"] + ": " + row["text"])
    for cell in result["cells"]:
        print(
            f"{cell['row_id']}/{cell['slot_index'] + 1}: dots={cell['raised_dots']} letter={cell['letter_candidate']} number={cell['number_candidate']} reason={cell['reason']}"
        )
    print(
        f"Image-to-result: mean {mean(values):.3f} ms, median {median(values):.3f} ms, p95 {np.percentile(values, 95):.3f} ms; {len(result['cells'])} cells"
    )
    print(args.output_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
