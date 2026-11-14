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


def build_parser():
    """Build the command-line interface without parsing or reading any files."""
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
    return parser


def resolve_layout(image_path, layout_path, parser):
    """Resolve bundled geometry only when the input matches its bundled photo."""
    if layout_path is not None:
        return layout_path
    layouts = {
        "lego.png": "lego_braille.json",
        "lego2.jpg": "lego2_braille_full.json",
    }
    bundled = ROOT / "samples" / image_path.name
    if (
        image_path.name not in layouts
        or not bundled.is_file()
        or not image_path.is_file()
        or image_path.read_bytes() != bundled.read_bytes()
    ):
        parser.error(
            "Provide --layout for a new photo; bundled geometry applies only to the bundled images"
        )
    return ROOT / "layouts" / layouts[image_path.name]


def parse_arguments(parser, argv=None):
    args = parser.parse_args(argv)
    args.layout = resolve_layout(args.image, args.layout, parser)
    if args.threads < 1 or args.warmup < 0 or args.runs < 1:
        parser.error("threads/runs must be positive and warmup nonnegative")
    return args


def measure_reader(reader, image, warmup, runs):
    """Return the final recognition and timings, excluding setup and file I/O."""
    for _ in range(warmup):
        reader.read(image)
    timings = []
    for _ in range(runs):
        result, crops = reader.read(image)
        timings.append(result["timing_ms"])
    return result, crops, timings


def build_report(args, layout, result, timings):
    """Combine recognition output with the existing benchmark report schema."""
    values = [t["image_to_result"] for t in timings]
    return {
        "image": str(args.image.resolve()),
        "layout": layout,
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


def write_artifacts(output_dir, image_path, image, result, crops, report):
    """Write the JSON report, previews and per-cell crops; return the report path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem + "_braille"
    report_path = output_dir / f"{stem}.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    annotation, sheet = previews(image, result, crops)
    write_image(output_dir / f"{stem}_annotated.png", annotation)
    write_image(output_dir / f"{stem}_cells.png", sheet)
    crop_dir = output_dir / f"{stem}_crops"
    crop_dir.mkdir(exist_ok=True)
    for cell, crop in zip(result["cells"], crops, strict=True):
        write_image(
            crop_dir / f"{cell['row_id']}_{cell['slot_index'] + 1:02}.png", crop
        )
    return report_path


def print_summary(result, latency, report_path):
    """Print the same row, cell and latency summary as the original CLI."""
    for row in result["rows"]:
        print(row["row_id"] + ": " + row["text"])
    for cell in result["cells"]:
        print(
            f"{cell['row_id']}/{cell['slot_index'] + 1}: dots={cell['raised_dots']} value={cell['value']} letter={cell['letter_candidate']} number={cell['number_candidate']} reason={cell['reason']}"
        )
    print(
        f"Image-to-result: mean {latency['mean']:.3f} ms, median {latency['median']:.3f} ms, p95 {latency['p95']:.3f} ms; {len(result['cells'])} cells"
    )
    print(report_path)


def main(argv=None):
    parser = build_parser()
    args = parse_arguments(parser, argv)
    cv2.setNumThreads(args.threads)
    image = cv2.imread(
        str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    layout = json.loads(args.layout.read_text())
    reader = BrailleReader(layout, args.mode)
    result, crops, timings = measure_reader(reader, image, args.warmup, args.runs)
    report = build_report(args, layout, result, timings)
    report_path = write_artifacts(
        args.output_dir, args.image, image, result, crops, report
    )
    print_summary(result, report["latency_ms"], report_path)


if __name__ == "__main__":
    main()
