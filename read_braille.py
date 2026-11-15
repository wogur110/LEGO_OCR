"""Read calibrated LEGO faces using six-dot Braille or printed-label OCR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from braille_reader import BrailleReader, previews

ROOT = Path(__file__).resolve().parent


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def build_parser():
    """Build the command-line interface without parsing or reading any files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path, default=ROOT / "samples/lego_final.jpg"
    )
    parser.add_argument(
        "--layout",
        type=Path,
        help="Calibrated geometry; bundled photos select their layout automatically",
    )
    parser.add_argument(
        "--method", choices=["braille", "ocr"], default="braille",
        help="Recognition method: six-dot Braille (default) or printed-label OCR",
    )
    parser.add_argument(
        "--mode", choices=["letters"], default="letters",
        help="Recognize A-Z and period only (default: letters)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: outputs/braille or outputs/ocr for the selected method)",
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser


def resolve_layout(image_path, layout_path, parser):
    """Resolve bundled geometry only when the input matches its bundled photo."""
    if layout_path is not None:
        return layout_path
    layouts = {
        "lego_final.jpg": "lego_final_braille.json",
        "lego_quest_1.png": "lego_quest_1_braille.json",
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
    selected = ROOT / "layouts" / layouts[image_path.name]
    calibration = json.loads(selected.read_text())
    expected_hash = calibration.get("source_image_sha256")
    if not expected_hash or hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
        parser.error(
            "Photo changed since calibration; provide --layout with updated face corners"
        )
    return selected


def parse_arguments(parser, argv=None):
    args = parser.parse_args(argv)
    args.layout = resolve_layout(args.image, args.layout, parser)
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / args.method
    return args


def write_artifacts(output_dir, image_path, image, result, crops, report):
    """Write the JSON report, previews and per-cell crops; return the report path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem + "_" + result.get("method", "braille")
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


def main(argv=None):
    """Recognize one decoded frame once, then export its result and previews."""
    parser = build_parser()
    args = parse_arguments(parser, argv)
    cv2.setNumThreads(args.threads)
    image = cv2.imread(
        str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    layout = json.loads(args.layout.read_text())
    if args.method == "ocr":
        try:
            from ocr_reader import CalibratedOCRReader

            reader = CalibratedOCRReader(layout, threads=args.threads)
        except ModuleNotFoundError as error:
            if error.name not in {"rapidocr_onnxruntime", "onnxruntime"}:
                raise
            parser.error(
                "OCR needs optional dependencies. In an OCR environment, install them with: "
                "python -m pip install -r requirements-ocr.txt"
            )
        timing_boundary = (
            "One decoded image through label rectification, OCR, letter normalization, "
            "period recovery, row assembly and JSON serialization; "
            "excludes initialization and file I/O"
        )
    else:
        reader = BrailleReader(layout, mode=args.mode)
        timing_boundary = (
            "One decoded image through rectification, stud detection, exact lookup, "
            "row assembly and JSON serialization; excludes initialization and file I/O"
        )
    result, crops = reader.read(image)
    report = {
        "image": str(args.image.resolve()),
        "layout": layout,
        "method": args.method,
        "cpu_threads": args.threads,
        "timing_boundary": timing_boundary,
        **result,
    }
    report_path = write_artifacts(args.output_dir, args.image, image, result, crops, report)
    for row in result["rows"]:
        print(f"{row['row_id']}: {row['text']}")
    print(f"Image-to-result: {result['timing_ms']['image_to_result']:.3f} ms; {len(result['cells'])} cells")
    print(report_path)
    return report


if __name__ == "__main__":
    main()
