"""Recognize calibrated LEGO letters with parallel OCR and six-dot fallback."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from combined_reader import CombinedReader

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, default=ROOT / 'samples/image.png')
    parser.add_argument('--layout', type=Path, help='Full-face corners for this image')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/recognition.json')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--ocr-confidence', type=float, default=0.85)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not 0 <= args.ocr_confidence <= 1:
        parser.error('--ocr-confidence must be between 0 and 1')
    if not args.image.is_file():
        parser.error(f'Image does not exist: {args.image}')
    if args.layout is None:
        args.layout = ROOT / 'layouts/image_braille.json'
        layout = json.loads(args.layout.read_text())
        if hashlib.sha256(args.image.read_bytes()).hexdigest() != layout.get('source_image_sha256'):
            parser.error('Provide --layout with updated face corners for a new image')
    else:
        layout = json.loads(args.layout.read_text())
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        parser.error(f'Cannot decode image: {args.image}')
    cv2.setNumThreads(args.threads)
    try:
        reader = CombinedReader(layout, threads=args.threads, ocr_confidence=args.ocr_confidence)
    except ModuleNotFoundError as error:
        if error.name not in {'rapidocr_onnxruntime', 'onnxruntime'}:
            raise
        parser.error('Install the combined dependencies with: python -m pip install -r requirements.txt')
    with reader:
        result = reader.read(image)
        report = {'image': str(args.image.resolve()), 'layout': str(args.layout.resolve()), **result}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for row in result['rows']:
            print(f"{row['row_id']}: {row['text']}")
        print(f"Image-to-result: {result['timing_ms']['image_to_result']:.3f} ms")
        print(args.output, flush=True)
    return report


if __name__ == '__main__':
    main()
