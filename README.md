# LEGO letter recognition

The final reader runs printed-label OCR and six-dot recognition in parallel.
For each calibrated brick, it accepts OCR at a score of at least **0.85**;
otherwise it uses the six-dot result. When every label passes OCR, `read()`
returns immediately without waiting for the six-dot worker. Unknown cells stay
`?`, retain their positions, and set `needs_review: true`.

OCR has priority even if the two readers disagree. Its score is a model score,
not a calibrated probability, so a confident OCR mistake can still be accepted.
The printed-period shape check can also accept `.` without an OCR score.

## Run

The existing OCR environment has all required dependencies:

```bash
.venv-ocr/bin/python main.py
```

For a fresh Python 3.11 or 3.12 environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

The default input is `samples/image.png` (346 × 198), with
`layouts/image_braille.json`. Output is `outputs/recognition.json`.
Use an explicit layout for another image:

```bash
.venv-ocr/bin/python main.py capture.png --layout capture_layout.json --output result.json
```

`--ocr-confidence` changes the acceptance threshold; `--threads` defaults to 4.
Both methods still require calibrated full-face corners for each brick. The
default layout is valid for the supplied image only. A moving camera or moved
bricks require updated corners; the reader does not detect or track brick faces.
The CLI refuses to reuse the default calibration on a different image unless
`--layout` is supplied explicitly.

## Python integration

Initialize once and reuse the reader for frames sharing the same calibration:

```python
import json
from pathlib import Path

import cv2
from combined_reader import CombinedReader

cv2.setNumThreads(4)
layout = json.loads(Path("layouts/image_braille.json").read_text())
frame = cv2.imread("samples/image.png", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

with CombinedReader(layout, threads=4, ocr_confidence=0.85) as reader:
    result = reader.read(frame)  # decoded uint8 BGR image -> dictionary
    words = [row["text"] for row in result["rows"]]
    # Reuse reader.read(next_frame) while these face corners remain valid.
```

Each row has `text`, `complete`, and `bbox`. Each cell has `value`, `accepted`,
`source` (`ocr`, `braille`, or `unknown`), `bbox`, `center`, and `face_polygon`,
plus recognition diagnostics. Coordinates are input-image pixels; boxes are
`[left, top, right, bottom]`. Unused six-dot diagnostics are `null` on the early
OCR return path. `backend_errors` reports failures of either reader.

Calls on one reader are serialized. The frame is copied before background work
starts. After an early return, unused six-dot work is canceled between cells;
the next call or `close()` drains it before reusing the workers. This bounds
background work and prevents results from different frames from mixing.

## Recognition components

- `main.py`: the combined command-line entry point.
- `combined_reader.py`: parallel execution, OCR priority, per-cell fallback,
  positions, and the final result.
- `ocr_reader.py`, `ocr_label_rules.py`, `period_detector.py`: rectified printed
  labels, local RapidOCR recognition, label normalization, and printed periods.
- `braille_reader.py`, `stud_detector.py`, `stud_quality.py`: rectified stud
  regions, circle/rim verification, grid fitting, and uncertain-dot rejection.
- `braille_geometry.py`: layout validation and perspective transforms.
- `braille_code.py`, `braille_dictionary.json`: A–Z and period mappings.

Only letters and `.` are recognized. OCR treats a leading printed `1` as `I`
and ignores one trailing label digit. Standard English six-dot letters are
used, with the project's extra `A` mapping for dots **1,6**. Period is **2,5,6**.
Numbers and other punctuation are not decoded. No expected words, reference
answers, or language-model corrections are used at inference time.

## Validation

Against the existing manually annotated reference photos, the combined reader
returned 56/56 correct cells: 3/3 on `image.png`, 48/48 on `lego_final.jpg`, and
5/5 on `lego_quest_1.png`. These are checks on known examples, not an estimate
of accuracy on unseen scenes. Covering one or all printed labels in the default
image still returned `ANT` using six-dot fallback. Strong Gaussian blur
(sigma 3) returned `???` with no accepted wrong letters.

Warm local CPU measurements on the default three-brick image, 30 runs after
5 warmups, with 4 OpenCV and 4 ONNX intra-op threads:

| Input | Mean | 95th percentile |
| --- | ---: | ---: |
| Original; all OCR accepted | 15.6 ms | 19.4 ms |
| One printed label covered | 23.8 ms | 29.4 ms |
| All printed labels covered | 21.9 ms | 29.4 ms |

Measured on an Intel Xeon Gold 5320, not on the Quest headset. Timing includes
the decoded-frame copy, any previous-worker cleanup, recognition, fusion, and
JSON validation. It excludes model initialization, camera capture, image-file
decoding, file output, and network transfer. Larger brick counts take longer.

The project contains only recognition code, its dependencies, and the default
sample/calibration. Previous signal/storage code, simulations, board utilities,
other samples, and historical reports are archived outside this directory.
