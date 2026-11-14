# Printable A4 LEGO workspace

Print **one** of the one-page PDF files:

- `lego_board_A4_QR.pdf`: four standard QR codes with distinct corner payloads
  (`LEGO-A4:TL`, `LEGO-A4:TR`, `LEGO-A4:BL`, `LEGO-A4:BR`).
- `lego_board_A4_ArUco.pdf`: four `DICT_4X4_100` markers, IDs 0, 1, 2, 3 at
  top-left, top-right, bottom-left, bottom-right. This is the marker format
  supported by the project's existing `BoardAligner`.

Both PDFs contain vector graphics, measure **297 x 210 mm (A4 landscape)**,
and have 36 mm square symbols. A **167 x 124 mm center rectangle** is completely
white; no box or grid is printed in that area. The QR codes have at least four
modules of clear white space around them. Markers and labels stay outside the
central placement region.

## Printing and placement

1. Select **A4**, **landscape**, and **100% / actual size**. Disable fit-to-page
   and shrink-to-printable-area. Print the PDF, not a screenshot or preview.
2. Measure the horizontal scale bar from its left outer edge to its right
   outer edge: it should be **50 mm**. This checks printer scaling.
3. Attach the sheet flat to a rigid board. A matte surface helps avoid glare.
4. Put the bricks in the white center. Keep the corner symbols and their white
   margins clear. Capture all four corners when possible; the current ArUco
   aligner requires at least two consistent known markers.

## Using it with the recognition project

This sheet supplies a physical reference plane. It does not automatically
detect LEGO boundaries, read Braille, or produce Quest world coordinates.
The existing END / WRITE_AGAIN / RESET implementation is unchanged.

The ArUco version matches the detector's dictionary, but has a different
geometry from the old simulated board. Capture an empty physical board and
calibrate the brick slots for this sheet before using `WordRecorder`. Do not
reuse the 960 x 640 simulated reference and its slot coordinates. Arbitrary
placement would require the planned automatic block detector.

The QR version is for QR-capable software; the current `BoardAligner` does not
decode QR codes. Its four payloads identify the corners without opening URLs.

The JSON files describe page dimensions and marker corners in millimeters,
with the page's top-left as origin, +x right and +y down. They are **board
geometry metadata**, not `BrailleReader` slot layouts. Marker dimensions
describe the paper plane. LEGO tops are above it, so camera-angle-dependent
parallax must be handled when estimating brick positions or rectifying faces.

## Included files and validation

- PDFs: use these for printing.
- SVGs: editable vector copies of the same boards.
- `*_reference.png`: PDF renders at 300 dpi.
- `*_preview.png`: smaller previews for viewing.
- `*_tilt_check.png`: synthetic tilted views used for decoding checks.
- `validation.json`: results of reading the PDF renders with OpenCV.

Validation covers marker decoding on digital renders and a synthetic
perspective transformation. It does not measure real printer quality,
headset tracking accuracy, six-dot accuracy, or camera latency.

All four QR codes decode together using OpenCV 4.10's
`QRCodeDetectorAruco` (a QR decoder despite its name), and individually using
`QRCodeDetector`. The latter's full-image multi-code detector missed the
top-left QR in one render; the report records this limitation. All four ArUco
markers were detected together in both tested views with `ArucoDetector`.

To regenerate the sheets from the project root:

```bash
.venv/bin/python printable_boards/a4/generate_boards.py
```

Generation uses the existing OpenCV/NumPy environment plus system tools
Ghostscript (`gs`) and Poppler (`pdftoppm`). No recognition dependencies are
changed. Regeneration does not refresh `validation.json`.
