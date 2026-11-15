# LEGO six-dot and calibrated OCR recognition

Recognize LEGO letters using raised Braille studs (the default) or printed-label
OCR in calibrated brick regions. Both methods output A-Z and period only and
run locally without a word vocabulary or cloud service. The six-dot method uses
OpenCV and NumPy; the optional OCR method uses PP-OCRv4 through RapidOCR/ONNX.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
.venv/bin/python main.py samples/lego_quest_1.png --output-dir outputs/quest_current
```

Each command processes **one image once**, then saves a JSON result, annotated
image, crop sheet and individual crops. Default input: `samples/lego_final.jpg`.
Default output folder: `outputs/braille/`. Options: `--layout`, `--mode`
(`letters` only), `--method` (`braille` or `ocr`), `--output-dir`, and `--threads`.
The mode argument is retained
for compatibility; numeric and automatic modes are not supported.

For calibrated OCR, create a separate optional environment. `requirements-ocr.txt`
is a complete alternative to `requirements.txt`; keeping the environments
separate avoids installing two overlapping OpenCV distributions:

```bash
python3 -m venv .venv-ocr
.venv-ocr/bin/python -m pip install -r requirements-ocr.txt
.venv-ocr/bin/python main.py --method ocr
.venv-ocr/bin/python main.py samples/lego_quest_1.png --method ocr
```

Both methods save JSON, an annotated image, a crop sheet and per-cell crops.
OCR defaults to `outputs/ocr/` with filenames ending in `_ocr`; six-dot outputs
retain `_braille`. Selecting OCR does not change the default method.

Bundled calibration is available only for the two current photos:

- `lego_final.jpg`: 48 bricks, 4000×3000 encoded pixels. This replaces the old
  16-brick image, which required different calibration.
- `lego_quest_1.png`: five bricks, 648×256 pixels.

Calibration stores a photo hash, so replacing a bundled image cannot silently
reuse its old corners. For a new photo, supply `--layout` with updated face
corners. JPEG EXIF rotation is ignored: output positions use encoded input
pixels, with origin at the top-left.

## Calibrated OCR

`ocr_reader.CalibratedOCRReader(layout).read(image)` returns `(result, crops)`.
Initialize the reader once and reuse it across frames to avoid repeated model
startup. Its constructor accepts `threads=4`; the CLI also applies `--threads`
to OpenCV. An optional `engine` argument allows explicitly sharing an existing
RapidOCR engine when creating a reader for a different layout.

1. Project the lower printed-label area of each calibrated full brick face into
   a 76x49 crop. No automatic text detector or stud detector is run.
2. Recognize crops in batches of six using PP-OCRv4, with minimum recognition
   shape 3x48x96. Accept OCR scores of at least 0.5.
3. Apply the letter-label rules: `0 -> O`, `JO -> J`, and leading `1 -> I`.
   Remove one numeric suffix (`A1 -> A`, `J0 -> J`, `19 -> I`). Other unsupported
   labels remain `?`; no expected word or physical dot pattern is consulted.
4. For unread, low-score or period-like OCR, check the crop for an isolated
   filled printed dot. Period recovery records its source and keeps the raw OCR
   evidence; it does not invent a confidence score.
5. Assemble rows by calibrated slot order, retaining every unread cell as `?`.

OCR cells include `value`, `accepted`, `reason`, `face_polygon`, `label_polygon`,
`raw_ocr_text`, `raw_ocr_score`, `score`, `source` and period diagnostics. They
contain no inferred Braille bits. Both readers share `rows`, `cells`,
`algorithm_id`, `decoder_id`, `image_size` and `timing_ms` output fields.
Model initialization and region preparation occur before timed image processing.
Capture, image decoding, transport and preview/file writing are also excluded.

The OCR option requires calibrated regions. Automatic OCR experiments remain
under `outputs/ocr_uncalibrated_20261114/`; this option does not use that path.
Dot-like dirt can resemble a period, and a period confidently misread as a valid
letter bypasses the fallback. These checks do not establish unseen-scene accuracy.

## Detector and dictionary

The baseline detector is **`six_dot_radial_v3`**:

1. Rectify each calibrated full face to a standard grid, using the upper three
   stud rows and excluding the nominal printed-label quarter.
2. Propose circles and check local contrast, edge coverage and radial shape.
3. Recover unconfirmed circles using smoothed and bounded contrast-normalized
   proposal views. Verify them against the original pixels and gradients.
4. Suppress surface texture using a per-face gradient floor: the larger of 15
   and twice the median coarse-gradient magnitude. When this exceeds 15, at
   least half of the rim must have supporting radial edges above that floor.
5. Return six present/absent/uncertain decisions and perform exact lookup.
   Uncertain patterns are not corrected using expected letters or words.

The decoder is **`lego_english_a16_period_v4`**. `braille_code.py` is the runtime
source; `braille_dictionary.json` is its matching export. Dot numbering:

```text
1 4
2 5
3 6
```

The dictionary contains **A–Z and `.` (dots 256) only**. Standard A uses
**dot 1**; the project additionally accepts **dots 1+6 as A**. Shared A–J
patterns always decode as letters. There is no digit interpretation or numeric
context.

Number (3456), capital (6), and grade-1 (56) indicators are unsupported, as are
all punctuation patterns except period. An unsupported or unread cell is
reported as `?` with diagnostics, and its row is incomplete. `?` is a diagnostic
placeholder, not a recognized punctuation character. Full UEB translation and
capitalization are outside scope.
Sources: [UEB Manual](https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf)
and [AFB cell diagrams](https://www.afb.org/diagrams-braille-cells).

## Recognition output

`BrailleReader(layout, mode="letters").read(image)` returns `(result, crops)`.
The result contains:

- `rows`: text, token interpretations and completeness for each calibrated row.
- `cells`: six dot decisions, detected pattern, letter/period candidates,
  acceptance status, diagnostics, original-image face polygons and dot-center
  positions. Numeric candidate and numeric-context fields are not emitted.
- `algorithm_id`, `decoder_id`, and `timing_ms`.

The timing starts with the decoded image and includes every calibrated face,
lookup, row assembly and JSON serialization. Initialization, capture, file I/O,
transport and rendering are outside this boundary.

## Unity signal/storage integration

`recording_pipeline.WordRecorder` retains the existing protocol:

| Signal | Effect |
|---|---|
| END + frame | Analyze one final frame; save newly occupied/changed letters and their box. |
| WRITE_AGAIN | Remove the latest saved word without analyzing a frame. |
| Next END | Read and save its replacement. |
| RESET | Clear stored words and board state. |

No START signal or continuous frame history is required. Repeated event IDs
are ignored. Include the recorder's current generation with queued signals to
reject messages from before RESET. Unity must serialize signals and associate
each END with its final frame. No live Unity transport is implemented here.

```python
from recording_pipeline import WordRecorder

recorder = WordRecorder(empty_reference, layout, "words.json", alignment_mode="none")
result = recorder.handle_signal(
    "END", frame, event_id="capture-1", generation=recorder.generation
)
```

Presence compares each calibrated crop to the empty reference: mean absolute
BGR difference above 25 means occupied; at most 15 means empty; the intermediate
band is uncertain. An uncertain presence or unreadable occupied cell causes
`needs_review` and saves no partial word. A confidently wrong letter can still
pass. Unsupported punctuation and indicator bricks cannot be saved as letters.
A recognized period is preserved in the word and its positions.

Saved records include the word, individual letters, IDs, polygons and boxes.
Only the word string is placed in `chatgpt_payload`; no external request is sent.
Recognition and storage accept only A-Z and period. Existing compatible saved
words still load; stored text containing other characters is rejected without
rewriting the file. Numeric coordinates and IDs are unchanged.
WRITE_AGAIN and RESET send no word. A downstream consumer needs its own correction
mechanism for a previously received wrong word.

Saved history is separate from current board occupancy. An identical letter
replaced in the same slot is indistinguishable in one frame; send END on the
cleared board between independent rebuilds. WRITE_AGAIN releases the latest
word's slots. Persisted boxes do not move when bricks disappear.

`alignment_mode="none"` uses fixed image coordinates; default ArUco alignment
maps the frame to the reference board. `board_alignment.py` also contains ORB
alignment support. Use separate storage when changing calibration or alignment.
These positions are pixels, not Unity world coordinates.

## Validation and limits

The current letters-only v4 decoder was checked on both original photos:
**48/48** characters for `lego_final.jpg` and **5/5** for `lego_quest_1.png`,
with all **318/318** dot decisions matching the existing independent manual
references. The CLI exported results and previews successfully. These checks
use the calibrated face corners; they do not establish accuracy on new views.
Results are under `outputs/letters_only_20261114/`.

The calibrated OCR main option was checked on the same two photos: **47/48**
and **5/5**, including all three periods, with no wrong accepted characters.
One O in `lego_final.jpg` stays unread. Both CLI methods produced JSON, previews
and crops; integration results are under `outputs/main_ocr_integration_20261114/`.

The historical v3 comparison used only the two selected source photos, with
identical manual geometry and decoder `lego_english_a16_period_v3` for both
detector versions. These measurements predate the letters-only v4 decoder and
are retained as historical detector results:

| Photo | Detector v2 | Detector v3 |
|---|---:|---:|
| `lego_final.jpg` | 45/48 | 48/48 |
| `lego_quest_1.png` | 5/5 | 5/5 |

In that comparison, all 318 original-photo dot decisions matched the manual
physical-stud annotations. Across 14 derived blur/brightness/contrast/resampling
variants, character accuracy improved from 352/371 (94.9%) to 366/371 (98.7%).
Wrong accepted characters fell from six to one; four cells remained unreadable.

The comparison, manual references, timings, annotations and crops are under
`outputs/selected_photos_v3/`. References are evaluation-only and never supplied
to the detector. Historical reports refer to their own source photos and versions.
Mean image-to-result time in that paired 30-run comparison was **388 ms** for
48 bricks and **29 ms** for five bricks (previous v2: 318 ms and 18 ms). These
are local CPU measurements with four OpenCV threads, excluding initialization
and file I/O. Recovery improves accuracy at an added processing cost.

These two development photos and their variants do not establish unseen Quest
accuracy. Strong blur, glare and surface texture can still cause errors.

Brick positions still require manual full-face calibration. This runtime does
not automatically locate freely placed bricks or track arbitrary camera motion.

## Maintained code

| File | Purpose |
|---|---|
| `main.py`, `read_braille.py` | Single-image command and output files. |
| `braille_reader.py` | Rectification, detection and result assembly. |
| `ocr_reader.py` | Calibrated printed-label crops, local OCR and result assembly. |
| `ocr_label_rules.py`, `period_detector.py` | Letter-only OCR normalization and printed-period recovery. |
| `stud_detector.py` | Six-dot detection and texture rejection. |
| `braille_code.py` | Exact dictionary and sequence decoding. |
| `braille_geometry.py` | Layout validation and coordinate transforms. |
| `braille_visualization.py` | Result previews. |
| `recording_pipeline.py`, `board_alignment.py` | Signal handling, storage and optional alignment. |

Test suites, benchmark scripts, simulation/replay scripts and color-example
Python modules were removed after validation. Existing photos, calibration/data
files, historical reports and printable-board assets are retained. The board
sheet generator is a separate retained utility, not a recognition dependency.
