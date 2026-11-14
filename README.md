# LEGO six-dot Braille analysis

Recognize raised stud patterns with OpenCV and an English six-dot Braille
decoder with a project-specific A alias. The only runtime dependencies are OpenCV
and NumPy. No model downloads or cloud services are required. An isolated color-based example
for `samples/lego_color.jpg` is documented below; six-dot reading remains the default.

## Base algorithm

The current baseline is **`six_dot_radial_v1`**: calibrated face rectification,
circle proposals, grayscale radial-edge verification and conditional recovery
with stronger smoothing. Exact dictionary lookup uses the separate decoder
**`lego_english_a16_punctuation_v2`**, which accepts dots 1+6 as an additional A pattern
and basic punctuation, including dots 2+5+6 as a period.

`main.py`, `BrailleReader`, the six-dot benchmarks and the default `WordRecorder`
all use this detector automatically. No algorithm-selection flag is needed.
Reader results and new command-line reports identify the detector with
`"algorithm_id": "six_dot_radial_v1"` and the decoder with
`"decoder_id": "lego_english_a16_punctuation_v2"`. Detector thresholds are unchanged
by the decoder update. Future changes should be compared against this baseline
and receive a new identifier for the affected component; behavior-preserving
refactors retain the identifiers. Texture and decoder integration tests guard
their behavior.

`outputs/lego/baseline/` and `outputs/lego_final/baseline/` contain current results
for this detector and decoder, including identifiers, timing measurements,
previews and manual-reference evaluations. Earlier `lego_final` runs are archived
under
`outputs/lego_final/historical/`; their measurements and source hashes retain
their original meanings.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py samples/lego.png --output-dir outputs/lego/baseline
.venv/bin/python main.py samples/lego_final.jpg --layout layouts/lego_final_braille.json --output-dir outputs/lego_final/baseline
.venv/bin/python main.py samples/lego2.jpg
.venv/bin/python main.py samples/lego.png --mode numbers
.venv/bin/python benchmark_full_images.py --runs 50
.venv/bin/python benchmark_braille_robustness.py
.venv/bin/python -m unittest discover -s tests -v
```

Default input: `samples/lego.png` (there is no `lego.jpg` in this project).
Bundled geometry covers all 34 bricks in this photo and 30 in `lego2.jpg`.
Use `--layout path.json` for another calibrated image. Built-in layout selection
checks that the input matches the bundled photo. `--mode` accepts `letters`
(default), `numbers`, or `auto` for number-indicator context.

## Dictionary

`braille_code.py` is the runtime dictionary; `braille_dictionary.json` is its
human-readable export, checked for equality by a test. Dot numbering from the
reading side is:

```text
1 4
2 5
3 6
```

A–J share their standard patterns with 1–9,0. A numeric indicator (dots 3456) or
explicit numeric mode distinguishes numbers from letters. Standard A (dot 1)
and numeric 1 (dot 1 in numeric context) remain unchanged. The project also
accepts **dots 1 and 6 as A** when all six dot decisions are certain. This is an
explicit application alias, not a standard English Braille mapping. The alias
does not create a numeric 1 candidate: dots 1+6 remain unknown in explicit
`numbers` mode.

The dictionary covers A–Z, digits, number/capital/grade-1 indicators and the
period/decimal point (dots 2+5+6), question mark (236), exclamation mark (235),
apostrophe (3), comma (2), and hyphen (36). The last row of `lego.png` is stored
in the export in order: `?`, `!`, apostrophe, comma, hyphen, `#`. The last brick
has dots 3456 and remains a number indicator, displayed as `#` in letters mode.
Period/comma retain automatic numeric context; the other punctuation ends it.
Contractions and complete capitalization rules are outside this subset;
unsupported patterns remain unknown. A recognized question mark has
`kind=punctuation`, `accepted=true`, and a complete row when the other cells
are recognized; an unread cell also displays `?` but remains incomplete. The dictionary
export identifies standard entries separately from the project A alias.
Sources: [UEB Manual](https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf)
and [AFB cell diagrams](https://www.afb.org/diagrams-braille-cells).

## Pipeline and outputs

`braille_reader.py` rectifies each calibrated brick face, samples only the upper
three stud rows, detects circles, assigns six dot decisions and performs exact
dictionary lookup. The nominal printed-label quarter is excluded; calibration
and camera angle must keep printed marks outside the actual stud crop. Results
contain dot bits, letter/number/punctuation candidates, original-image
coordinates and rejection reasons.

Circle candidates must pass contrast, edge coverage and grayscale radial-edge
verification. At each circle rim, edge directions should follow the radius;
straight surface ridges have nearly fixed edge directions. Verification uses
48 angles and a radius tolerance of two pixels at two smoothing scales. A strong
radial-support score of at least 0.40 or a weighted radial-alignment score of at
least 0.90 is required. These are heuristic diagnostics, not probabilities.
If an otherwise qualifying strict candidate fails this check, stronger smoothing
generates replacement circle candidates for its dot slot. All recovered circles
must pass the same checks. Detection uses no expected words or color-to-letter
mapping.

The source is organized by responsibility:

| Module | Responsibility |
|---|---|
| `braille_geometry.py` | Layout validation, canonical dot positions and coordinate transforms. |
| `stud_detector.py` | Circle proposals, radial verification, recovery and six dot decisions. |
| `braille_code.py` | Standard patterns, project alias, punctuation and sequence decoding. |
| `braille_reader.py` | Face rectification, dictionary decoding and row/result assembly. |
| `braille_visualization.py` | Annotated images and stud-crop previews. |
| `read_braille.py` | Command-line arguments, measurement, reports and file output. |
| `recording_pipeline.py` | END, WRITE_AGAIN, RESET and persistent word storage. |

Existing imports of `BrailleReader`, `detect_dots`, `validate_layout`, `project`,
`CENTERS`, `DESTINATION` and `previews` from `braille_reader` remain supported.
The module refactor preserves detection thresholds and public imports. The
subsequent decoder update adds punctuation candidates and decoder metadata.

Current generated results are retained for `lego.png` and `lego_final.jpg`:

- `outputs/lego/baseline/`: current report, previews, crops and evaluation for
  `lego.png`.
- `outputs/lego_final/baseline/`: current report, previews, crops and evaluation
  for `lego_final.jpg`.
- `outputs/lego_final/manual_reference.json`: manual printed-label and physical-dot
  annotations, used only for evaluation after recognition.
- `outputs/lego_final/historical/pre_texture_fix/`: original pre-fix run.
- `outputs/lego_final/historical/radial_before_refactor/`: previous improved run
  and its original before/after comparison, filtered to `lego_final.jpg`.
- `outputs/lego_final/historical/before_project_decoder/`: the prior baseline
  using the alphabet/number-only decoder.
- The recognition commands above refresh their reports, previews and crops.
  Each evaluation records a specific measured run and must be refreshed separately.

Other generated result folders were removed. Source photos, calibration layouts,
simulation datasets and printable board assets remain available. Benchmark and
replay commands regenerate their own outputs when explicitly run.

`board_alignment.py` provides geometric ORB/ArUco alignment independently of
character recognition. The robustness benchmark combines ORB alignment with the
six-dot reader. `tracking_exclusion_polygons` in layouts mask fixed regions out
of reference feature extraction. Without that field, brick label quarters are
excluded. `main.py` uses supplied static geometry; it is not a live tracker.

## Measured behavior and limits

For `lego_final.jpg`, all 16 detected stud patterns match the saved manual
annotations. The current decoder produces **`JUDY / FOX / RAIN / TREE.`**:
15/15 printed letters match, and all four rows are complete, including the final
period. The detector is unchanged; the newly supported A alias and period
resolve the two previously undecoded cells.

For `lego.png`, output in `letters` mode is
**`?BCDEFG / HIJKLMN / OPQRSTU / VWXYZ<CAP>. / ?!',-#`**. Printed-letter agreement
is 25/26 and the period is correct. A's top-left stud has only a verified weak
circle candidate; the strict candidate fails shape verification. Its six-dot
pattern therefore remains uncertain and the A alias cannot resolve it. The
last row now decodes all six symbols correctly. All-label agreement is 33/34;
only A remains uncertain. `<CAP>` and `#` denote recognized capital and number
indicators in this display mode.

Latency values are run-specific; consult the reports in each image's baseline
folder for the measured calls, warmups, thread count and timing distribution.
The boundary is decoded full-image processing across every calibrated region
(34 for `lego.png`, 16 for `lego_final.jpg`) through JSON serialization. It
excludes initialization, image decoding/file I/O, calibration, capture, transport
and rendering. These
are local CPU measurements; host load can vary between runs. See
`outputs/lego/baseline/lego_braille.json`,
`outputs/lego_final/baseline/lego_final_braille.json` and their `evaluation.json` files.
Historical timing comparisons remain under
`outputs/lego_final/historical/radial_before_refactor/comparison.json`.

These are manual annotations made during development, not independently verified
ground truth or an unseen test set. Synthetic texture tests cover specified
brightness/ridge conditions, not arbitrary materials. Extreme high-contrast
ridges can still create false studs, and missing a real stud can produce a wrong
valid letter. Shape scores are not calibrated confidence values. The update
does not provide automatic brick localization or solve arbitrary blur/movement.

## END-only capture, WRITE_AGAIN and RESET

Both six-dot and color recognition use `recording_pipeline.WordRecorder`. There
is no START signal, continuous frame processing, letter buffer or temporal vote.
Only an END signal supplies an image. Every fresh END analyzes exactly that one
final decoded frame, including optional alignment and all calibrated slots
(four in the included word simulations).

| Signal | Image analysis | Effect |
|---|---|---|
| END + final frame | One frame | Find newly occupied or changed letters, order left-to-right, save word and box. |
| WRITE_AGAIN | None | Remove only the latest saved word and release its occupied slots for rereading. |
| Next END after WRITE_AGAIN | One frame | Save the replacement word. Older saved words remain unchanged. |
| RESET | None | Clear words, current-board snapshot, pending rewrite and signal receipts. |

A current-board snapshot is separate from historical saved words. A slot counts
as new if it was empty at the last accepted END, or its recognized letter has
changed. Unchanged letters in unchanged slots are excluded. Removed bricks do
not delete saved words. An END showing an empty board updates occupancy without
saving a word or removing word history. This allows the same area and same
letters to be reused.

**One-frame ambiguity:** an identical letter replaced in the same slot cannot
be distinguished from the old brick. When rebuilding a word with overlapping
letters in the same slots, send END on the cleared board before constructing
that word. WRITE_AGAIN automatically reopens the latest word's slots, including
letters that are unchanged in the replacement. Bricks moved to another slot
are treated as new at that slot; physical identity is not tracked.

WRITE_AGAIN persists the removal immediately and does not take a frame. The
replacement remains pending until a later successful END. An unreadable END
saves nothing; the known-wrong word stays removed and older words remain. Repeated
WRITE_AGAIN while a replacement is pending does not remove another word.
Repeated END/WRITE_AGAIN event IDs are ignored without image analysis or a
second outgoing word, including after restarting the process.

Example synchronous adapter calls (reference and layout are already loaded):

```python
recorder = WordRecorder(reference, layout, "words.json")  # Six-dot by default.
generation = recorder.generation
result = recorder.handle_signal(
    "END", last_frame, event_id="capture-1", generation=generation
)
recorder.handle_signal("WRITE_AGAIN", event_id="undo-1", generation=generation)
result = recorder.handle_signal(
    "END", corrected_last_frame, event_id="capture-2", generation=generation
)
result = recorder.handle_signal("RESET", generation=generation)
generation = result["generation"]
```

For color, supply `reader_factory=partial(ColorReader, dictionary=dictionary)`.
Use a unique event ID for each new END or WRITE_AGAIN, including a fresh capture
retry after NEEDS_REVIEW; reuse the ID only for delivery retries. The generation
is reset metadata, not a START signal. Keep it on queued requests so delayed
END/WRITE_AGAIN/RESET from before RESET cannot change new storage. A Unity adapter
must serialize signals and associate each END with its final captured frame.
No live Unity connection is implemented in this simulation.

By default, alignment uses ArUco markers to map moving-camera coordinates onto
the reference board. The optional `alignment_mode="none"` uses unchanged image
coordinates instead, with no marker detection or camera registration. Presence compares each rectified crop to the empty reference: mean
absolute BGR difference above 25 means occupied, up to 15 means empty, and the
intermediate band is uncertain. This is a background-change heuristic; hands,
shadows and lighting shifts can affect it. Any uncertain presence or unreadable
occupied slot rejects the entire END, preventing a partial word from being
silently saved. A confident but wrong letter can still pass.

Storage version 2 atomically saves words, board occupancy, signal receipts and
pending rewrite state together. Loading nonempty version-1 recording storage
preserves its words; its first readable END establishes a baseline without
adding a word. Empty version-1 storage upgrades directly. Reader, calibration and alignment-mode mismatches in version 2 are rejected
rather than mixing incompatible positions.

Saved records contain the ordered characters (including repeated letters and
supported punctuation), unique IDs, board polygons, `box_xyxy`, and the final frame's
`image_box_xyxy`. Positions
come from **fixed placement-slot calibration**, not automatic brick-boundary
detection. Board-reference pixels are not Unity world coordinates. With alignment disabled,
`coordinate_system` is `calibrated_image_pixels`; the saved rectangles describe
the fixed image regions and do not follow camera motion. Previously
saved boxes never move when physical bricks move or disappear.

END preserves recognized punctuation in the saved word and its character positions;
for example, `A.` and `?.` remain intact in storage and `chatgpt_payload`.
Indicator bricks such as `#` are interpreted by the sequence decoder; the
word-only END recorder still rejects them as non-character input.
Only the word string appears in `chatgpt_payload`; WRITE_AGAIN and RESET emit
no word. No external request is sent. A downstream application that has already
received a wrong word needs a separate local correction/update mechanism;
word-only messages do not retract an earlier message automatically.

## Test one final frame per word

```bash
.venv/bin/python simulate_recording_dataset.py
.venv/bin/python test_recording_pipeline.py
.venv/bin/python color_example.py
.venv/bin/python test_recording_pipeline.py --method color
.venv/bin/python -m unittest discover -s tests -v
```

The new datasets are `datasets/end_only_six_dot/` and `datasets/end_only_color/`.
They contain only final END images; there are no START or intermediate frame
events. Photo-derived brick faces are placed in four calibrated slots with a
different planar camera transform for each image. Ground-truth words, positions
and transforms are loaded only after replay, never supplied to the recognizer.

The replay checks word order, calibrated boxes, single-frame END processing,
WRITE_AGAIN without image analysis, subsequent replacement, duplicate suppression,
empty-board handling and RESET. It exercises generated examples rather than
held-out real headset recordings. Test outputs are regenerated when run.

Replay artifacts include a signal timeline, word preview, per-END report, storage
snapshots and outgoing word strings. No external API request is made. The final
RESET leaves active test storage empty.

The older `datasets/four_words/` and `datasets/color_words/` retain historical
START/FRAME inputs. The current replay rejects those event streams; regenerate
with the commands above. Their old output reports have been removed.

## Quality test without markers or camera alignment

Run the complete six-dot quality experiment:

```bash
.venv/bin/python benchmark_no_alignment.py --runs 10
```

It writes `outputs/braille_no_alignment/report.json`, annotated photo results,
`word_conditions.png`, and an END/WRITE_AGAIN/RESET replay under `signals/`.
The marker-free final-frame dataset is `datasets/end_only_six_dot_no_alignment/`.
For only the signal replay, run:

```bash
.venv/bin/python simulate_recording_dataset.py --without-markers --stationary
.venv/bin/python test_recording_pipeline.py --alignment none
```

For application code, construct
`WordRecorder(reference, layout, storage, alignment_mode="none")`.
The existing ArUco mode remains the default. Use separate storage when changing
alignment modes. The no-alignment mode requires the same image dimensions as
its calibration; changed dimensions produce NEEDS_REVIEW instead of silently
rescaling the geometry.

**What is disabled:** marker detection, board registration and feature tracking.
Tests guard against BoardAligner, ArUco detector or ORB construction in this
mode. No marker graphics occur in the generated word frames.

**What is still provided:** manually calibrated brick-face corners. The reader
still normalizes each fixed face crop to its six-dot grid. No true camera
transform is passed to the reader, and no face location is updated from the
current image. This tests recognition in fixed image regions, not automatic
localization of freely placed bricks. On a moving headset, positions can be
wrong even when a word is recognized correctly.

Fixed image regions cannot compensate for camera movement. Blur, resampling and
misaligned crops can produce incorrect accepted letters, so rejecting uncertain
patterns alone does not guarantee the word is correct. Synthetic planar tests
do not model real 3D parallax, changing illumination or rolling shutter.

## Color-only example: JUDY, FOX, RAIN

`color_example.py` uses only color to identify letters for `samples/lego_color.jpg`.
It uses a separate color dictionary; `main.py` continues to use the six-dot
detector and project decoder described above. Run:

```bash
.venv/bin/python color_example.py
# Explicitly rebuild the example dictionary from its reference photo:
.venv/bin/python color_example.py --calibrate
```

Assignment is row-major: top row left-to-right, then bottom row left-to-right.

| Letter | Color | Letter | Color |
|---|---|---|---|
| J | tan | O | teal |
| U | light gray | X | yellow |
| D | blue | R | lime |
| Y | brown | A | dark teal |
| F | red | I | near-black |
| N | orange | | |

`color_dictionary.json` stores the measured CIELAB prototype, display RGB,
acceptance radius and source index for each letter. These are user-assigned
example codes, not Braille meanings or universal color names. The geometry in
`layouts/lego_color.json` contains eleven manually calibrated top faces.

`color_reader.py` warps each face to 80x160, samples an inset area, excludes the
darkest 35% and brightest 15% of pixels by lightness, and takes the median Lab
color. It uses Euclidean Lab distance (Delta E 76) to choose a dictionary entry.
A match must be within its acceptance radius and at least 2 distance units
better than the next candidate. Radius is the smaller of 15 and 45% of that
prototype's nearest-neighbor separation. Otherwise the output is unknown.
No stud count, brick dimensions, location or text is used as a letter feature.
The two darkest entries are only about 6.6 units apart, so their acceptance
radii are about 3.0. These settings are example-specific.

The existing recorder accepts an optional reader factory; its default remains
the six-dot reader. This color example supplies `ColorReader`, including for
the empty-board reference. A test disables stud detection entirely and verifies
that a flat color patch with no studs still records its assigned letter.

When run, the color example generates `datasets/end_only_color/` and exercises
the same END / WRITE_AGAIN / RESET flow. Static outputs are written to
`outputs/color_example/`; the standalone replay writes to
`outputs/end_only_color/`. Previous color results have been removed.

The color dictionary is calibrated on its reference photo. Synthetic
augmentations do not establish accuracy on unseen physical scenes, and exposure
or white-balance changes can affect matching. Geometry remains manually
calibrated; this is not automatic localization of freely placed bricks.
