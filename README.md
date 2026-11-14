# LEGO six-dot Braille analysis

Recognize raised stud patterns with OpenCV and the standard English six-dot
Braille alphabet/number dictionary. The only runtime dependencies are OpenCV
and NumPy. No model downloads or cloud services are required. An isolated color-based example
for `samples/lego_color.jpg` is documented below; six-dot reading remains the default.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
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

A–J share their patterns with 1–9,0. A numeric indicator (dots 3456) or explicit
numeric mode distinguishes numbers from letters. The standard mappings remain
unchanged by photo contents. The brick printed A1 in `lego.png` visibly has
**dots 1 and 6**, so it is not decoded as A (dot 1 only).

The dictionary covers A–Z, digits, number/capital/grade-1 indicators. This is an
uncontracted alphabet/integer subset, not a full translator for punctuation,
contractions or capitalization. Unsupported patterns remain unknown.
Sources: [UEB Manual](https://www.brailleauthority.org/ueb/NZ%20UEB%20Manual%202011.pdf)
and [AFB cell diagrams](https://www.afb.org/diagrams-braille-cells).

## Pipeline and outputs

`braille_reader.py` rectifies each calibrated brick face, samples only the upper
three stud rows, detects circles, assigns six dot decisions and performs exact
dictionary lookup. The printed-label quarter is excluded. Results contain dot
bits, letter/number candidates, original-image coordinates and rejection reasons.

- `outputs/braille/`: per-image results, overlays and stud crops.
- `outputs/braille_full_images/`: both-photo accuracy and whole-image timing.
- `outputs/braille_robustness/`: blur tests, simulated moving frames, overlays,
  alignment results and timing samples.

`board_alignment.py` provides geometric ORB/ArUco alignment independently of
character recognition. The robustness benchmark combines ORB alignment with the
six-dot reader. `tracking_exclusion_polygons` in layouts mask fixed regions out
of reference feature extraction. Without that field, brick label quarters are
excluded. `main.py` uses supplied static geometry; it is not a live tracker.

## Measured behavior and limits

Whole-image reference benchmark (50 calls, four CPU threads): 93.99 ms mean for
34 bricks in `lego.png`, and 88.49 ms for 30 bricks in `lego2.jpg`. These are
previous measurements, not fresh timings after this code cleanup. The boundary
is decoded full image through serialized results; file I/O, camera acquisition,
manual calibration and rendering are excluded.

The first photo matched 34/34 visible patterns and 25/26 printed alphabet labels;
the second matched 29/30 patterns and all 26 letters, with an uncertain capital
sign. Under Gaussian blur sigma=2 pixels, correct letters dropped to 2/26 and
16/26. Sharp simulated movement with ORB yielded 300/312 and 302/312 correct
letters at 202/219 ms per frame; motion plus seven-pixel blur yielded 167/312
and 219/312, including incorrect letters and one lost frame. Early rejection
can reduce latency without improving usable accuracy.

Simulations are planar image warps, not real moving-camera validation. They do
not model parallax, changing lighting, rolling shutter or occlusion. No temporal
voting is implemented. These two development photos do not establish accuracy
on unseen scenes. Full measurements and explicit failure counts remain in the
retained reports.

## END-only capture, WRITE_AGAIN and RESET

Both six-dot and color recognition use `recording_pipeline.WordRecorder`. There
is no START signal, continuous frame processing, letter buffer or temporal vote.
Only an END signal supplies an image. Every fresh END analyzes exactly that one
final decoded frame, including alignment and all four calibrated slots.

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

Alignment uses ArUco markers to map moving-camera coordinates onto the reference
board. Presence compares each rectified crop to the empty reference: mean
absolute BGR difference above 25 means occupied, up to 15 means empty, and the
intermediate band is uncertain. This is a background-change heuristic; hands,
shadows and lighting shifts can affect it. Any uncertain presence or unreadable
occupied slot rejects the entire END, preventing a partial word from being
silently saved. A confident but wrong letter can still pass.

Storage version 2 atomically saves words, board occupancy, signal receipts and
pending rewrite state together. Loading nonempty version-1 recording storage
preserves its words; its first readable END establishes a baseline without
adding a word. Empty version-1 storage upgrades directly. Reader/calibration
mismatches in version 2 are rejected rather than mixing incompatible positions.

Saved records contain the ordered letters (including repeated letters), unique
IDs, board polygons, `box_xyxy`, and the final frame's `image_box_xyxy`. Positions
come from **fixed placement-slot calibration**, not automatic brick-boundary
detection. Board-reference pixels are not Unity world coordinates. Previously
saved boxes never move when physical bricks move or disappear.

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

Fresh replay results:

| Reader | Initial END frames | WRITE_AGAIN | Following END |
|---|---|---|---|
| Six-dot | JUDY, FOX, RAIN, TREE, each correct from one frame | Removes TREE; no CV | Saves REED from one frame |
| Color | JUDY, FOX, RAIN, each correct from one frame | Removes RAIN; no CV | Saves FAIR from one frame |

Both runs preserve older records exactly during rewrite, suppress duplicate
signals, retain words after an empty-board END, and reject delayed signals after
RESET. All 12 six-dot and 11 color integration checks pass. Thirty unit tests
also pass, covering unreadable frames, camera movement, repeated letters,
restart, failed disk writes and reset generations; signal cases run with both
readers. This validates these generated examples, not accuracy on unseen scenes.

The six-dot replay analyzes six frames total: four words, one replacement and
one empty-board check. Color analyzes five: three words, one replacement and
one empty-board check. Duplicates, WRITE_AGAIN and RESET analyze zero frames.
In this individual replay, mean analysis/END-including-storage time was
20.41/21.66 ms for six-dot and 14.04/15.06 ms for color. These small samples include
an empty board and are not a general latency benchmark. Timing excludes camera
capture, transmission, image decoding and reference initialization.

Inspect `outputs/end_only_six_dot/` and `outputs/end_only_color/`:

- `signal_timeline.json`: each signal, input image, CV-call count, words afterward,
  pending-rewrite flag and outgoing string.
- `words_preview.png`: final-frame images and saved words.
- `report.json`: exact word/box comparisons, checks and per-END latency.
- `words_before_rewrite.json`, `words_after_write_again.json`, and
  `words_before_reset.json`: deliberate evaluation snapshots.
- `replay_storage.json`: active test storage, empty after final RESET.
- `chatgpt_payloads.jsonl`: outgoing word strings only; no API call.

The older `datasets/four_words/`, `datasets/color_words/` and their recording
reports are retained as historical START/FRAME experiments. The current replay
rejects those older event streams; regenerate with the commands above.

## Color-only example: JUDY, FOX, RAIN

`color_example.py` uses only color to identify letters for `samples/lego_color.jpg`.
The normal `main.py` and six-dot dictionary are unchanged. Run:

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

The color example now generates `datasets/end_only_color/` and runs the same
END / WRITE_AGAIN / RESET implementation as the six-dot test. Its static outputs
remain in `outputs/color_example/`: `reference_annotated.png`,
`static_report.json` and perturbed image previews. Its current signal replay is
in the `end_only/` subfolder; the standalone color replay command writes to
`outputs/end_only_color/`. Both preserve snapshots for evaluation and leave
active storage empty after RESET.

Static tests (30 measured full-image calls per condition, four CPU threads):

| Condition | Correct / 11 | Wrong | Unknown | Mean ms |
|---|---:|---:|---:|---:|
| reference | 11 | 0 | 0 | 15.18 |
| gaussian_sigma_2 | 11 | 0 | 0 | 14.54 |
| motion_7px | 11 | 0 | 0 | 14.67 |
| brightness_0.75 | 4 | 0 | 7 | 10.55 |
| brightness_1.15 | 11 | 0 | 0 | 10.70 |
| warm_white_balance | 7 | 0 | 4 | 10.84 |

The table above records the earlier 30-call static color measurements; the
letter classifier is unchanged. Current END-only timings and their boundary
are reported in the signal test section above. Static timing includes all
eleven calibrated faces without marker alignment; END timing includes the
four-slot board and marker alignment.

Reference accuracy is a calibration check on the same photo used to build the
dictionary. Augmentations and composites are synthetic, not held-out physical
measurements. Recognition tolerates the tested blur, but exposure/white balance
changes cause rejection. Geometry is calibrated; this is not automatic detection
of arbitrarily placed bricks. A single-frame decision cannot recover from systematic color errors.
