"""Replay END-only final frames, deferred WRITE_AGAIN, and RESET for either reader."""

import argparse
import copy
import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from benchmark_full_images import stats
from board_alignment import project
from read_braille import write_image
from recording_pipeline import WordRecorder

ROOT = Path(__file__).resolve().parent


def replay(dataset, out, *, reader_factory=None, recognition_method="six_dot"):
    cv2.setNumThreads(4)
    cv2.setRNGSeed(2026)
    out.mkdir(parents=True, exist_ok=True)
    # This named file is isolated test storage, never application storage.
    storage = out / "replay_storage.json"
    storage.write_text(
        json.dumps({"version": 1, "board_id": "simulation-board", "words": []})
    )
    reference = cv2.imread(str(dataset / "reference.png"))
    layout = json.loads((dataset / "layout.json").read_text())
    recorder = WordRecorder(reference, layout, storage, reader_factory=reader_factory)
    events = [
        json.loads(line) for line in (dataset / "events.jsonl").read_text().splitlines()
    ]
    if any(e["op"] in {"start", "frame"} for e in events):
        raise ValueError(
            "Regenerate this dataset: END-only replay does not consume START or intermediate frames"
        )
    trace, payloads, snapshots = [], [], {}
    generations = {0: recorder.generation}
    for index, event in enumerate(events):
        before_count = recorder.analysis_count
        if event["op"] == "snapshot":
            snapshots[event["name"]] = copy.deepcopy(recorder.words)
            (out / (event["name"] + ".json")).write_text(
                json.dumps(recorder.words, indent=2) + "\n"
            )
            result = {"status": "snapshot", "frames_analyzed": 0}
        else:
            if event["op"] != "end" and "image" in event:
                raise ValueError("Only END may carry an image")
            frame = (
                cv2.imread(str(dataset / event["image"])) if "image" in event else None
            )
            if "image" in event and frame is None:
                raise ValueError(f"Unreadable input image: {event['image']}")
            generation = generations[event["generation"]]
            result = recorder.handle_signal(
                event["op"],
                frame,
                event_id=event.get("event_id"),
                generation=generation,
            )
            if result["status"] == "reset":
                generations[event["generation"] + 1] = result["generation"]
            if result.get("chatgpt_payload") is not None:
                payloads.append(result["chatgpt_payload"])
        trace.append(
            {
                "event_index": index,
                "signal": event["op"].upper(),
                "event_id": event.get("event_id"),
                "image": event.get("image"),
                "result": result,
                "analysis_calls": recorder.analysis_count - before_count,
                "stored_words_after": [w["word"] for w in recorder.words],
                "board_state_after": copy.deepcopy(recorder.board_state),
                "rewrite_pending_after": recorder.rewrite_pending is not None,
            }
        )
    (out / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    timeline = [
        {
            "event_index": t["event_index"],
            "signal": t["signal"],
            "input_image": t["image"],
            "status": t["result"]["status"],
            "frames_analyzed": t["analysis_calls"],
            "stored_words": t["stored_words_after"],
            "rewrite_pending": t["rewrite_pending_after"],
            "outgoing_word": t["result"].get("chatgpt_payload"),
        }
        for t in trace
    ]
    (out / "signal_timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    (out / "chatgpt_payloads.jsonl").write_text(
        "".join(json.dumps(word) + "\n" for word in payloads)
    )
    # Truth is only consulted AFTER every signal has been handled.
    truth = json.loads((dataset / "ground_truth.json").read_text())
    comparisons = []
    for target in truth["expected_commits"]:
        result = trace[target["event_index"]]["result"]
        actual = result.get("record")
        comparisons.append(
            {
                "expected": target["word"],
                "actual": actual["word"] if actual else None,
                "status": result["status"],
                "frames_analyzed": result["frames_analyzed"],
                "expected_box": target["box_xyxy"],
                "actual_box": actual["box_xyxy"] if actual else None,
                "word_correct": actual is not None and actual["word"] == target["word"],
                "box_max_error_board_px": float(
                    np.max(np.abs(np.asarray(actual["box_xyxy"]) - target["box_xyxy"]))
                )
                if actual
                else None,
                "analysis_latency_ms": result.get("analysis_latency_ms"),
                "end_latency_ms": result.get("latency_ms"),
            }
        )
    errors = []
    for annotation in truth["frames"]:
        result = trace[annotation["event_index"]]["result"]
        if result.get("alignment", {}).get("status") != "tracked":
            continue
        transform = np.float64(annotation["reference_to_frame"])
        for observation in result["observations"]:
            actual = np.float32(observation["image_polygon"])
            expected = project(layout["cells"][observation["slot"]]["quad"], transform)
            errors.extend(np.linalg.norm(actual - expected, axis=1).tolist())
    before = snapshots["words_before_rewrite"]
    after_undo = snapshots["words_after_write_again"]
    saved = snapshots["words_before_reset"]
    reloaded = WordRecorder(reference, layout, storage, reader_factory=reader_factory)
    ignored = [
        t
        for t in trace
        if t["result"]["status"] in {"already_saved", "already_processed", "ignored"}
    ]
    checks = {
        "all_requested_words_exact_from_single_frames": [w["word"] for w in before]
        == truth["words"]
        and all(c["word_correct"] for c in comparisons),
        "no_start_or_intermediate_frames": all(
            e["op"] not in {"start", "frame"} for e in events
        ),
        "one_analysis_per_fresh_end": all(
            t["analysis_calls"]
            == int(
                t["signal"] == "END"
                and t["result"]["status"]
                not in {"already_saved", "already_processed", "ignored"}
            )
            for t in trace
        ),
        "boxes_match_calibrated_slots": all(
            c["box_max_error_board_px"] == 0 for c in comparisons
        ),
        "write_again_removes_only_latest_word": len(before) > 0
        and after_undo == before[:-1],
        "write_again_uses_no_image_or_cv": all(
            t["image"] is None
            and t["analysis_calls"] == 0
            and t["result"].get("chatgpt_payload") is None
            for t in trace
            if t["signal"] == "WRITE_AGAIN"
        ),
        "next_end_saves_replacement_preserving_older_words": saved[:-1] == before[:-1]
        and [w["word"] for w in saved] == truth["final_words"],
        "duplicates_and_stale_signals_do_not_analyze_or_emit": bool(ignored)
        and all(
            t["analysis_calls"] == 0 and t["result"].get("chatgpt_payload") is None
            for t in ignored
        ),
        "dismantling_does_not_change_storage": snapshots["words_after_board_clear"]
        == saved,
        "reset_clears_memory_and_disk": recorder.words == reloaded.words == []
        and recorder.board_state == reloaded.board_state == []
        and recorder.rewrite_pending is None,
        "only_word_strings_in_outgoing_payloads": payloads
        == [*truth["words"], truth["replacement_word"]]
        and all(type(p) is str for p in payloads),
    }
    tree = next((w for w in before if w["word"] == "TREE"), None)
    if "TREE" in truth["words"]:
        checks["tree_has_two_distinct_e_tracks"] = bool(
            tree
            and len(tree["letters"]) == 4
            and tree["letters"][2]["letter"] == tree["letters"][3]["letter"] == "E"
            and tree["letters"][2]["track_id"] != tree["letters"][3]["track_id"]
        )
    analyzed = [t["result"] for t in trace if t["analysis_calls"]]
    report = {
        "protocol": "END frame -> save; WRITE_AGAIN -> remove latest only; next END frame -> replacement; RESET -> clear",
        "recognition_method": recognition_method,
        "simulation": truth["simulation"],
        "images": len(truth["frames"]),
        "events": len(events),
        "signal_counts": {
            op: sum(e["op"] == op for e in events)
            for op in ["end", "write_again", "reset"]
        },
        "words": comparisons,
        "stored_words_before_rewrite": [w["word"] for w in before],
        "stored_words_after_write_again": [w["word"] for w in after_undo],
        "stored_words_before_reset": [w["word"] for w in saved],
        "checks": checks,
        "analyzed_frames": recorder.analysis_count,
        "analysis_latency_ms": stats([r["analysis_latency_ms"] for r in analyzed]),
        "end_including_storage_latency_ms": stats([r["latency_ms"] for r in analyzed]),
        "camera_corner_error_px": stats(errors),
        "payloads": payloads,
        "timing_boundary": "Decoded END frame through alignment, four slot readings, presence/difference checks and assembly. END total additionally includes atomic storage and serialization. Excludes capture/transport, file decoding and initialization. These are individual replay calls, not a large latency benchmark.",
        "limitations": [
            "Photo-derived planar composites, not real Quest footage; four calibrated slots.",
            "All new letters must be visible together in the one END frame. No temporal voting or recovery.",
            "New means a different letter or previously empty slot. An identical letter replaced at the same slot is indistinguishable; END on an empty board first enables reuse while preserving history.",
            "Saved boxes come from slot calibration. ArUco estimates the moving camera projection; no 3D Unity coordinates.",
            "Unrecognized visible slots reject the entire capture, but confidently wrong classifications remain possible.",
            "No ChatGPT request was sent. WRITE_AGAIN changes local storage; word-only payloads do not retract any previously delivered external message.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    tiles = []
    for index, path in enumerate(truth["complete_word_frames"]):
        original = cv2.imread(str(dataset / path))
        result = trace[truth["expected_commits"][index]["event_index"]]["result"]
        for letter in result.get("record", {}).get("letters", []):
            poly = np.int32(letter["image_polygon"])
            cv2.polylines(original, [poly], True, (0, 220, 0), 2)
        image = cv2.resize(original, (576, 384))
        tile = np.full((430, 576, 3), 245, np.uint8)
        tile[46:] = image
        pair = comparisons[index]
        caption = f"END: {pair['expected']} -> {pair['actual']} | one frame"
        cv2.putText(
            tile,
            caption,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 245))
    write_image(
        out / "words_preview.png",
        np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["six_dot", "color"], default="six_dot")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dictionary", type=Path, default=ROOT / "color_dictionary.json"
    )
    args = parser.parse_args()
    dataset = args.dataset or ROOT / "datasets" / (
        "end_only_color" if args.method == "color" else "end_only_six_dot"
    )
    out = args.output_dir or ROOT / "outputs" / (
        "end_only_color" if args.method == "color" else "end_only_six_dot"
    )
    factory = None
    if args.method == "color":
        from color_reader import ColorReader

        factory = partial(
            ColorReader, dictionary=json.loads(args.dictionary.read_text())
        )
    report = replay(
        dataset,
        out,
        reader_factory=factory,
        recognition_method="example_color" if args.method == "color" else "six_dot",
    )
    print(json.dumps(report, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
