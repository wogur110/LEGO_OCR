"""Example-specific color-to-letter lookup using calibrated brick-face regions.

Only color statistics are classified. Shape, studs, position, and printed text
are not features in the letter classifier.
"""

import copy
import json
from time import perf_counter

import cv2
import numpy as np
from braille_reader import validate_layout

DESTINATION = np.float32([[0, 0], [79, 0], [79, 159], [0, 159]])


def extract_color(crop):
    # Ignore edges/shadows around the face; include studs without interpreting them.
    interior = crop[16:144, 12:68]
    lab = cv2.cvtColor(interior.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3)
    # A robust middle-brightness subset suppresses deep stud shadows and highlights.
    low, high = np.percentile(pixels[:, 0], [35, 85])
    selected = pixels[(pixels[:, 0] >= low) & (pixels[:, 0] <= high)]
    return np.median(selected, axis=0).astype(float)


def classify_color(feature, dictionary):
    entries = dictionary["entries"]
    prototypes = np.asarray([e["lab"] for e in entries])
    distances = np.linalg.norm(prototypes - feature, axis=1)
    order = np.argsort(distances)
    first, second = int(order[0]), int(order[1])
    best, alternative = float(distances[first]), float(distances[second])
    margin = alternative - best
    radius = entries[first]["acceptance_radius_delta_e76"]
    accepted = best <= radius and margin >= dictionary["minimum_margin_delta_e76"]
    return {
        "letter_candidate": entries[first]["letter"] if accepted else None,
        "nearest_candidate": entries[first]["letter"],
        "accepted": accepted,
        "reason": "ok"
        if accepted
        else "unknown_color"
        if best > radius
        else "ambiguous_color",
        "color_lab": np.asarray(feature).tolist(),
        "color_distance": best,
        "color_margin": margin,
        "color_name": entries[first]["color_name"] if accepted else None,
    }


class ColorReader:
    def __init__(self, layout, dictionary):
        validate_layout(layout, tuple(layout["image_size"]))
        if dictionary.get("version") != 1 or len(dictionary.get("entries", [])) < 2:
            raise ValueError(
                "Color dictionary needs version 1 and at least two entries"
            )
        self.layout = copy.deepcopy(layout)
        self.dictionary = dictionary

    def read(self, image):
        start = perf_counter()
        validate_layout(self.layout, (image.shape[1], image.shape[0]))
        cells, crops = [], []
        for cell in self.layout["cells"]:
            q = np.float32(cell["quad"])
            transform = cv2.getPerspectiveTransform(q, DESTINATION)
            crop = cv2.warpPerspective(
                image, transform, (80, 160), flags=cv2.INTER_CUBIC
            )
            decision = classify_color(extract_color(crop), self.dictionary)
            cells.append(
                {
                    **decision,
                    "row_id": cell["row_id"],
                    "slot_index": cell["slot_index"],
                    "face_polygon": q.tolist(),
                }
            )
            crops.append(crop)
        result = {
            "method": "example_color",
            "cells": cells,
            "image_size": [image.shape[1], image.shape[0]],
        }
        json.dumps(result, allow_nan=False)
        result["timing_ms"] = {"image_to_result": (perf_counter() - start) * 1000}
        return result, crops
