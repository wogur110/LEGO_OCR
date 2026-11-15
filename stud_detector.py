"""Detect the six stud positions in a rectified 80-by-120 LEGO face crop.

Candidate circles must satisfy local contrast, edge coverage, and circular edge
orientation checks. Strict evidence asserts a stud; weak rims or unexplained
local structure remain uncertain. Absence needs local support. These scores
are diagnostics, not probabilities.
"""

from __future__ import annotations

import cv2
import numpy as np

from braille_geometry import CENTERS
from stud_quality import inspect_unconfirmed_slots

# Baseline for future detector comparisons. Advance the ID when recognition
# behavior changes; code-only refactors keep the same identity.
ALGORITHM_ID = "six_dot_local_quality_v4"
ALGORITHM_DESCRIPTION = (
    "calibrated six-dot stud detection with radial verification, "
    "coarse and contrast-normalized proposals, texture-adaptive rim support, "
    "and local missing-stud uncertainty; no OCR"
)

ANGLES = np.arange(48) * np.pi / 24
RADIAL_COS = np.cos(ANGLES).astype(np.float32)[None, :]
RADIAL_SIN = np.sin(ANGLES).astype(np.float32)[None, :]
RADIAL_OFFSETS = np.arange(-2, 3, dtype=np.float32)[:, None]


def _gradient_fields(gray, sigma):
    """Compute each smoothed gradient field once per crop."""
    kernel = (5, 5) if sigma == 1.0 else (0, 0)
    smooth = cv2.GaussianBlur(gray, kernel, sigma)
    return (
        cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3),
    )


def _radial_scores(gradients, x, y, radius, magnitude_floor=15):
    """Measure whether edge directions follow a circular rim.

    Both gradient polarities count, so a dark rim or a bright rim can qualify.
    Missing samples outside the crop contribute no support.
    """
    gx, gy = gradients
    xx = (x + (radius + RADIAL_OFFSETS) * RADIAL_COS).astype(np.float32)
    yy = (y + (radius + RADIAL_OFFSETS) * RADIAL_SIN).astype(np.float32)
    sx = cv2.remap(gx, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sy = cv2.remap(gy, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    magnitude = np.hypot(sx, sy)
    alignment = np.abs(sx * RADIAL_COS + sy * RADIAL_SIN) / (magnitude + 1e-6)
    support = float(((alignment >= 0.82) & (magnitude >= magnitude_floor)).any(axis=0).mean())
    strongest = magnitude.argmax(axis=0)
    weighted = float(np.average(
        alignment[strongest, np.arange(len(ANGLES))],
        weights=magnitude.max(axis=0) + 1e-6,
    ))
    return support, weighted


def _prepare_edges(raw, clahe, coarse_smoothed=False):
    """Prepare the proposal image and distances to its nearest edges."""
    kernel, sigma = ((0, 0), 2.0) if coarse_smoothed else ((5, 5), 0.7)
    enhanced = cv2.GaussianBlur(clahe.apply(raw), kernel, sigma)
    edges = cv2.Canny(enhanced, 30, 60)
    distance_to_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return enhanced, distance_to_edge


def _score_candidate(
    circle,
    raw,
    distance_to_edge,
    fine_gradients,
    coarse_gradients,
    coarse_magnitude_floor,
    shape_cache,
    pass_name,
    source,
    rejected_dots,
):
    """Associate a proposal with a dot and collect its rim diagnostics."""
    x, y, radius = circle
    distances = np.linalg.norm(CENTERS - [x, y], axis=1)
    index = int(distances.argmin())
    if distances[index] >= 18:
        return None
    if source == "coarse_smoothed" and index not in rejected_dots:
        return None
    xx = np.round(x + radius * np.cos(ANGLES)).astype(int)
    yy = np.round(y + radius * np.sin(ANGLES)).astype(int)
    inside = (xx >= 0) & (xx < 80) & (yy >= 0) & (yy < 120)
    if int(inside.sum()) < 24:
        return None
    coverage = float((distance_to_edge[yy[inside], xx[inside]] <= 2).mean())
    cx, cy = CENTERS[index].astype(int)
    patch = raw[cy - 15 : cy + 16, cx - 15 : cx + 16]
    contrast = float(np.percentile(patch, 95) - np.percentile(patch, 5))
    key = (float(x), float(y), float(radius))
    if key not in shape_cache:
        _, fine_alignment = _radial_scores(fine_gradients, *key)
        coarse_support, _ = _radial_scores(
            coarse_gradients, *key, magnitude_floor=coarse_magnitude_floor
        )
        shape_cache[key] = (fine_alignment, coarse_support)
    fine_alignment, coarse_support = shape_cache[key]
    shape_verified = coarse_support >= 0.4 or fine_alignment >= 0.90
    # A textured face has many strong, approximately parallel edges. Fine
    # alignment alone can mistake a ridge or rounded face corner for a stud.
    # Require half the rim to have radial edges above the crop's noise floor.
    required_support = 0.5 if coarse_magnitude_floor > 15 else 0.0
    shape_verified = shape_verified and coarse_support >= required_support
    return {
        "center": [float(x), float(y)],
        "radius": float(radius),
        "dot": index + 1,
        "edge_coverage": coverage,
        "local_contrast": contrast,
        "center_error": float(distances[index]),
        "pass": pass_name,
        "radial_alignment": fine_alignment,
        "coarse_radial_support": coarse_support,
        "coarse_gradient_floor": coarse_magnitude_floor,
        "required_coarse_support": required_support,
        "shape_verified": shape_verified,
        "proposal_source": source,
    }


def _record_candidate(item, evidence, rejected_dots):
    """Keep valid evidence and flag plausible but unconfirmed circle locations."""
    index = item["dot"] - 1
    pass_name = item["pass"]
    coverage = item["edge_coverage"]
    threshold = 0.5 if pass_name == "strict" else 0.6
    has_edge_and_contrast = coverage >= threshold and item["local_contrast"] >= 15
    # Weak-only circles and strict circles failing coverage or shape can be
    # displaced by texture/blur. This only enables a new proposal pass; it does
    # not promote the candidate or relax any acceptance threshold.
    if (
        item["proposal_source"] == "normal"
        and evidence[index]["strict"] is None
        and coverage >= 0.4
        and item["local_contrast"] >= 15
        and (pass_name == "weak" or not (has_edge_and_contrast and item["shape_verified"]))
    ):
        rejected_dots.add(index)
    if has_edge_and_contrast and item["shape_verified"]:
        old = evidence[index][pass_name]
        if old is None or coverage > old["edge_coverage"]:
            evidence[index][pass_name] = item


def detect_dots(crop, clahe):
    """Return six ternary decisions and the supporting circle diagnostics."""
    if crop.shape[:2] != (120, 80):
        raise ValueError("Stud crop must be 80 pixels wide and 120 pixels high")
    raw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(
        cv2.Laplacian(cv2.GaussianBlur(raw, (3, 3), 0.5), cv2.CV_32F).var()
    )
    # Whole-crop sharpness depends on stud count and is diagnostic only.
    # Native scale is reported by the reader; actual local rim evidence drives
    # decisions, including for the approximately seven-pixel Quest studs.
    if float(raw.std()) < 3:
        return {
            "bits": [None] * 6,
            "local_quality": [],
            "grid_max_offset_error_px": None,
            "sharpness": sharpness,
            "circles": [],
            "dot_evidence": [],
            "reason": "low_detail",
        }

    enhanced, distance_to_edge = _prepare_edges(raw, clahe)
    fine_gradients = _gradient_fields(raw, 1.0)
    coarse_gradients = _gradient_fields(raw, 2.0)
    coarse_magnitude_floor = max(
        15.0, 2.0 * float(np.median(np.hypot(*coarse_gradients)))
    )
    shape_cache = {}
    evidence = [{"dot": i + 1, "strict": None, "weak": None} for i in range(6)]
    circles_output = []
    rejected_dots = set()
    for pass_name, votes, source in [
        ("strict", 12, "normal"),
        ("weak", 10, "normal"),
        ("strict", 12, "coarse_smoothed"),
        ("strict", 12, "contrast_normalized"),
    ]:
        if source == "coarse_smoothed":
            if not rejected_dots:
                continue
            # Surface ridges can displace a proposal away from the true stud.
            # Smooth only to propose replacements for unconfirmed dot positions.
            # Verify every proposal against the same normal edge map: stronger
            # smoothing can erase a real rim even when it improves the circle.
            # Recovery uses the same contrast, adaptive shape and coverage
            # checks as the original proposals.
            enhanced, _ = _prepare_edges(raw, clahe, coarse_smoothed=True)
        if source == "contrast_normalized":
            # Bounded gain helps Hough find faint rims. Discard this pass's edge
            # map: all acceptance scores still use the original raw pixels.
            # In particular, amplification cannot pass the raw contrast floor.
            low, high = np.percentile(raw, [5, 95])
            contrast = float(high - low)
            gain = min(3.0, max(1.0, 70.0 / max(contrast, 1.0)))
            if gain <= 1.05 or contrast < 15 or all(e["strict"] for e in evidence):
                continue
            normalized = np.clip(
                (raw.astype(np.float32) - (low + high) / 2) * gain + 128, 0, 255
            ).astype(np.uint8)
            enhanced, _ = _prepare_edges(normalized, clahe)
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=22,
            param1=60,
            param2=votes,
            minRadius=8,
            maxRadius=17,
        )
        if circles is None:
            continue
        for circle in circles[0]:
            item = _score_candidate(
                circle,
                raw,
                distance_to_edge,
                fine_gradients,
                coarse_gradients,
                coarse_magnitude_floor,
                shape_cache,
                pass_name,
                source,
                rejected_dots,
            )
            if item is None:
                continue
            circles_output.append(item)
            _record_candidate(item, evidence, rejected_dots)

    local_quality, uncertain = inspect_unconfirmed_slots(raw, evidence, circles_output)
    for quality in local_quality:
        if not quality["absence_supported"]:
            continue
        # Rim-only studs can have the same center intensity as the background.
        # Check their expected location even when Hough proposed no circle.
        support, _ = _radial_scores(
            coarse_gradients, *quality["aligned_center"], quality["reference_radius"],
            magnitude_floor=coarse_magnitude_floor,
        )
        quality["coarse_radial_support"] = support
        if support >= (0.5 if coarse_magnitude_floor > 15 else 0.4):
            uncertain.add(quality["dot"] - 1)
            quality["absence_supported"] = False
            quality["uncertainty_reasons"].append("unconfirmed_radial_structure")
    bits = [
        True if e["strict"] is not None else None if i in uncertain else False
        for i, e in enumerate(evidence)
    ]
    strict = [e["strict"] for e in evidence if e["strict"] is not None]
    grid_error = None
    reason = "ambiguous_dot" if any(b is None for b in bits) else "ok"
    if len(strict) >= 2:
        offsets = np.array([np.array(c["center"]) - CENTERS[c["dot"] - 1] for c in strict])
        grid_error = float(np.linalg.norm(offsets - np.median(offsets, axis=0), axis=1).max())
        # A quarter of the 40-pixel pitch: individually circular marks must
        # still agree on one translated grid. Do not guess a replacement cell.
        if grid_error > 10:
            bits = [None] * 6
            reason = "inconsistent_stud_grid"
            for quality in local_quality:
                quality["absence_supported"] = False
                quality["uncertainty_reasons"].append(reason)
    return {
        "bits": bits,
        "local_quality": local_quality,
        "grid_max_offset_error_px": grid_error,
        "sharpness": sharpness,
        "circles": circles_output,
        "dot_evidence": evidence,
        "reason": reason,
    }
