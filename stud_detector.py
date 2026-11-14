"""Detect the six stud positions in a rectified 80-by-120 LEGO face crop.

Candidate circles must satisfy local contrast, edge coverage, and circular edge
orientation checks. Strict evidence asserts a stud; weak evidence leaves that
position uncertain. These scores are diagnostics, not probabilities.
"""

from __future__ import annotations

import cv2
import numpy as np

from braille_geometry import CENTERS

# Baseline for future detector comparisons. Advance the ID when recognition
# behavior changes; code-only refactors keep the same identity.
ALGORITHM_ID = "six_dot_radial_v1"
ALGORITHM_DESCRIPTION = (
    "calibrated six-dot stud detection with radial verification "
    "and coarse-smoothed recovery; no OCR"
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


def _radial_scores(gradients, x, y, radius):
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
    support = float(((alignment >= 0.82) & (magnitude >= 15)).any(axis=0).mean())
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
        coarse_support, _ = _radial_scores(coarse_gradients, *key)
        shape_cache[key] = (fine_alignment, coarse_support)
    fine_alignment, coarse_support = shape_cache[key]
    shape_verified = coarse_support >= 0.4 or fine_alignment >= 0.90
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
        "shape_verified": shape_verified,
        "proposal_source": source,
    }


def _record_candidate(item, evidence, rejected_dots):
    """Keep the strongest valid evidence and identify dots needing recovery."""
    index = item["dot"] - 1
    pass_name = item["pass"]
    coverage = item["edge_coverage"]
    threshold = 0.5 if pass_name == "strict" else 0.6
    has_edge_and_contrast = coverage >= threshold and item["local_contrast"] >= 15
    if (
        item["proposal_source"] == "normal"
        and pass_name == "strict"
        and has_edge_and_contrast
        and not item["shape_verified"]
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
    if sharpness < 8 or float(raw.std()) < 3:
        return {
            "bits": [None] * 6,
            "sharpness": sharpness,
            "circles": [],
            "dot_evidence": [],
            "reason": "low_detail",
        }

    enhanced, distance_to_edge = _prepare_edges(raw, clahe)
    fine_gradients = _gradient_fields(raw, 1.0)
    coarse_gradients = _gradient_fields(raw, 2.0)
    shape_cache = {}
    evidence = [{"dot": i + 1, "strict": None, "weak": None} for i in range(6)]
    circles_output = []
    rejected_dots = set()
    for pass_name, votes, source in [
        ("strict", 12, "normal"),
        ("weak", 10, "normal"),
        ("strict", 12, "coarse_smoothed"),
    ]:
        if source == "coarse_smoothed":
            if not rejected_dots:
                continue
            # Surface ridges can displace a proposal away from the true stud.
            # Retry only those rejected dot positions, retaining all other gates.
            enhanced, distance_to_edge = _prepare_edges(raw, clahe, coarse_smoothed=True)
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
                shape_cache,
                pass_name,
                source,
                rejected_dots,
            )
            if item is None:
                continue
            circles_output.append(item)
            _record_candidate(item, evidence, rejected_dots)

    bits = [
        True if e["strict"] is not None else None if e["weak"] is not None else False
        for e in evidence
    ]
    return {
        "bits": bits,
        "sharpness": sharpness,
        "circles": circles_output,
        "dot_evidence": evidence,
        "reason": "ambiguous_dot" if any(b is None for b in bits) else "ok",
    }
