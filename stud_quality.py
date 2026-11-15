"""Local empty-slot evidence; no dictionary, source-photo or expected-bit inputs."""
from __future__ import annotations

import numpy as np

from braille_geometry import CENTERS

# Reuse one grid; fitted locations/radii are data-derived for each face.
_PIXEL_Y, _PIXEL_X = np.mgrid[:120, :80].astype(np.float32)


def _center_ring_contrast(raw, center, radius):
    """Absolute center/background contrast; both stud polarities are supported."""
    x, y = center
    distance_squared = (_PIXEL_X - x) ** 2 + (_PIXEL_Y - y) ** 2
    inner_radius = max(3.0, 0.6 * radius)
    inner = distance_squared <= inner_radius ** 2
    outer = (distance_squared >= (1.3 * radius) ** 2) & (
        distance_squared <= (1.6 * radius) ** 2
    )
    inner_count = int(inner.sum())
    outer_count = int(outer.sum())
    inner_coverage = min(1.0, inner_count / (np.pi * inner_radius ** 2))
    outer_coverage = min(1.0, outer_count / (np.pi * radius ** 2 * (1.6**2 - 1.3**2)))
    contrast = (
        abs(float(np.median(raw[inner])) - float(np.median(raw[outer])))
        if inner_count and outer_count else None
    )
    return contrast, inner_coverage, outer_coverage


def inspect_unconfirmed_slots(raw, evidence, circles):
    """Return diagnostics and uncertain zero-based slots; never assert a stud.

    Call after circle proposal/verification. Strict circle detections are anchors.
    If there are no strict anchors, use shape-verified candidates to place the
    diagnostic grid; this never grants them present/absent confidence. With no
    usable candidates, fall back to the nominal grid and radius. No confirmed
    studs means their contrast cannot establish local visibility, so all
    unresolved slots remain uncertain.

    Confirmed circle geometry aligns the empty-slot checks to the physical grid.
    Contrast greater than 20% of confirmed stud contrast (minimum 8 grayscale
    levels) is unexplained structure, so an unresolved slot is not marked empty.
    This is a conservative diagnostic heuristic, not a calibrated probability.
    """
    strict = [item['strict'] for item in evidence if item.get('strict') is not None]
    fallback = [
        item for item in circles
        if item.get('shape_verified') and item.get('local_contrast', 0) >= 15
        and item.get('edge_coverage', 0) >= 0.25
    ]
    anchors = strict or fallback
    if anchors:
        offsets = np.array([
            np.asarray(item['center']) - CENTERS[item['dot'] - 1]
            for item in anchors
        ])
        offset = np.median(offsets, axis=0)
        radius = float(np.median([item['radius'] for item in anchors]))
    else:
        offset = np.zeros(2, dtype=np.float32)
        radius = 12.0
    centers = CENTERS + offset
    visible_contrasts = [
        _center_ring_contrast(raw, centers[item['dot'] - 1], radius)[0]
        for item in strict
    ]
    visible_contrasts = [value for value in visible_contrasts if value is not None]
    reference_contrast = float(np.median(visible_contrasts)) if visible_contrasts else None
    contrast_limit = max(8.0, 0.2 * reference_contrast) if reference_contrast is not None else 8.0
    diagnostics = []
    uncertain = set()
    for index, item in enumerate(evidence):
        if item.get('strict') is not None:
            continue
        contrast, inner_coverage, outer_coverage = _center_ring_contrast(
            raw, centers[index], radius
        )
        reasons = []
        for candidate in circles:
            if candidate['dot'] != index + 1 or candidate['local_contrast'] < 15:
                continue
            # A partial rim may fail the stricter present-stud verification.
            # It still contradicts an empty slot; retain the texture safeguard.
            support_floor = 0.5 if candidate['coarse_gradient_floor'] > 15 else 0.25
            partial_rim = (
                candidate['edge_coverage'] >= 0.4
                and candidate['radial_alignment'] >= 0.82
                and candidate['coarse_radial_support'] >= support_floor
            )
            if candidate['shape_verified'] or partial_rim:
                reasons.append('unconfirmed_rim')
                break
        if not strict:
            reasons.append('no_confirmed_stud_reference')
        if inner_coverage < 0.75 or outer_coverage < 0.4:
            reasons.append('insufficient_local_area')
        if contrast is None or contrast >= contrast_limit:
            reasons.append('unexplained_local_contrast')
        if item.get('weak') is not None:
            reasons.append('weak_circle')
        if reasons:
            uncertain.add(index)
        diagnostics.append({
            'dot': index + 1,
            'aligned_center': centers[index].astype(float).tolist(),
            'reference_radius': radius,
            'center_ring_contrast': contrast,
            'confirmed_stud_contrast': reference_contrast,
            'absence_contrast_limit': contrast_limit,
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage,
            'absence_supported': not reasons,
            'uncertainty_reasons': reasons,
        })
    return diagnostics, uncertain
