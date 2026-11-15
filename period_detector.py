"""Find a standalone filled printed period in a rectified LEGO label crop.

No Braille decoding, text recognition, expected label, or cell index is used.
The label crop should include modest padding above the printed label, matching
the 76x49 region prepared by CalibratedOCRReader. Size gates scale with crop area.
A dark dirt speck matching a printed period remains visually ambiguous.
"""

import cv2
import numpy as np


ALGORITHM_ID = 'printed_period_component_v1'


def detect_period(crop):
    if crop is None or crop.ndim not in (2, 3):
        raise ValueError('Expected a grayscale or BGR label crop')
    height, width = crop.shape[:2]
    if min(height, width) < 16:
        return {'accepted': False, 'algorithm_id': ALGORITHM_ID, 'reason': 'crop_too_small'}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    if gray.dtype != np.uint8:
        raise ValueError('Expected an 8-bit label crop')
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.5)
    # Exclude top stud remnants and physical side/bottom edges. These are fixed
    # relative label-crop margins, not locations obtained from a reference photo.
    x0, x1 = round(width * 0.08), round(width * 0.92)
    y0, y1 = round(height * 0.20), round(height * 0.90)
    roi = smooth[y0:y1, x0:x1]
    background = float(np.percentile(roi, 75))
    threshold = max(0.0, background - max(28.0, 0.30 * background))
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = (roi < threshold).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    image_area = height * width
    substantial = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= max(3, image_area * 0.001)]
    total_ink = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in substantial)
    candidates = []
    for index in substantial:
        x, y, w, h, pixels = (int(v) for v in stats[index])
        center_x, center_y = (float(v) for v in centers[index])
        area_fraction = pixels / image_area
        component = (labels == index).astype(np.uint8)
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outer = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(outer))
        perimeter = float(cv2.arcLength(outer, True))
        circularity = 4 * np.pi * contour_area / max(perimeter ** 2, 1.0)
        holes = sum(float(cv2.contourArea(c)) for j, c in enumerate(contours)
                    if hierarchy is not None and hierarchy[0, j, 3] != -1)
        fill_ratio = pixels / (w * h)
        aspect = w / h
        dominance = pixels / max(total_ink, 1)
        radius = max(2, round(max(w, h) * 0.35))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        ring = cv2.dilate(component, kernel).astype(bool) & ~component.astype(bool)
        contrast = float(np.median(smooth[ring]) - np.mean(smooth[component.astype(bool)])) if ring.any() else 0.0
        core_dark = float(smooth[round(center_y), round(center_x)]) < threshold
        checks = {
            'size': 0.006 <= area_fraction <= 0.045,
            'aspect': 0.65 <= aspect <= 1.5,
            'filled': fill_ratio >= 0.55 and holes / max(contour_area, 1.0) <= 0.08 and core_dark,
            'round': circularity >= 0.65,
            'isolated': dominance >= 0.80,
            'contrast': contrast >= 25,
            'interior': x > x0 and y > y0 and x + w < x1 and y + h < y1,
            'printed_position': 0.20 <= center_x / width <= 0.80 and 0.35 <= center_y / height <= 0.85,
        }
        candidates.append({
            'box_xywh': [x, y, w, h], 'center': [center_x, center_y],
            'pixels': pixels, 'area_fraction': area_fraction, 'aspect': aspect,
            'fill_ratio': fill_ratio, 'hole_area': holes, 'circularity': float(circularity),
            'ink_dominance': dominance, 'contrast': contrast, 'checks': checks,
            'accepted': all(checks.values()),
        })
    accepted = [c for c in candidates if c['accepted']]
    return {
        'accepted': len(accepted) == 1, 'algorithm_id': ALGORITHM_ID,
        'reason': 'isolated_filled_printed_dot' if len(accepted) == 1 else 'no_unique_printed_dot',
        'background': background, 'threshold': threshold, 'total_ink_pixels': total_ink,
        'roi_xyxy': [x0, y0, x1, y1], 'candidates': candidates,
    }
