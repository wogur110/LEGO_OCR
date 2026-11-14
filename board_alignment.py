"""Planar board alignment for six-dot Braille images; no character recognition."""

import cv2
import numpy as np

from braille_reader import validate_layout


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def project(points, homography):
    return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), homography)[0]


class BoardAligner:
    def __init__(self, reference, layout, method="aruco", tracking_mask=None):
        if method not in {"aruco", "orb"}:
            raise ValueError("Alignment method must be aruco or orb")
        self.method = method
        self.height, self.width = reference.shape[:2]
        # Validate calibration even when callers use the aligner alone.
        validate_layout(layout, (self.width, self.height))
        if method == "aruco":
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self.markers = self._markers(reference)
            if len(self.markers) < 2:
                raise ValueError(
                    "Reference needs at least two unique DICT_4X4_100 markers"
                )
        else:
            self.detector = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            if tracking_mask is None:
                tracking_mask = np.full(reference.shape[:2], 255, np.uint8)
            if tracking_mask.shape != reference.shape[:2]:
                raise ValueError("Tracking mask must match reference dimensions")
            tracking_mask = tracking_mask.copy().astype(np.uint8)
            labels = np.zeros(reference.shape[:2], np.uint8)
            polygons = layout.get("tracking_exclusion_polygons")
            if polygons is None:
                polygons = []
                for cell in layout["cells"]:
                    q = np.float32(cell["quad"])
                    polygons.append(
                        [
                            q[0] * 0.25 + q[3] * 0.75,
                            q[1] * 0.25 + q[2] * 0.75,
                            q[2],
                            q[3],
                        ]
                    )
            for polygon in polygons:
                q = np.asarray(polygon, np.float32)
                if q.shape != (4, 2) or not np.isfinite(q).all():
                    raise ValueError(
                        "Tracking exclusions must be finite quadrilaterals"
                    )
                cv2.fillConvexPoly(labels, np.int32(q), 255)
            labels = cv2.dilate(labels, np.ones((17, 17), np.uint8))
            tracking_mask[labels > 0] = 0
            self.ref_points, self.ref_descriptors = self._features(
                reference, tracking_mask
            )
            if self.ref_descriptors is None or len(self.ref_points) < 12:
                raise ValueError(
                    "Reference has too few static features outside excluded regions"
                )

    def _markers(self, image):
        corners, identifiers, _ = self.detector.detectMarkers(gray(image))
        if identifiers is None:
            return {}
        ids = identifiers.ravel().tolist()
        # A duplicated marker ID is ambiguous; do not choose one arbitrarily.
        return {
            i: c.reshape(4, 2)
            for i, c in zip(ids, corners, strict=True)
            if ids.count(i) == 1
        }

    def _features(self, image, mask=None):
        scale = min(1.0, 1200 / max(image.shape[:2]))
        small = cv2.resize(
            gray(image), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        if mask is not None:
            mask = cv2.resize(
                mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        keypoints, descriptors = self.detector.detectAndCompute(small, mask)
        points = np.float32([k.pt for k in keypoints]).reshape(-1, 2) / scale
        return points, descriptors

    def align(self, frame):
        info = {"status": "lost", "method": self.method, "matches": 0, "inliers": 0}
        if self.method == "aruco":
            current = self._markers(frame)
            ids = sorted(set(current) & set(self.markers))
            info["marker_ids"] = ids
            if len(ids) < 2:
                return None, {**info, "reason": "fewer_than_two_known_markers"}
            src = np.concatenate([current[i] for i in ids])
            dst = np.concatenate([self.markers[i] for i in ids])
            minimum = 8
        else:
            points, descriptors = self._features(frame)
            if descriptors is None or len(descriptors) < 12:
                return None, {**info, "reason": "too_few_features"}
            pairs = self.matcher.knnMatch(descriptors, self.ref_descriptors, k=2)
            reverse = self.matcher.match(self.ref_descriptors, descriptors)
            backward = {m.queryIdx: m.trainIdx for m in reverse}
            good = [
                p[0]
                for p in pairs
                if len(p) == 2
                and p[0].distance < 0.75 * p[1].distance
                and backward.get(p[0].trainIdx) == p[0].queryIdx
            ]
            info["matches"] = len(good)
            if len(good) < 12:
                return None, {**info, "reason": "too_few_matches"}
            src = np.float32([points[m.queryIdx] for m in good])
            dst = np.float32([self.ref_points[m.trainIdx] for m in good])
            minimum = 12
        info["matches"] = len(src)
        homography, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995
        )
        if homography is None or mask is None or not np.isfinite(homography).all():
            return None, {**info, "reason": "homography_failed"}
        inliers = mask.ravel().astype(bool)
        info["inliers"] = int(inliers.sum())
        info["inlier_ratio"] = float(inliers.mean())
        if info["inliers"] < minimum or info["inlier_ratio"] < 0.45:
            return None, {**info, "reason": "insufficient_consensus"}
        if (
            self.method == "aruco"
            and int(np.all(inliers.reshape(-1, 4), axis=1).sum()) < 2
        ):
            return None, {**info, "reason": "fewer_than_two_consistent_markers"}
        errors = np.linalg.norm(
            project(src[inliers], homography) - dst[inliers], axis=1
        )
        info["median_reprojection_error_px"] = float(np.median(errors))
        hull = cv2.convexHull(dst[inliers]).reshape(-1, 2)
        if info["median_reprojection_error_px"] > 2.0 or cv2.contourArea(hull) < 600:
            return None, {**info, "reason": "weak_geometry"}
        # Reject mirrored/folded views and a projective horizon through the frame.
        h, w = frame.shape[:2]
        boundary = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        denom = np.c_[boundary, np.ones(4)] @ homography[2]
        mapped = project(boundary, homography)
        if (
            abs(np.linalg.det(homography)) < 1e-10
            or not np.isfinite(mapped).all()
            or not ((denom > 1e-6).all() or (denom < -1e-6).all())
            or not cv2.isContourConvex(mapped)
            or cv2.contourArea(mapped, oriented=True) <= 0
        ):
            return None, {**info, "reason": "invalid_projective_geometry"}
        return homography, {
            **info,
            "status": "tracked",
            "reason": "ok",
            "homography_frame_to_reference": homography.tolist(),
            "support_polygon_reference": hull.tolist(),
        }
