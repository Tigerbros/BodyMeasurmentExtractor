"""Heuristics for turning landmarks into body measurements."""
from __future__ import annotations

import math

import numpy as np

from app.pose import Landmark, PoseDetection, has_visible_landmarks


DEFAULT_HEIGHT_CM = 170.0
REQUIRED_LANDMARKS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
]


def to_pixel_xy(
    landmark: Landmark,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    x = max(0, min(int(round(landmark.x * image_width)), image_width - 1))
    y = max(0, min(int(round(landmark.y * image_height)), image_height - 1))
    return x, y


def euclidean_distance(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
) -> float:
    return math.dist(point_a, point_b)


def average_y(*points: tuple[int, int]) -> int:
    return int(round(sum(point[1] for point in points) / len(points)))


def estimate_height_pixels(landmarks_px: dict[str, tuple[int, int]]) -> float | None:
    top_candidates = [
        landmarks_px[name][1]
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        if name in landmarks_px
    ]
    bottom_candidates = [
        landmarks_px[name][1]
        for name in ("left_ankle", "right_ankle", "left_heel", "right_heel")
        if name in landmarks_px
    ]

    if not top_candidates or not bottom_candidates:
        return None

    top_y = min(top_candidates)
    bottom_y = max(bottom_candidates)
    height_pixels = float(bottom_y - top_y)
    if height_pixels <= 0:
        return None
    return height_pixels


def measure_mask_width(
    mask: np.ndarray | None,
    y: int,
    threshold: float = 0.5,
) -> float | None:
    if mask is None or mask.size == 0:
        return None
    if y < 0 or y >= mask.shape[0]:
        return None

    row = mask[y]
    body_pixels = np.where(row > threshold)[0]

    if body_pixels.size < 2:
        return None

    return float(body_pixels[-1] - body_pixels[0])


def estimate_circumference_from_width(width_cm: float, factor: float) -> float:
    return width_cm * factor


def score_confidence(
    detection: PoseDetection,
    used_mask: bool,
    estimated_values_cm: dict[str, float],
) -> float:
    visibility_scores = [
        detection.image_landmarks[name].visibility
        for name in REQUIRED_LANDMARKS
        if name in detection.image_landmarks
    ]
    landmark_score = sum(visibility_scores) / len(visibility_scores)
    confidence = 0.45 + (0.35 * landmark_score)
    if used_mask:
        confidence += 0.15

    plausible_ranges = {
        "shoulder_width_cm": (25.0, 65.0),
        "chest_cm": (60.0, 160.0),
        "waist_cm": (45.0, 150.0),
        "hip_cm": (60.0, 170.0),
    }
    implausible_count = sum(
        not (lower <= estimated_values_cm[name] <= upper)
        for name, (lower, upper) in plausible_ranges.items()
    )
    if implausible_count:
        confidence -= 0.2 * implausible_count

    return round(min(confidence, 0.95), 2)


def estimate_measurements_from_landmarks(
    frame: np.ndarray,
    detection: PoseDetection,
    assumed_height_cm: float = DEFAULT_HEIGHT_CM,
) -> dict[str, float] | None:
    if frame is None or frame.size == 0:
        return None

    if not has_visible_landmarks(detection, REQUIRED_LANDMARKS, min_visibility=0.4):
        return None

    landmarks_px = {
        name: to_pixel_xy(landmark, detection.image_width, detection.image_height)
        for name, landmark in detection.image_landmarks.items()
    }

    height_pixels = estimate_height_pixels(landmarks_px)
    if height_pixels is None:
        return None

    cm_per_pixel = assumed_height_cm / height_pixels

    left_shoulder = landmarks_px["left_shoulder"]
    right_shoulder = landmarks_px["right_shoulder"]
    left_hip = landmarks_px["left_hip"]
    right_hip = landmarks_px["right_hip"]

    shoulder_width_cm = euclidean_distance(left_shoulder, right_shoulder) * cm_per_pixel

    shoulder_y = average_y(left_shoulder, right_shoulder)
    hip_y = average_y(left_hip, right_hip)
    torso_height = max(hip_y - shoulder_y, 1)

    # Use vertical torso fractions to pick horizontal body slices.
    chest_y = int(round(shoulder_y + (0.35 * torso_height)))
    waist_y = int(round(shoulder_y + (0.7 * torso_height)))

    chest_width_px = measure_mask_width(detection.segmentation_mask, chest_y)
    waist_width_px = measure_mask_width(detection.segmentation_mask, waist_y)
    hip_width_px = measure_mask_width(detection.segmentation_mask, hip_y)

    used_mask = all(width is not None for width in (chest_width_px, waist_width_px, hip_width_px))

    if not used_mask:
        shoulder_span_px = euclidean_distance(left_shoulder, right_shoulder)
        hip_span_px = euclidean_distance(left_hip, right_hip)
        chest_width_px = chest_width_px or (shoulder_span_px * 0.95)
        waist_width_px = waist_width_px or ((shoulder_span_px + hip_span_px) / 2 * 0.82)
        hip_width_px = hip_width_px or (hip_span_px * 1.05)

    chest_cm = estimate_circumference_from_width(chest_width_px * cm_per_pixel, factor=1.7)
    waist_cm = estimate_circumference_from_width(waist_width_px * cm_per_pixel, factor=1.55)
    hip_cm = estimate_circumference_from_width(hip_width_px * cm_per_pixel, factor=1.75)
    estimated_values_cm = {
        "shoulder_width_cm": shoulder_width_cm,
        "chest_cm": chest_cm,
        "waist_cm": waist_cm,
        "hip_cm": hip_cm,
    }

    return {
        "height_cm": round(assumed_height_cm, 1),
        "shoulder_width_cm": round(shoulder_width_cm, 1),
        "chest_cm": round(chest_cm, 1),
        "waist_cm": round(waist_cm, 1),
        "hip_cm": round(hip_cm, 1),
        "confidence": score_confidence(
            detection,
            used_mask=used_mask,
            estimated_values_cm=estimated_values_cm,
        ),
    }
