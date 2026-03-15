"""Pose detection wrapper around MediaPipe."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "pose_landmarker_lite.task"
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


@dataclass(slots=True)
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass(slots=True)
class PoseDetection:
    image_landmarks: dict[str, Landmark]
    world_landmarks: dict[str, Landmark]
    image_width: int
    image_height: int
    segmentation_mask: np.ndarray | None = None


@lru_cache(maxsize=1)
def _create_pose_detector() -> Any:
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe could not be imported in the active environment."
        ) from exc

    if not MODEL_ASSET_PATH.exists():
        raise RuntimeError(
            f"Pose model asset was not found at {MODEL_ASSET_PATH}."
        )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_ASSET_PATH)),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
    )
    return PoseLandmarker.create_from_options(options)


def detect_pose_landmarks(frame: np.ndarray) -> PoseDetection | None:
    if frame is None or frame.size == 0:
        return None

    detector = _create_pose_detector()

    try:
        from mediapipe import Image as MPImage
        from mediapipe import ImageFormat
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe image helpers could not be imported in the active environment."
        ) from exc

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if not results.pose_landmarks:
        return None

    pose_landmarks = results.pose_landmarks[0]
    world_landmarks = results.pose_world_landmarks[0] if results.pose_world_landmarks else None

    image_landmark_map: dict[str, Landmark] = {}
    world_landmark_map: dict[str, Landmark] = {}

    for index, name in enumerate(LANDMARK_NAMES):
        image_landmark = pose_landmarks[index]
        image_landmark_map[name] = Landmark(
            x=float(image_landmark.x),
            y=float(image_landmark.y),
            z=float(image_landmark.z),
            visibility=float(image_landmark.visibility),
        )

        if world_landmarks is not None:
            world_landmark = world_landmarks[index]
            world_landmark_map[name] = Landmark(
                x=float(world_landmark.x),
                y=float(world_landmark.y),
                z=float(world_landmark.z),
                visibility=float(world_landmark.visibility),
            )

    height, width = frame.shape[:2]

    return PoseDetection(
        image_landmarks=image_landmark_map,
        world_landmarks=world_landmark_map,
        image_width=width,
        image_height=height,
        segmentation_mask=_extract_segmentation_mask(results),
    )


def _extract_segmentation_mask(results: Any) -> np.ndarray | None:
    masks = getattr(results, "segmentation_masks", None)
    if not masks:
        return None

    mask = masks[0].numpy_view()
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask[:, :, 0]
    return mask


def has_visible_landmarks(
    detection: PoseDetection | None,
    landmark_names: list[str],
    min_visibility: float = 0.5,
) -> bool:
    if detection is None:
        return False

    for name in landmark_names:
        landmark = detection.image_landmarks.get(name)
        if landmark is None or landmark.visibility < min_visibility:
            return False

    return True
