"""End-to-end processing pipeline for images and video inputs."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from tempfile import NamedTemporaryFile
from typing import Iterable

import cv2
import numpy as np
from fastapi import UploadFile

from app.measurements import estimate_measurements_from_landmarks
from app.pose import detect_pose_landmarks
from app.schemas import MeasurementResponse


@dataclass(slots=True)
class FrameResult:
    height_cm: float
    shoulder_width_cm: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    confidence: float


class MeasurementPipeline:
    def __init__(self, max_video_frames: int = 5) -> None:
        self.max_video_frames = max_video_frames

    async def run(
        self,
        images: list[UploadFile] | None = None,
        video: UploadFile | None = None,
    ) -> MeasurementResponse:
        frames = await self._load_frames(images=images, video=video)

        if not frames:
            raise ValueError("No usable frames were found in the upload.")

        frame_results: list[FrameResult] = []

        for frame in frames:
            landmarks = detect_pose_landmarks(frame)
            if landmarks is None:
                continue

            estimate = estimate_measurements_from_landmarks(frame, landmarks)
            if estimate is None:
                continue

            frame_results.append(
                FrameResult(
                    height_cm=estimate["height_cm"],
                    shoulder_width_cm=estimate["shoulder_width_cm"],
                    chest_cm=estimate["chest_cm"],
                    waist_cm=estimate["waist_cm"],
                    hip_cm=estimate["hip_cm"],
                    confidence=estimate["confidence"],
                )
            )

        if not frame_results:
            raise ValueError("Could not estimate measurements from the provided media.")

        return self._aggregate_results(frame_results)
    

    async def _load_frames(
        self,
        images: list[UploadFile] | None,
        video: UploadFile | None,
    ) -> list[np.ndarray]:
        if images:
            return [await self._read_image(file) for file in images]

        if video:
            return await self._read_video_frames(video)

        raise ValueError("Provide either images or a video.")
    

    async def _read_image(self, file: UploadFile) -> np.ndarray:
        raw = await file.read()
        array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError(f"Failed to decode image: {file.filename}")

        return frame
    

    async def _read_video_frames(self, file: UploadFile) -> list[np.ndarray]:
        raw = await file.read()
        suffix = ""
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", maxsplit=1)[-1]

        with NamedTemporaryFile(delete=True, suffix=suffix) as temp_file:
            temp_file.write(raw)
            temp_file.flush()

            capture = cv2.VideoCapture(temp_file.name)
            frames: list[np.ndarray] = []

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total_frames == 0:
                capture.release()
                return frames

            sample_points = np.linspace(
                0,
                max(total_frames - 1, 0),
                num=min(self.max_video_frames, total_frames),
                dtype=int,
            )

            for index in sample_points:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(frame)

            capture.release()
            return frames

    def _aggregate_results(self, results: Iterable[FrameResult]) -> MeasurementResponse:
        results = list(results)

        return MeasurementResponse(
            height_cm=round(median(item.height_cm for item in results), 1),
            shoulder_width_cm=round(median(item.shoulder_width_cm for item in results), 1),
            chest_cm=round(median(item.chest_cm for item in results), 1),
            waist_cm=round(median(item.waist_cm for item in results), 1),
            hip_cm=round(median(item.hip_cm for item in results), 1),
            confidence=round(median(item.confidence for item in results), 2),
            assumptions=[
                "Single person in frame",
                "Full body visible",
                "Phone camera perspective introduces scale error",
            ],
        )
