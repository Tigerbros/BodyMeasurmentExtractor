from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from fastapi import UploadFile


async def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.measurements import REQUIRED_LANDMARKS, estimate_measurements_from_landmarks
    from app.pipeline import MeasurementPipeline
    from app.pose import detect_pose_landmarks, has_visible_landmarks

    pipeline = MeasurementPipeline(max_video_frames=8)
    with (root / "sample_data" / "videos" / "fit_walking_short.mp4").open("rb") as video_file:
        upload = UploadFile(filename="fit_walking_short.mp4", file=video_file)
        frames = await pipeline._read_video_frames(upload)

    print("frames", len(frames))

    for index, frame in enumerate(frames):
        detection = detect_pose_landmarks(frame)
        print("frame", index, "has_detection", detection is not None)
        if detection is None:
            continue

        visible = has_visible_landmarks(detection, REQUIRED_LANDMARKS, min_visibility=0.4)
        print("frame", index, "visible_required", visible)
        missing = [
            name
            for name in REQUIRED_LANDMARKS
            if name not in detection.image_landmarks
            or detection.image_landmarks[name].visibility < 0.4
        ]
        print("frame", index, "missing_or_low_visibility", missing)
        estimate = estimate_measurements_from_landmarks(frame, detection)
        print("frame", index, "estimate", estimate)


if __name__ == "__main__":
    asyncio.run(main())
