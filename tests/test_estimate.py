"""API-level test for the estimate endpoint."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import MeasurementResponse


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "sample_data" / "images"
VIDEOS_DIR = ROOT_DIR / "sample_data" / "videos"


def open_binary(path: Path):
    return path.open("rb")


class EstimateEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_estimate_accepts_two_images(self) -> None:
        mock_response = MeasurementResponse(
            height_cm=170.0,
            shoulder_width_cm=42.8,
            chest_cm=96.4,
            waist_cm=78.3,
            hip_cm=101.9,
            confidence=0.74,
            assumptions=[
                "Single person in frame",
                "Full body visible",
                "Phone camera perspective introduces scale error",
            ],
        )

        with (
            open_binary(IMAGES_DIR / "staircase_person_sunglasses.jpg") as image_one,
            open_binary(IMAGES_DIR / "standing woman.jpg") as image_two,
            patch("app.main.pipeline.run", new=AsyncMock(return_value=mock_response)) as run_mock,
        ):
            response = self.client.post(
                "/estimate",
                files={
                    "image_1": ("staircase_person_sunglasses.jpg", image_one, "image/jpeg"),
                    "image_2": ("standing woman.jpg", image_two, "image/jpeg"),
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response.model_dump())
        run_mock.assert_awaited_once()

    def test_estimate_accepts_video(self) -> None:
        mock_response = MeasurementResponse(
            height_cm=170.0,
            shoulder_width_cm=41.5,
            chest_cm=94.1,
            waist_cm=79.8,
            hip_cm=99.7,
            confidence=0.68,
            assumptions=[
                "Single person in frame",
                "Frame sampling may miss ideal poses",
                "Phone camera perspective introduces scale error",
            ],
        )

        with (
            open_binary(VIDEOS_DIR / "fit_walking_short.mp4") as video_file,
            patch("app.main.pipeline.run", new=AsyncMock(return_value=mock_response)) as run_mock,
        ):
            response = self.client.post(
                "/estimate",
                files={"video": ("fit_walking_short.mp4", video_file, "video/mp4")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response.model_dump())
        run_mock.assert_awaited_once()

    def test_estimate_rejects_single_image(self) -> None:
        with open_binary(IMAGES_DIR / "staircase_person_sunglasses.jpg") as image_file:
            response = self.client.post(
                "/estimate",
                files={"image_1": ("staircase_person_sunglasses.jpg", image_file, "image/jpeg")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "Provide at least 2 images. You uploaded 1."},
        )
