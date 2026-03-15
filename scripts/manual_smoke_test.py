from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.main import app

    client = TestClient(app)

    with (
        (root / "sample_data" / "images" / "staircase_person_sunglasses.jpg").open("rb") as image_one,
        (root / "sample_data" / "images" / "standing woman.jpg").open("rb") as image_two,
    ):
        response = client.post(
            "/estimate",
            files={
                "image_1": ("staircase_person_sunglasses.jpg", image_one, "image/jpeg"),
                "image_2": ("standing woman.jpg", image_two, "image/jpeg"),
            },
        )
        print("IMAGE_STATUS", response.status_code)
        print("IMAGE_JSON", response.json())

    with (root / "sample_data" / "videos" / "fit_walking_short.mp4").open("rb") as video_file:
        response = client.post(
            "/estimate",
            files={"video": ("fit_walking_short.mp4", video_file, "video/mp4")},
        )
        print("VIDEO_STATUS", response.status_code)
        print("VIDEO_JSON", response.json())


if __name__ == "__main__":
    main()
