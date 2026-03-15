"""FastAPI entrypoint for the body measurement MVP."""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.pipeline import MeasurementPipeline
from app.schemas import ErrorResponse, HealthResponse, MeasurementResponse


app = FastAPI(
    title="Body Measurement Extraction API",
    version="0.1.0",
    description=(
        "Estimate rough body measurements from a short video or 2-4 phone photos.\n\n"
        "**POST /estimate** — upload either:\n"
        "- A single video file (use the `video` field), OR\n"
        "- 2 to 4 images (use `image_1` + `image_2`, and optionally `image_3` / `image_4`)\n\n"
        "Leave unused fields empty."
    ),
)
pipeline = MeasurementPipeline()


async def _run_pipeline(
    images: list[UploadFile] | None = None,
    video: UploadFile | None = None,
) -> MeasurementResponse:
    try:
        return await pipeline.run(images=images, video=video)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/estimate",
    response_model=MeasurementResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    tags=["measurements"],
    summary="Estimate body measurements from images or a video",
)
async def estimate_measurements(
    video:   UploadFile | None = File(None, description="A short video file (MP4 / MOV). Leave empty if uploading images."),
    image_1: UploadFile | None = File(None, description="1st image (required when using images)."),
    image_2: UploadFile | None = File(None, description="2nd image (required when using images)."),
    image_3: UploadFile | None = File(None, description="3rd image (optional)."),
    image_4: UploadFile | None = File(None, description="4th image (optional)."),
) -> MeasurementResponse:
    """
    Single endpoint — choose ONE mode:

    **Video mode:** fill in `video` only, leave all image fields empty.

    **Image mode:** fill in at least `image_1` and `image_2` (up to 4 images), leave `video` empty.
    """
    has_video = video is not None and (video.filename or "") != ""
    image_candidates = [image_1, image_2, image_3, image_4]
    images = [f for f in image_candidates if f is not None and (f.filename or "") != ""]

    # --- nothing uploaded at all ---
    if not has_video and not images:
        raise HTTPException(
            status_code=400,
            detail="Upload either a video file or between 2 and 4 images.",
        )

    # --- both uploaded ---
    if has_video and images:
        raise HTTPException(
            status_code=400,
            detail="Upload either a video OR images, not both at the same time.",
        )

    # --- video mode ---
    if has_video:
        content_type = video.content_type or ""
        if not content_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail=f"Expected a video file but received content-type '{content_type}'.",
            )
        return await _run_pipeline(video=video)

    # --- image mode ---
    if len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Provide at least 2 images. You uploaded {len(images)}.",
        )

    non_images = [f for f in images if not (f.content_type or "").startswith("image/")]
    if non_images:
        raise HTTPException(
            status_code=400,
            detail="All image fields must be image files (JPEG / PNG).",
        )

    return await _run_pipeline(images=images)
