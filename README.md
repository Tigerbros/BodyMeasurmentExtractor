# Body Measurement Extractor

FastAPI MVP for estimating rough body measurements from either:

- 2 to 4 phone photos
- 1 short video

The current implementation is designed for the Haze internship take-home challenge and prioritizes:

- a clean REST API
- CPU-friendly inference
- simple, explainable heuristics
- fast deployment

## What It Returns

The API estimates:

- `height_cm`
- `shoulder_width_cm`
- `chest_cm`
- `waist_cm`
- `hip_cm`
- `confidence`
- `assumptions`

Example response:

```json
{
  "height_cm": 170.0,
  "shoulder_width_cm": 42.8,
  "chest_cm": 96.4,
  "waist_cm": 78.3,
  "hip_cm": 101.9,
  "confidence": 0.74,
  "assumptions": [
    "Single person in frame",
    "Full body visible",
    "Phone camera perspective introduces scale error"
  ]
}
```

## How It Works

At a high level:

1. Accept uploaded images or a video.
2. Decode images or sample a few frames from the video.
3. Run MediaPipe pose estimation.
4. Extract visible body landmarks and segmentation mask.
5. Estimate body height in pixels.
6. Use a fixed height anchor to derive a rough `cm_per_pixel` scale.
7. Estimate shoulder width from landmarks.
8. Estimate chest, waist, and hips from horizontal body slices.
9. Aggregate measurements across frames using the median.

## Project Structure

```text
app/
  main.py           FastAPI routes and request validation
  pipeline.py       Input orchestration, frame loading, aggregation
  pose.py           MediaPipe pose wrapper
  measurements.py   Measurement heuristics
  schemas.py        Pydantic response models
tests/
  test_health.py
  test_estimate.py
sample_data/
  images/
  videos/
```

## Local Setup

Create and activate the Conda environment:

```powershell
conda env create -f environment.yml
conda activate haze-mvp
```

If you prefer installing from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Run The API

```powershell
uvicorn app.main:app --reload
```

Once running, open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## API Usage

### Health Check

```powershell
curl http://127.0.0.1:8000/health
```

### Estimate From Images

```powershell
curl -X POST "http://127.0.0.1:8000/estimate" ^
  -F "files=@sample_data/images/staircase_person_sunglasses.jpg" ^
  -F "files=@sample_data/images/woman_standing.jpg"
```

### Estimate From Video

```powershell
curl -X POST "http://127.0.0.1:8000/estimate" ^
  -F "files=@sample_data/videos/fit_walking_short.mp4"
```

## Sample Media

Test assets are stored in:

- `sample_data/images/`
- `sample_data/videos/`

These files are intended for:

- local smoke testing
- endpoint demos
- Loom recording prep

Source links are listed in `sample_data/README.md`.

## Assumptions

The current estimator works best when:

- one person is present
- the full body is visible
- the subject is standing upright
- the background is not too cluttered
- clothing is not extremely loose
- images are front-facing or near front-facing

## Known Limitations

This is a rough heuristic MVP, not a production-grade body measurement system.

Important limitations:

- absolute scale is estimated from a fixed height anchor
- returned `height_cm` is currently a scaling assumption, not a truly measured height
- loose clothing can distort chest, waist, and hip estimates
- side-view information is limited unless the uploaded media includes it
- camera distance and perspective can affect results
- MediaPipe/environment compatibility should be verified inside the clean Conda env

## Deployment

The repo includes a simple `render.yaml` for Render deployment.

Start command:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Next Steps

Before submission, the main remaining tasks are:

- finish the smoke tests
- run the app end-to-end in the clean environment
- verify MediaPipe works correctly in that environment
- deploy and test the public URL
- record the Loom with one good case and one bad case
