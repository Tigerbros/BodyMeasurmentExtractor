# Sample Data

This folder contains sample media for local smoke testing and demo runs.

## Images

- `images/staircase_person_sunglasses.jpg`
  - Source: https://commons.wikimedia.org/wiki/File:Person_standing_on_indoor_staircase_wearing_sunglasses.jpg
- `images/woman_standing.jpg`
  - Source: https://commons.wikimedia.org/wiki/File:A_woman_standing.jpg

## Video

- `videos/fit_walking_short.mp4`
  - Trimmed locally from: https://commons.wikimedia.org/wiki/File:Fit_walking.webmhd.webm
  - Duration: about 8 seconds

## Notes

- JPEG is the preferred format for the sample photos because it is smaller and plenty good for endpoint testing.
- The short MP4 is intended for faster API runs than the full source video.

## Recommended Demo Cases

- Reasonable case:
  - `images/staircase_person_sunglasses.jpg`
  - `images/woman_standing.jpg`
- Struggling case:
  - `images/staircase_person_sunglasses.jpg`
  - `videos/fit_walking_short.mp4`
  - Why: cluttered background and motion are more likely to stress the heuristics.
- Video demo:
  - `videos/fit_walking_short.mp4`
  - Good for showing how frame sampling behaves on motion.
