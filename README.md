# MotionTracking

Real-time motion tracking with a hacker/Matrix aesthetic. Uses MediaPipe and OpenCV to detect body pose, hands, and face landmarks from a webcam and renders them as overlays on the live feed.
<img width="630" height="420" alt="{EE9E4A4B-A165-41C7-BDB7-4F4040FEAF44}" src="https://github.com/user-attachments/assets/bcd8cb1a-ac07-461d-a0f6-33a078532ee3" /> <img width="610" height="490" alt="{3C5317AA-9F71-4493-AB8C-CA5852012BC7}" src="https://github.com/user-attachments/assets/9c174c7c-3b28-42a5-bf39-077741917378" />

## Features

- **Pose mode** — full body skeleton (33 landmarks) with background darkening using MediaPipe's segmentation mask
- **Hands mode** — hand skeleton for up to 2 hands (21 landmarks each)
- **Face mode** — neural-link mesh connecting the 3 nearest neighbours of each of the 468 face landmarks
- Mirrored video feed for a natural feel
- Neon green glow on landmarks
- Models are downloaded automatically on first use (~5–28 MB per mode)

## Controls

| Key | Action |
|-----|--------|
| `1` | Switch to Pose mode |
| `2` | Switch to Hands mode |
| `3` | Switch to Face mode |
| `q` / `Esc` | Quit |

## Requirements

- Python 3.10+
- Webcam
- Internet connection on first run (to download MediaPipe models)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install -e .
```

## Usage

```bash
motion-tracking
```

Or directly:

```bash
python -m motion_tracking.main
```

## Development

Install dev dependencies (includes pytest and ruff):

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest
```

Lint:

```bash
ruff check src/ tests/
```

## Tech stack

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) — landmark detection models
- [OpenCV](https://opencv.org/) — video capture and rendering

## License

MIT
