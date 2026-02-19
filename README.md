# MotionTracking

Real-time pose estimation and motion tracking using MediaPipe and OpenCV.
Captures video from the webcam and overlays skeleton landmarks on the live feed.

## Prerequisites

- Python 3.10+
- Webcam

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
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

## Testing

```bash
pytest
```

## Linting

```bash
ruff check src/ tests/
```
