import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from motion_tracking.drawing import (
    _HAND_CONNECTIONS,
    _POSE_CONNECTIONS,
    _draw_landmarks,
    _draw_neural_links,
)
from motion_tracking.modes import TrackingMode

# ── Model file management ─────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

_MODEL_FILES: dict[TrackingMode, tuple[str, str]] = {
    TrackingMode.POSE: (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    TrackingMode.HANDS: (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    TrackingMode.FACE_MESH: (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}


def _ensure_model(mode: TrackingMode) -> Path:
    _MODELS_DIR.mkdir(exist_ok=True)
    filename, url = _MODEL_FILES[mode]
    path = _MODELS_DIR / filename
    if not path.exists():
        print(f"Downloading model: {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded: {filename}")
    return path


# ── Motion tracker ────────────────────────────────────────────────────────────

class MotionTracker:
    def __init__(self, mode: TrackingMode) -> None:
        self._mode = mode
        self._detector = None
        self._init_detector()

    @property
    def mode(self) -> TrackingMode:
        return self._mode

    def _init_detector(self) -> None:
        model_path = _ensure_model(self._mode)
        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        run_mode = mp_vision.RunningMode.VIDEO

        if self._mode == TrackingMode.POSE:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=run_mode,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=True,
            )
            self._detector = mp_vision.PoseLandmarker.create_from_options(options)
        elif self._mode == TrackingMode.HANDS:
            options = mp_vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=run_mode,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(options)
        elif self._mode == TrackingMode.FACE_MESH:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=run_mode,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def set_mode(self, mode: TrackingMode) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        self._init_detector()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        if self._mode == TrackingMode.POSE:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
            if result.segmentation_masks:
                mask = np.squeeze(result.segmentation_masks[0].numpy_view())
                h, w = frame.shape[:2]
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                darken = (0.2 + 0.8 * mask)[:, :, np.newaxis]
                frame[:] = np.clip(frame * darken, 0, 255).astype(np.uint8)
            for pose_landmarks in result.pose_landmarks:
                _draw_landmarks(frame, pose_landmarks, _POSE_CONNECTIONS)
        elif self._mode == TrackingMode.HANDS:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
            for hand_landmarks in result.hand_landmarks:
                _draw_landmarks(frame, hand_landmarks, _HAND_CONNECTIONS)
        elif self._mode == TrackingMode.FACE_MESH:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
            for face_landmarks in result.face_landmarks:
                _draw_neural_links(frame, face_landmarks, n_neighbors=3)
                _draw_landmarks(frame, face_landmarks, point_radius=1)

        return frame

    def __enter__(self) -> "MotionTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
