import enum
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Landmark connection graphs ────────────────────────────────────────────────

_POSE_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
])

_HAND_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Wrist to pinky base
])


# ── Tracking mode ─────────────────────────────────────────────────────────────

class TrackingMode(enum.Enum):
    POSE = 1
    HANDS = 2
    FACE_MESH = 3

    @classmethod
    def from_key(cls, key: int) -> "TrackingMode | None":
        mapping = {
            ord("1"): cls.POSE,
            ord("2"): cls.HANDS,
            ord("3"): cls.FACE_MESH,
        }
        return mapping.get(key)

    @property
    def display_name(self) -> str:
        names = {
            TrackingMode.POSE: "Pose",
            TrackingMode.HANDS: "Hands",
            TrackingMode.FACE_MESH: "Face Mesh",
        }
        return names[self]


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


# ── Drawing helper ────────────────────────────────────────────────────────────

def _draw_landmarks(
    frame: np.ndarray,
    landmarks: list,
    connections: frozenset[tuple[int, int]] | None = None,
    point_color: tuple = (0, 255, 0),
    line_color: tuple = (255, 255, 255),
    point_radius: int = 4,
    line_thickness: int = 2,
) -> None:
    h, w = frame.shape[:2]
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                s = landmarks[start_idx]
                e = landmarks[end_idx]
                cv2.line(
                    frame,
                    (int(s.x * w), int(s.y * h)),
                    (int(e.x * w), int(e.y * h)),
                    line_color,
                    line_thickness,
                )
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), point_radius, point_color, -1)


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
            for pose_landmarks in result.pose_landmarks:
                _draw_landmarks(frame, pose_landmarks, _POSE_CONNECTIONS)
        elif self._mode == TrackingMode.HANDS:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
            for hand_landmarks in result.hand_landmarks:
                _draw_landmarks(frame, hand_landmarks, _HAND_CONNECTIONS)
        elif self._mode == TrackingMode.FACE_MESH:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
            for face_landmarks in result.face_landmarks:
                _draw_landmarks(frame, face_landmarks, point_radius=1)

        return frame

    def __enter__(self) -> "MotionTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None


# ── UI overlay ────────────────────────────────────────────────────────────────

def draw_mode_overlay(frame: np.ndarray, mode: TrackingMode) -> np.ndarray:
    label = f"Mode: {mode.display_name}"
    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame
