import enum

import cv2
import numpy as np


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


def process_frame(frame: np.ndarray, mode: TrackingMode) -> np.ndarray:
    if mode == TrackingMode.POSE:
        return frame
    elif mode == TrackingMode.HANDS:
        return frame
    elif mode == TrackingMode.FACE_MESH:
        return frame
    return frame


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
