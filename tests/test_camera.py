from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from motion_tracking.camera import MotionTracker, TrackingMode, draw_mode_overlay


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestTrackingMode:
    def test_from_key_returns_pose_for_1(self):
        assert TrackingMode.from_key(ord("1")) == TrackingMode.POSE

    def test_from_key_returns_hands_for_2(self):
        assert TrackingMode.from_key(ord("2")) == TrackingMode.HANDS

    def test_from_key_returns_face_mesh_for_3(self):
        assert TrackingMode.from_key(ord("3")) == TrackingMode.FACE_MESH

    def test_from_key_returns_none_for_unknown_key(self):
        assert TrackingMode.from_key(ord("z")) is None

    def test_from_key_returns_none_for_quit_key(self):
        assert TrackingMode.from_key(ord("q")) is None

    def test_display_name_pose(self):
        assert TrackingMode.POSE.display_name == "Pose"

    def test_display_name_hands(self):
        assert TrackingMode.HANDS.display_name == "Hands"

    def test_display_name_face_mesh(self):
        assert TrackingMode.FACE_MESH.display_name == "Face Mesh"

    def test_all_modes_have_display_names(self):
        for mode in TrackingMode:
            assert isinstance(mode.display_name, str)
            assert len(mode.display_name) > 0


# Patch order (bottom = first arg after self):
#   mock_mp         → motion_tracking.camera.mp
#   mock_vision     → motion_tracking.camera.mp_vision
#   mock_python     → motion_tracking.camera.mp_python
#   mock_ensure     → motion_tracking.camera._ensure_model
@patch("motion_tracking.camera._ensure_model", return_value=Path("/fake/model.task"))
@patch("motion_tracking.camera.mp_python")
@patch("motion_tracking.camera.mp_vision")
@patch("motion_tracking.camera.mp")
class TestMotionTracker:
    def test_init_creates_pose_detector(self, mock_mp, mock_vision, mock_python, mock_ensure):
        MotionTracker(TrackingMode.POSE)
        mock_vision.PoseLandmarker.create_from_options.assert_called_once()

    def test_init_creates_hands_detector(self, mock_mp, mock_vision, mock_python, mock_ensure):
        MotionTracker(TrackingMode.HANDS)
        mock_vision.HandLandmarker.create_from_options.assert_called_once()

    def test_init_creates_face_mesh_detector(self, mock_mp, mock_vision, mock_python, mock_ensure):
        MotionTracker(TrackingMode.FACE_MESH)
        mock_vision.FaceLandmarker.create_from_options.assert_called_once()

    def test_mode_property_returns_current_mode(self, mock_mp, mock_vision, mock_python, mock_ensure):
        tracker = MotionTracker(TrackingMode.POSE)
        assert tracker.mode == TrackingMode.POSE

    def test_process_frame_returns_ndarray_same_shape(self, mock_mp, mock_vision, mock_python, mock_ensure):
        for mode in TrackingMode:
            mock_result = MagicMock()
            mock_result.pose_landmarks = []
            mock_result.hand_landmarks = []
            mock_result.face_landmarks = []
            mock_result.segmentation_masks = []
            detector_mock = MagicMock()
            detector_mock.detect_for_video.return_value = mock_result
            mock_vision.PoseLandmarker.create_from_options.return_value = detector_mock
            mock_vision.HandLandmarker.create_from_options.return_value = detector_mock
            mock_vision.FaceLandmarker.create_from_options.return_value = detector_mock

            frame = _blank_frame()
            tracker = MotionTracker(mode)
            result = tracker.process_frame(frame)

            assert isinstance(result, np.ndarray)
            assert result.shape == frame.shape

    def test_set_mode_closes_old_detector(self, mock_mp, mock_vision, mock_python, mock_ensure):
        tracker = MotionTracker(TrackingMode.POSE)
        old_detector = mock_vision.PoseLandmarker.create_from_options.return_value

        tracker.set_mode(TrackingMode.HANDS)

        old_detector.close.assert_called_once()
        mock_vision.HandLandmarker.create_from_options.assert_called_once()

    def test_set_mode_noop_if_same_mode(self, mock_mp, mock_vision, mock_python, mock_ensure):
        tracker = MotionTracker(TrackingMode.POSE)
        old_detector = mock_vision.PoseLandmarker.create_from_options.return_value

        tracker.set_mode(TrackingMode.POSE)

        old_detector.close.assert_not_called()
        mock_vision.PoseLandmarker.create_from_options.assert_called_once()

    def test_context_manager_closes_detector(self, mock_mp, mock_vision, mock_python, mock_ensure):
        with MotionTracker(TrackingMode.POSE):
            detector = mock_vision.PoseLandmarker.create_from_options.return_value

        detector.close.assert_called_once()


class TestDrawModeOverlay:
    def test_returns_ndarray(self):
        frame = _blank_frame()
        result = draw_mode_overlay(frame, TrackingMode.POSE)
        assert isinstance(result, np.ndarray)

    def test_output_shape_preserved(self):
        frame = _blank_frame()
        result = draw_mode_overlay(frame, TrackingMode.POSE)
        assert result.shape == frame.shape

    def test_overlay_modifies_frame(self):
        frame = _blank_frame()
        draw_mode_overlay(frame, TrackingMode.POSE)
        assert frame.max() > 0
