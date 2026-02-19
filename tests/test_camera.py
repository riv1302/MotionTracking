import numpy as np

from motion_tracking.camera import TrackingMode, draw_mode_overlay, process_frame


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


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestProcessFrame:
    def test_returns_ndarray_for_each_mode(self):
        frame = _blank_frame()
        for mode in TrackingMode:
            result = process_frame(frame, mode)
            assert isinstance(result, np.ndarray)

    def test_output_shape_preserved(self):
        frame = _blank_frame()
        for mode in TrackingMode:
            result = process_frame(frame, mode)
            assert result.shape == frame.shape


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
