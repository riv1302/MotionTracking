import cv2
import numpy as np

from motion_tracking.modes import TrackingMode
from motion_tracking.sidebar import build_sidebar
from motion_tracking.tracker import MotionTracker

WINDOW_NAME = "Motion Tracking"
QUIT_KEYS = {ord("q"), 27}


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0)")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with MotionTracker(TrackingMode.FACE_MESH) as tracker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                frame = tracker.process_frame(frame)
                sidebar = build_sidebar(frame.shape[0], tracker.mode)
                cv2.imshow(WINDOW_NAME, np.hstack([sidebar, frame]))

                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    break

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                new_mode = TrackingMode.from_key(key)
                if new_mode is not None:
                    tracker.set_mode(new_mode)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
