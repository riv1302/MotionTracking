import cv2
import numpy as np

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
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                 # Wrist to pinky base
])


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_landmarks(
    frame: np.ndarray,
    landmarks: list,
    connections: frozenset[tuple[int, int]] | None = None,
    point_color: tuple = (0, 255, 0),
    line_color: tuple = (0, 200, 0),
    point_radius: int = 2,
    line_thickness: int = 1,
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
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), point_radius, point_color, -1)


def _draw_neural_links(
    frame: np.ndarray,
    landmarks: list,
    n_neighbors: int = 3,
    color: tuple = (0, 200, 0),
    thickness: int = 1,
) -> None:
    h, w = frame.shape[:2]
    coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks])  # (N, 2)

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    sq_dist = (diff ** 2).sum(axis=2)                           # (N, N)
    np.fill_diagonal(sq_dist, np.inf)
    nearest = np.argsort(sq_dist, axis=1)[:, :n_neighbors]      # (N, n_neighbors)

    for i, neighbors in enumerate(nearest):
        for j in neighbors:
            cv2.line(
                frame,
                (int(coords[i, 0]), int(coords[i, 1])),
                (int(coords[j, 0]), int(coords[j, 1])),
                color,
                thickness,
            )
