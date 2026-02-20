import cv2
import numpy as np

from motion_tracking.modes import TrackingMode

# ── Sidebar constants ─────────────────────────────────────────────────────────

SIDEBAR_WIDTH = 220

_C_ACTIVE = (0, 255, 0)      # bright green  — active mode
_C_DIM = (0, 100, 0)         # dark green    — inactive modes
_C_HEADER = (0, 180, 0)      # mid green     — section headers
_C_TITLE = (160, 160, 160)   # light grey    — app title
_C_SEP = (0, 60, 0)          # very dark green — separators

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_MARGIN = 12


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _put(panel: np.ndarray, text: str, y: int, color: tuple,
         scale: float = 0.45, thickness: int = 1) -> None:
    cv2.putText(panel, text, (_MARGIN, y), _FONT, scale, color,
                thickness, cv2.LINE_AA)


def _draw_separator(panel: np.ndarray, y: int) -> None:
    cv2.line(panel, (_MARGIN, y), (SIDEBAR_WIDTH - _MARGIN, y), _C_SEP, 1)


def _draw_title(panel: np.ndarray) -> None:
    _put(panel, "MOTION", 18, _C_TITLE, scale=0.5)
    _put(panel, "TRACKING", 34, _C_TITLE, scale=0.5)


def _draw_mode_section(panel: np.ndarray, mode: TrackingMode,
                       y_start: int) -> None:
    _put(panel, "MODE", y_start, _C_HEADER, scale=0.4, thickness=1)
    for i, m in enumerate(TrackingMode):
        y = y_start + 20 + i * 26
        active = m == mode
        color = _C_ACTIVE if active else _C_DIM
        prefix = ">" if active else " "
        _put(panel, f"{prefix} [{m.value}] {m.display_name}", y, color)


def _draw_controls_section(panel: np.ndarray, y_start: int) -> None:
    _put(panel, "CONTROLS", y_start, _C_HEADER, scale=0.4, thickness=1)
    controls = [
        ("1 / 2 / 3", "Mode"),
        ("Q / Esc", "Quit"),
    ]
    for i, (key, action) in enumerate(controls):
        y = y_start + 20 + i * 22
        _put(panel, key, y, _C_DIM)
        _put(panel, f"               <- {action}", y, _C_DIM, scale=0.38)


# ── Public API ────────────────────────────────────────────────────────────────

def build_sidebar(height: int, mode: TrackingMode) -> np.ndarray:
    panel = np.zeros((height, SIDEBAR_WIDTH, 3), dtype=np.uint8)
    sep1_y = 50
    mode_section_y = sep1_y + 15
    sep2_y = mode_section_y + 20 + len(TrackingMode) * 26 + 8
    controls_y = sep2_y + 15

    _draw_title(panel)
    _draw_separator(panel, sep1_y)
    _draw_mode_section(panel, mode, mode_section_y)
    _draw_separator(panel, sep2_y)
    _draw_controls_section(panel, controls_y)
    return panel
