import enum


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
