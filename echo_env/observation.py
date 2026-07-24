from dataclasses import dataclass, field


@dataclass
class FrameImg:
    image: object            # PIL.Image.Image at runtime
    view_name: str
    frame_index: int
    kind: str                # "thumbnail" | "preview" | "highres" | "crop"
    bbox: tuple | None = None


@dataclass
class Observation:
    tool: str
    frames: list = field(default_factory=list)
    text: str = ""
    ok: bool = True
    error: str | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @classmethod
    def failure(cls, tool: str, error: str) -> "Observation":
        return cls(tool=tool, frames=[], text=error, ok=False, error=error)
