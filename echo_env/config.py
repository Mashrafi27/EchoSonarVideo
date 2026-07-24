import os
from dataclasses import dataclass

# repo root is two levels below the shared data parent, mirroring echo_rl.config
_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class EnvConfig:
    preprocessed_dir: str = ""
    n_preview_frames: int = 5
    n_highres_frames: int = 8
    preview_max_side: int = 168
    highres_max_side: int = 336
    n_overview_views: int = 19
    min_crop_side: int = 28
    max_aspect: float = 100.0
    max_tool_calls: int = 8
    max_calls_per_turn: int = 3
    max_total_frames: int = 32
    seed: int = 0

    @classmethod
    def from_env(cls) -> "EnvConfig":
        def _i(name, default):
            return int(os.environ.get(name, default))
        return cls(
            preprocessed_dir=os.environ.get(
                "ECHO_PREPROCESSED_DIR", os.path.join(_PARENT, "preprocessed_data")),
            n_preview_frames=_i("ECHO_N_PREVIEW_FRAMES", 5),
            n_highres_frames=_i("ECHO_N_HIGHRES_FRAMES", 8),
            preview_max_side=_i("ECHO_PREVIEW_MAX_SIDE", 168),
            highres_max_side=_i("ECHO_HIGHRES_MAX_SIDE", 336),
            n_overview_views=_i("ECHO_N_OVERVIEW_VIEWS", 19),
            min_crop_side=_i("ECHO_MIN_CROP_SIDE", 28),
            max_aspect=float(os.environ.get("ECHO_MAX_ASPECT", 100.0)),
            max_tool_calls=_i("ECHO_MAX_TOOL_CALLS", 8),
            max_calls_per_turn=_i("ECHO_MAX_CALLS_PER_TURN", 3),
            max_total_frames=_i("ECHO_MAX_TOTAL_FRAMES", 32),
            seed=_i("ECHO_SEED", 0),
        )
