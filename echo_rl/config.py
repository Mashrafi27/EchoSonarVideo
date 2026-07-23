import os
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)                       # .../EchoSonarVideo
_PARENT = os.path.dirname(_PROJ)                     # .../project


@dataclass
class Config:
    preprocessed_dir: str
    vqa_train: str
    vqa_test: str
    gold_dir: str
    out_dir: str
    n_overview_views: int = 19
    n_preview_frames: int = 5
    n_highres_frames: int = 8
    seed: int = 0

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            preprocessed_dir=os.environ.get(
                "ECHO_PREPROCESSED_DIR", os.path.join(_PARENT, "preprocessed_data")),
            vqa_train=os.environ.get(
                "ECHO_VQA_TRAIN", os.path.join(_PROJ, "Archive 2 (1)", "train_vqa_with_thinking.jsonl")),
            vqa_test=os.environ.get(
                "ECHO_VQA_TEST", os.path.join(_PROJ, "Archive 2 (1)", "test_vqa.jsonl")),
            gold_dir=os.environ.get(
                "ECHO_GOLD_DIR", os.path.join(_PARENT, "output_with_labels", "output")),
            out_dir=os.environ.get("ECHO_OUT_DIR", os.path.join(_PROJ, "build")),
        )
