import os
from dataclasses import dataclass
from echo_rl.data.views import parse_clip_dirname


def _num_stem(fname: str) -> int:
    return int(os.path.splitext(fname)[0])


@dataclass
class Clip:
    di_id: str
    view: str
    path: str
    frame_files: list

    @property
    def frame_count(self) -> int:
        return len(self.frame_files)

    def frame_path(self, idx: int) -> str:
        return os.path.join(self.path, self.frame_files[idx])


def study_dir(preprocessed_dir: str, study_uuid: str) -> str:
    return os.path.join(preprocessed_dir, study_uuid)


def index_study(study_dir_path: str) -> list:
    if not os.path.isdir(study_dir_path):
        raise FileNotFoundError(study_dir_path)
    clips = []
    for name in sorted(os.listdir(study_dir_path)):
        if not name.startswith("di-"):
            continue
        cpath = os.path.join(study_dir_path, name)
        if not os.path.isdir(cpath):
            continue
        frames = [f for f in os.listdir(cpath) if f.endswith(".png")]
        if not frames:
            continue
        frames.sort(key=_num_stem)
        di_id, view = parse_clip_dirname(name)
        clips.append(Clip(di_id=di_id, view=view, path=cpath, frame_files=frames))
    return clips
