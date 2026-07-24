from dataclasses import dataclass
from echo_rl.data.studies import index_study, study_dir
from echo_rl.data.views import canonical_view, base_view


@dataclass
class ViewEntry:
    view_name: str
    clip: object  # echo_rl.data.studies.Clip

    @property
    def frame_count(self) -> int:
        return self.clip.frame_count


class StudyManifest:
    def __init__(self, views):
        # keep a stable, sorted order
        self.views = sorted(views, key=lambda v: v.view_name)
        self._by_lower = {v.view_name.lower(): v for v in self.views}

    def view_names(self):
        return [v.view_name for v in self.views]

    def overview(self, limit: int):
        return self.views[:limit]

    def resolve(self, view_name: str):
        if not view_name:
            return None
        q = view_name.strip()
        # 1. exact case-insensitive
        hit = self._by_lower.get(q.lower())
        if hit:
            return hit
        # 2. canonical_view match
        qc = canonical_view(q)
        for v in self.views:
            if canonical_view(v.view_name) == qc:
                return v
        # 3. base_view fallback: prefer a clip whose full name IS the base query
        qb = base_view(q)
        base_matches = [v for v in self.views if base_view(v.view_name) == qb]
        for v in base_matches:
            if v.view_name.lower() == qb.lower():
                return v
        return base_matches[0] if base_matches else None


def build_manifest(preprocessed_dir: str, study_uuid: str) -> StudyManifest:
    sdir = study_dir(preprocessed_dir, study_uuid)
    clips = index_study(sdir)
    views = [ViewEntry(view_name=c.view, clip=c) for c in clips]
    return StudyManifest(views)
