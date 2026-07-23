def parse_clip_dirname(name: str) -> tuple[str, str]:
    if "_" not in name:
        raise ValueError(f"clip dir name has no view label: {name!r}")
    di_id, view = name.split("_", 1)
    return di_id, canonical_view(view)


def canonical_view(view_name: str) -> str:
    return " ".join(view_name.split())


def base_view(view_name: str) -> str:
    return canonical_view(view_name).split(" ")[0]
