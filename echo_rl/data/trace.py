import re

_HEADER = re.compile(r"####\s*\d+\.\s*\*\*(.+?)\*\*")


def parse_thinking(thinking: str) -> list:
    parts = _HEADER.split(thinking)
    # parts = [pre, view1, body1, view2, body2, ...]
    out = []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i].strip(), parts[i + 1].strip()))
    return out


def findings_text(body: str) -> str:
    m = re.search(r"Clinical Findings:\*\*(.*?)(?:\*\*Implications|$)", body, re.S)
    return (m.group(1) if m else body).strip()
