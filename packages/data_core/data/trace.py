import re

# Per-view section headers in the source thinking traces. The original pattern only
# accepted the numbered-and-bold form (`#### 1. **A4C View**), which matched just 4% of
# train_vqa_with_thinking.jsonl -- so 2283 of 2382 built trajectories ended up with ZERO
# tool turns and the cold start taught the model never to call the echo tool. The traces
# do name their views, just in several shapes:
#     #### 1. **A4C View**
#     #### View: A3C
#     #### PSAX Zoomed Out
#     #### PSAX Zoomed Out (Medium Importance)
# This generalized pattern (optional numbering, optional "View:" label, optional bold,
# anchored per line) covers 93.5% of the source records.
_HEADER = re.compile(r"^####\s*(?:\d+\.\s*)?(?:View:\s*)?\*{0,2}\s*(.+?)\s*\*{0,2}\s*$", re.M)

# Trailing parenthetical qualifiers the traces attach to a view name, e.g.
# "PSAX Zoomed Out (Medium Importance)".
_QUALIFIER = re.compile(r"\s*\([^)]*\)\s*$")


def parse_thinking(thinking: str) -> list:
    parts = _HEADER.split(thinking or "")
    # parts = [pre, view1, body1, view2, body2, ...]
    out = []
    for i in range(1, len(parts) - 1, 2):
        out.append((strip_view_qualifier(parts[i]), parts[i + 1].strip()))
    return out


def strip_view_qualifier(view: str) -> str:
    """'PSAX Zoomed Out (Medium Importance)' -> 'PSAX Zoomed Out'."""
    return _QUALIFIER.sub("", (view or "").strip()).strip()


def findings_text(body: str) -> str:
    m = re.search(r"Clinical Findings:\*\*(.*?)(?:\*\*Implications|$)", body, re.S)
    return (m.group(1) if m else body).strip()
