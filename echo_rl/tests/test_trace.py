from echo_rl.data.trace import parse_thinking, findings_text

SAMPLE = (
    "### Report\n\n"
    "#### 1. **PLAX Standard View**\n"
    "- **Detected Structures:** LA, LV\n"
    "- **Clinical Findings:**\n  - LV normal.\n"
    "- **Implications:**\n  - none\n\n"
    "#### 2. **A4C View**\n"
    "- **Clinical Findings:**\n  - RV normal.\n"
)


def test_parse_thinking_sections():
    secs = parse_thinking(SAMPLE)
    assert [v for v, _ in secs] == ["PLAX Standard View", "A4C View"]
    assert "LV normal" in secs[0][1]


def test_findings_text():
    secs = parse_thinking(SAMPLE)
    ft = findings_text(secs[0][1])
    assert "LV normal" in ft and "Implications" not in ft


def test_parse_thinking_empty():
    assert parse_thinking("no headers here") == []


def test_parses_unnumbered_and_labelled_headers():
    # The source traces use several header shapes; only the numbered-bold one used to
    # match, which starved 96% of cold-start trajectories of tool turns.
    th = (
        "### Aortic Root Assessment\n"
        "#### View: A3C\n"
        "**Clinical Findings:**\n- normal\n"
        "#### PSAX Zoomed Out\n"
        "**Clinical Findings:**\n- mild TR\n"
        "#### 3. **A4C View**\n"
        "**Clinical Findings:**\n- normal RV\n"
    )
    sections = parse_thinking(th)
    assert [v for v, _ in sections] == ["A3C", "PSAX Zoomed Out", "A4C View"]
    assert "normal RV" in sections[2][1]


def test_strips_trailing_qualifier():
    th = "#### PSAX Zoomed Out (Medium Importance)\n**Clinical Findings:**\n- x\n"
    assert [v for v, _ in parse_thinking(th)] == ["PSAX Zoomed Out"]


def test_none_thinking_is_empty():
    assert parse_thinking(None) == []
