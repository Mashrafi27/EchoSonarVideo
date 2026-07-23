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
