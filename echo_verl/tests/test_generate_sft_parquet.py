"""SFT row shaping: placeholder counts must match the media columns exactly.

verl's MultiTurnSFTDataset._build_messages asserts
`image_offset == len(images)` and `video_offset == len(videos)`, so a mismatch here
is a hard crash at the first training step.
"""
from echo_verl.generate_sft_parquet import build_sft_row


def _rec():
    return {
        "study_uuid": "st-TEST",
        "question": "Is there any abnormality?",
        "trajectory": {
            "overview": {"views": [{"view": "A4C", "frame": "a4c/5.png"},
                                   {"view": "PLAX", "frame": "plax/4.png"}]},
            "turns": [
                {"type": "tool", "name": "select_view", "args": {"view": "A4C"},
                 "obs": {"view": "A4C", "frames": ["a4c/0.png", "a4c/5.png"]}},
                {"type": "think", "text": "A4C shows normal RV."},
            ],
            "answer": "No significant abnormality.",
        },
    }


def test_overview_is_one_image_per_view_and_no_video():
    row = build_sft_row(_rec())
    assert row["videos"] == []
    # 2 overview views + 2 tool-observation frames
    assert row["images"][:2] == [{"image": "a4c/5.png"}, {"image": "plax/4.png"}]


def test_placeholder_counts_match_media_columns():
    row = build_sft_row(_rec())
    text = "".join(m["content"] for m in row["messages"])
    assert text.count("<video>") == len(row["videos"])
    assert text.count("<image>") == len(row["images"])


def test_tool_obs_frames_become_images():
    row = build_sft_row(_rec())
    assert row["images"][2:] == [{"image": "a4c/0.png"}, {"image": "a4c/5.png"}]
    tool_msgs = [m for m in row["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "<image><image>"


def test_all_message_content_is_string():
    # verl splits on the literal placeholders, so content must be str, not a list.
    row = build_sft_row(_rec())
    assert all(isinstance(m["content"], str) for m in row["messages"])
    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]


def test_user_turn_keeps_question_after_the_view_menu():
    row = build_sft_row(_rec())
    user = row["messages"][0]["content"]
    assert user.startswith("<image>")
    assert user.count("<image>") == 2          # one per view
    assert "Is there any abnormality?" in user
