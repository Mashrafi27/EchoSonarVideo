"""Guard the sample and answer provenance of the small DeepEyes baseline."""
import pytest

from eval.run_deepeyes_baseline import (
    TYPES, SINGLE_FRAME_SYSTEM_PROMPT, extract_answer, messages_for,
    sample_records, sample_single_frame,
)


def test_single_frame_samples_clip_not_just_preview_and_keeps_reference_out_of_prompt(tmp_path):
    import random

    views = []
    for name in ('A4C', 'PLAX'):
        clip = tmp_path / name
        clip.mkdir()
        for index in range(12):
            (clip / f'{index}.png').touch()
        views.append(dict(view=name, frame=str(clip / '6.png'), frame_count=12))
    rec = dict(study_uuid='study', question='Question sentinel', answer='Reference sentinel',
               overview=dict(views=views))
    selected = sample_single_frame([rec], 0)
    assert selected == sample_single_frame([rec], 0)
    assert len(selected) == len(selected[0]['overview']['views']) == 1
    rng = random.Random(0)
    rng.choice([rec])
    expected_view = rng.choice(views)
    expected_frame = rng.randrange(12)
    frame = selected[0]['overview']['views'][0]
    assert frame['view'] == expected_view['view']
    assert frame['frame_index'] == expected_frame
    assert frame['frame'].endswith(f'/{expected_frame}.png')
    assert frame['frame'] != expected_view['frame']
    assert len(rec['overview']['views']) == 2
    messages = messages_for(selected[0], SINGLE_FRAME_SYSTEM_PROMPT)
    assert sum(item['type'] == 'image' for item in messages[1]['content']) == 1
    assert 'Reference sentinel' not in str(messages)
    assert 'Question sentinel' in str(messages)


def test_sample_is_deterministic_balanced_and_has_distinct_studies():
    records = [dict(study_uuid=f'study_{i}', question_type=kind)
               for i in range(30) for kind in TYPES]
    sample = sample_records(records, 2, 0)
    assert sample == sample_records(records, 2, 0)
    assert len(sample) == len({r['study_uuid'] for r in sample}) == 10
    assert all(sum(r['question_type'] == k for r in sample) == 2 for k in TYPES)


def test_single_frame_batch_balances_types_without_reusing_studies(tmp_path):
    for i in range(4):
        (tmp_path / f'{i}.png').touch()
    views = [dict(view='A4C', frame=str(tmp_path / '2.png'), frame_count=4)]
    records = [dict(study_uuid=f'study_{i}', question_type=kind, overview=dict(views=views))
               for i in range(30) for kind in TYPES]
    selected = sample_single_frame(records, seed=1, per_type=2)
    assert selected == sample_single_frame(records, seed=1, per_type=2)
    assert len(selected) == len({r['study_uuid'] for r in selected}) == 10
    assert all(sum(r['question_type'] == kind for r in selected) == 2 for kind in TYPES)
    assert all(len(r['overview']['views']) == 1 for r in selected)
    assert all(0 <= r['overview']['views'][0]['frame_index'] < 4 for r in selected)


def test_insufficient_studies_fails_instead_of_silently_shrinking_sample():
    with pytest.raises(ValueError):
        sample_records([dict(study_uuid='one', question_type=k) for k in TYPES], 2, 0)


def test_prompt_contains_question_and_views_but_no_reference_answer():
    rec = {'question': 'Question sentinel', 'answer': 'Reference sentinel',
           'overview': {'views': [{'view': 'A4C', 'frame': 'private_path.png'}]}}
    messages = messages_for(rec)
    assert 'Question sentinel' in str(messages)
    assert 'A4C' in str(messages)
    assert 'Reference sentinel' not in str(messages)
    assert 'private_path' not in str(messages)


@pytest.mark.parametrize('text, expected', [
    ('<think>maybe</think><answer>Yes</answer>', 'Yes'),
    ('<think>unfinished', None),
    ('<think>done</think><tool_call>{}</tool_call>', None),
    ('<think>done</think>No', 'No'),
])
def test_answer_extraction_does_not_score_reasoning_or_tool_requests(text, expected):
    assert extract_answer(text) == expected
