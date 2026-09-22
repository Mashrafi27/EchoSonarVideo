"""Guard the sample and answer provenance of the small DeepEyes baseline."""
import pytest

from eval.run_deepeyes_baseline import TYPES, extract_answer, messages_for, sample_records


def test_sample_is_deterministic_balanced_and_has_distinct_studies():
    records = [dict(study_uuid=f'study_{i}', question_type=kind)
               for i in range(30) for kind in TYPES]
    sample = sample_records(records, 2, 0)
    assert sample == sample_records(records, 2, 0)
    assert len(sample) == len({r['study_uuid'] for r in sample}) == 10
    assert all(sum(r['question_type'] == k for r in sample) == 2 for k in TYPES)


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
