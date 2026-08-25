import pytest

from dialogue_frame_finder.selection import select_earliest_valid
from dialogue_frame_finder.verification import VerificationResult

pytestmark = pytest.mark.unit


def _result(start, score, valid=True):
    return VerificationResult(
        window_start=0,
        window_end=0,
        alignment=[],
        lexical_similarity=score,
        coverage=score,
        contiguity=score,
        score=score,
        valid=valid,
        first_word_position=int(start * 10),
        first_word_start=start,
        first_word_end=start + 0.3,
    )


def test_select_earliest_valid_picks_earliest_time_not_highest_score():
    # user's worked example: 10:15 (lower score) must beat 47:12 (higher score)
    earlier_weaker = _result(start=615.30, score=0.91)
    later_stronger = _result(start=2830.10, score=0.99)
    chosen, others = select_earliest_valid([later_stronger, earlier_weaker])
    assert chosen is earlier_weaker
    assert others == [later_stronger]


def test_select_earliest_valid_ignores_invalid_results():
    invalid_earlier = _result(start=10.0, score=0.9, valid=False)
    valid_later = _result(start=20.0, score=0.7, valid=True)
    chosen, others = select_earliest_valid([invalid_earlier, valid_later])
    assert chosen is valid_later
    assert others == []


def test_select_earliest_valid_returns_none_when_nothing_valid():
    chosen, others = select_earliest_valid([_result(10.0, 0.9, valid=False)])
    assert chosen is None
    assert others == []


def test_select_earliest_valid_empty_input():
    chosen, others = select_earliest_valid([])
    assert chosen is None
    assert others == []


def test_select_earliest_valid_retains_remaining_candidates_sorted_by_time():
    a = _result(start=5.0, score=0.8)
    b = _result(start=15.0, score=0.95)
    c = _result(start=25.0, score=0.7)
    chosen, others = select_earliest_valid([c, b, a])
    assert chosen is a
    assert others == [b, c]
