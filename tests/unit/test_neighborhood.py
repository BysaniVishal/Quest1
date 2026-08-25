import pytest

from dialogue_frame_finder.neighborhood import extract_window, merge_windows

pytestmark = pytest.mark.unit


def test_extract_window_basic_case():
    # anchor is the target's first word (target_start=0), target has 5 words,
    # transcript is long enough that clipping never kicks in
    start, end = extract_window(
        anchor_position=100, anchor_target_start=0, target_len=5,
        transcript_len=1000, tolerance=3,
    )
    assert start == 97
    assert end == 107  # 100 + 5 - 1 + 3


def test_extract_window_anchor_mid_target():
    # anchor is the target's last word (target_start=4 of a 5-word target)
    start, end = extract_window(
        anchor_position=100, anchor_target_start=4, target_len=5,
        transcript_len=1000, tolerance=3,
    )
    expected_target_start = 100 - 4
    assert start == expected_target_start - 3
    assert end == 100 + 3


def test_extract_window_clips_at_transcript_start():
    start, end = extract_window(
        anchor_position=1, anchor_target_start=0, target_len=5,
        transcript_len=1000, tolerance=3,
    )
    assert start == 0


def test_extract_window_clips_at_transcript_end():
    start, end = extract_window(
        anchor_position=97, anchor_target_start=4, target_len=5,
        transcript_len=100, tolerance=3,
    )
    assert end == 99  # transcript_len - 1


def test_merge_windows_overlapping_intervals_combine():
    merged = merge_windows([(0, 10), (5, 15)])
    assert merged == [(0, 15)]


def test_merge_windows_adjacent_intervals_combine():
    merged = merge_windows([(0, 10), (11, 20)])
    assert merged == [(0, 20)]


def test_merge_windows_non_overlapping_stay_separate():
    merged = merge_windows([(0, 5), (20, 25)])
    assert merged == [(0, 5), (20, 25)]


def test_merge_windows_empty_input():
    assert merge_windows([]) == []


def test_merge_windows_sorts_unordered_input():
    merged = merge_windows([(20, 25), (0, 5)])
    assert merged == [(0, 5), (20, 25)]


def test_merge_windows_fully_nested_interval_absorbed():
    merged = merge_windows([(0, 20), (5, 10)])
    assert merged == [(0, 20)]
