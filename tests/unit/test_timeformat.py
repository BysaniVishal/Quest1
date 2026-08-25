import pytest

from dialogue_frame_finder.timeformat import format_timestamp

pytestmark = pytest.mark.unit


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00.000"


def test_format_timestamp_seconds_and_milliseconds():
    assert format_timestamp(1.5) == "00:00:01.500"


def test_format_timestamp_minute_boundary():
    assert format_timestamp(61.001) == "00:01:01.001"


def test_format_timestamp_hour_boundary():
    assert format_timestamp(3661.001) == "01:01:01.001"


def test_format_timestamp_rounds_to_nearest_millisecond():
    assert format_timestamp(1.2345) == "00:00:01.234"
    assert format_timestamp(1.2344) == "00:00:01.234"


def test_format_timestamp_rounding_carries_into_next_second():
    assert format_timestamp(0.9996) == "00:00:01.000"


def test_format_timestamp_rounding_carries_into_next_minute():
    assert format_timestamp(59.9996) == "00:01:00.000"


def test_format_timestamp_rounding_carries_into_next_hour():
    assert format_timestamp(3599.9996) == "01:00:00.000"


def test_format_timestamp_large_hour_values():
    assert format_timestamp(7384.25) == "02:03:04.250"


def test_format_timestamp_negative_raises():
    with pytest.raises(ValueError):
        format_timestamp(-0.1)
