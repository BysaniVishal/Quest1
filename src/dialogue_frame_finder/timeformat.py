"""Seconds -> HH:MM:SS.sss formatting.

Rounds to whole milliseconds as an integer FIRST, then derives hours/
minutes/seconds via integer arithmetic -- avoids floating-point edge cases
(e.g. 59.9996s must roll over into the next minute as 00:01:00.000, not
render as "00:00:60.000" or drift from float rounding).
"""


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError(f"cannot format a negative timestamp: {seconds}")
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
