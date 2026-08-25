"""CLI entry point: python -m dialogue_frame_finder <url> "<target dialogue>"."""

import argparse
import sys

from .media_resolver import MediaResolver
from .pipeline import run_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dialogue_frame_finder",
        description="Locate the exact video frame where a spoken dialogue begins.",
    )
    parser.add_argument("url", help="Publicly accessible video URL (e.g. YouTube, OK.ru)")
    parser.add_argument("dialogue", help="Target spoken dialogue text to locate")
    parser.add_argument("--output-dir", default="outputs", help="Directory for downloaded media and the extracted frame image")
    parser.add_argument(
        "--format", default=None,
        help="yt-dlp format selector, e.g. 'worst' for a much smaller/faster download. "
             "Default is yt-dlp's own best-quality pick, which can be large (a "
             "multi-hour source may be 1GB+) and more exposed to a flaky source's "
             "connection resets (e.g. OK.ru) simply by taking longer. Resolution "
             "doesn't affect correctness, so 'worst' is a good first thing to try if "
             "downloads keep failing or disk space is tight.",
    )
    parser.add_argument(
        "--retries", type=int, default=None,
        help="Max media-resolution attempts before giving up (default: 6). Raise this "
             "for a source with known intermittent connection resets, like OK.ru.",
    )
    args = parser.parse_args(argv)

    media_resolver = None
    if args.format is not None or args.retries is not None:
        kwargs = {}
        if args.format is not None:
            kwargs["format_selector"] = args.format
        if args.retries is not None:
            kwargs["max_attempts"] = args.retries
        media_resolver = MediaResolver(args.output_dir, **kwargs)

    result = run_pipeline(args.url, args.dialogue, args.output_dir, media_resolver=media_resolver)
    record = result.output

    frame_display = record.frame if record.frame is not None else "N/A (not available for this source/timestamp)"

    print(f"Status    : {record.status.value}")
    print(f"Timestamp : {record.timestamp}")
    print(f"Frame     : {frame_display}")
    print(f'Text      : "{record.text}"')
    print(f"Image     : {record.image_path}")
    if record.confidence_score is not None:
        print(f"Confidence: {record.confidence_score:.3f}")

    return 0 if record.status.value != "NO_CONFIDENT_MATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
