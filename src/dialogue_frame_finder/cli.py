"""CLI entry point: python -m dialogue_frame_finder <url> "<target dialogue>"."""

import argparse
import sys

from .captions import CaptionSource
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
    parser.add_argument(
        "--use-captions", action="store_true",
        help="Optional latency optimization: if the source has usable caption/subtitle "
             "tracks, use them to coarsely localize the target dialogue, then run ASR on "
             "just a short local audio window instead of the whole video. Captions are "
             "NEVER used for the final timestamp/frame -- real ASR still produces the "
             "word-level timing; captions only narrow which audio gets transcribed. Off "
             "by default; falls back to the full-video ASR path automatically whenever "
             "captions are unavailable or don't yield a confident local match.",
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

    caption_source = CaptionSource() if args.use_captions else None

    result = run_pipeline(
        args.url, args.dialogue, args.output_dir,
        media_resolver=media_resolver, caption_source=caption_source,
    )
    record = result.output

    frame_display = record.frame if record.frame is not None else "N/A (not available for this source/timestamp)"

    print(f"Status    : {record.status.value}")
    print(f"Timestamp : {record.timestamp}")
    print(f"Frame     : {frame_display}")
    print(f'Text      : "{record.text}"')
    print(f"Image     : {record.image_path}")
    if args.use_captions:
        print(f"ASR source: {result.transcript_source}")
    if record.confidence_score is not None:
        print(f"Confidence: {record.confidence_score:.3f}")

    return 0 if record.status.value != "NO_CONFIDENT_MATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
