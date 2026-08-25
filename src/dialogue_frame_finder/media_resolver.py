"""MediaResolver: the ONLY provider-specific component in the pipeline.

Everything downstream (audio extraction, ASR, indexed retrieval,
verification, onset refinement, frame mapping) operates on a plain resolved
local media file and is identical for every provider (design.docx section
12). Provider quirks discovered during the Phase 0 feasibility spike against
the real supplied OK.ru example and a real YouTube video are captured here,
not scattered through the rest of the codebase:

- OK.ru: the initial page fetch is intermittently flaky
  (ConnectionResetError) -- needs retry with backoff, not a one-shot
  request. Its CDN's TLS chain is not trusted by ffmpeg's bundled gnutls
  even though yt-dlp's own Python layer accepts it as a warning -- needs
  `-tls_verify 0` passed to yt-dlp's ffmpeg-backed downloader.
- YouTube: current extraction needs a JS runtime (Node, already present in
  the dev environment, works via `js_runtimes: ["node"]`) and, as of this
  writing, standard extraction hits YouTube's bot-check; the
  `player_client=android,web_safari` extractor args resolved it without
  needing cookies during the Phase 0 spike. This is a known, currently
  widespread yt-dlp/YouTube friction point that could change without
  notice -- it is isolated here specifically so a future fix only touches
  this file.

yt_dlp's `YoutubeDL` class is injected (default: the real one) so this can
be unit-tested without real network access -- see test_media_resolver.py.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse


class MediaResolutionError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedMedia:
    local_path: str
    duration: Optional[float]
    provider: str


def detect_provider(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "ok.ru" in host or "odnoklassniki" in host:
        return "okru"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    return "generic"


def _provider_ydl_opts(provider: str) -> Dict[str, Any]:
    if provider == "okru":
        return {
            # OK.ru's CDN presents a TLS chain that common HTTP clients
            # don't trust by default. ffmpeg_i's -tls_verify 0 covers
            # ffmpeg-backed downloads; nocheckcertificate covers yt-dlp's
            # own Python-side (requests/urllib3) downloads -- which format
            # gets used, and therefore which of the two actually applies,
            # can vary by request/format, so both are needed together
            # (found the hard way: -tls_verify 0 alone still left
            # "unable to download video data: SSL: CERTIFICATE_VERIFY_FAILED"
            # failing for some formats).
            "downloader_args": {"ffmpeg_i": ["-tls_verify", "0"]},
            "nocheckcertificate": True,
            "retries": 5,
            "socket_timeout": 30,
        }
    if provider == "youtube":
        return {
            # yt-dlp's Python API (unlike its --js-runtimes CLI flag) wants
            # a dict of {runtime: config}, not a list -- found the hard way
            # while running Phase 8's real E2E validation.
            "js_runtimes": {"node": {}},
            "extractor_args": {"youtube": {"player_client": ["android", "web_safari"]}},
        }
    return {}


class MediaResolver:
    def __init__(
        self,
        output_dir: Path,
        ydl_class: Optional[Callable] = None,
        # Default of 6 (not 3) reflects real-world OK.ru behavior observed
        # during Phase 8 E2E validation: its initial page fetch resets the
        # connection intermittently, and 3 attempts were not always enough
        # even with backoff -- see also the Phase 0 feasibility spike.
        max_attempts: int = 6,
        retry_backoff_seconds: float = 5.0,
        format_selector: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        # Optional yt-dlp format selector (e.g. "worst", "best[height<=480]")
        # to bound bandwidth/storage. None keeps yt-dlp's own default
        # ("best"). Resolution is explicitly not something the rest of the
        # pipeline depends on (design.docx's resolution-independence
        # requirement), so this is safe to lower without affecting
        # correctness downstream.
        self.format_selector = format_selector
        if ydl_class is None:
            import yt_dlp
            ydl_class = yt_dlp.YoutubeDL
        self._ydl_class = ydl_class

    def resolve(self, url: str) -> ResolvedMedia:
        provider = detect_provider(url)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            **_provider_ydl_opts(provider),
        }
        if self.format_selector:
            opts["format"] = self.format_selector

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self._ydl_class(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    local_path = ydl.prepare_filename(info)
                    return ResolvedMedia(
                        local_path=local_path,
                        duration=info.get("duration"),
                        provider=provider,
                    )
            except Exception as exc:  # yt_dlp raises its own exception types
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(self.retry_backoff_seconds)

        raise MediaResolutionError(
            f"failed to resolve {url} after {self.max_attempts} attempts"
        ) from last_error
