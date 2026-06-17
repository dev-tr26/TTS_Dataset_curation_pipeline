"""
src/ingestion/ingest.py
Stage 1 — Download YouTube videos, extract metadata, track provenance.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Iterator

import yt_dlp

from tts_guj_eng_sarvam_dataset.src.schemas import Language, VideoInput, VideoMetadata

logger = logging.getLogger(__name__)


def load_video_inputs(csv_path: str | Path) -> list[VideoInput]:
    """Parse videos.csv → list of VideoInput."""
    rows: list[VideoInput] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(VideoInput(
                    video_url=row["video_url"].strip(),
                    language=Language(row["language"].strip()),
                ))
            except Exception as exc:
                logger.warning("Skipping malformed row %s: %s", row, exc)
    logger.info("Loaded %d video inputs from %s", len(rows), csv_path)
    return rows


def _extract_video_id(url: str) -> str:
    """Extract 11-char YouTube video id from any YouTube URL."""
    import re
    patterns = [
        r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def download_video(
    video_input: VideoInput,
    output_dir: Path,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> VideoMetadata:
    """
    Download audio for a single YouTube video using yt-dlp.
    Retries on transient failures. Returns VideoMetadata with paths set.
    """
    video_id = _extract_video_id(video_input.video_url)
    audio_path = output_dir / f"{video_id}.wav"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
         "ffmpeg_location": r"C:\Users\trang\Downloads\ffmpeg-8.1.1\bin\ffprobe.exe",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
        "writeinfojson": True,
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[%s] Download attempt %d/%d", video_id, attempt, max_retries)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_input.video_url, download=True)

            # Read back metadata from yt-dlp info dict
            meta = VideoMetadata(
                video_id=video_id,
                title=info.get("title", "unknown"),
                channel=info.get("uploader", "unknown"),
                duration_seconds=float(info.get("duration", 0)),
                url=video_input.video_url,
                language=video_input.language,
                raw_audio_path=str(audio_path),
                ingestion_status="success",
            )
            _save_provenance(meta, output_dir)
            logger.info("[%s] Downloaded successfully (%.1fs)", video_id, meta.duration_seconds)
            return meta

        except Exception as exc:
            last_exc = exc
            logger.warning("[%s] Attempt %d failed: %s", video_id, attempt, exc)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    # All retries exhausted
    logger.error("[%s] All %d download attempts failed: %s", video_id, max_retries, last_exc)
    return VideoMetadata(
        video_id=video_id,
        title="unknown",
        channel="unknown",
        duration_seconds=0.0,
        url=video_input.video_url,
        language=video_input.language,
        ingestion_status="failed",
        error_message=str(last_exc),
    )


def _save_provenance(meta: VideoMetadata, output_dir: Path) -> None:
    """Persist provenance JSON alongside the audio file."""
    prov_path = output_dir / f"{meta.video_id}_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, indent=2, default=str)


def ingest_all(
    csv_path: str | Path,
    output_dir: str | Path,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[VideoMetadata]:
    """Main entry point for Stage 1."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_video_inputs(csv_path)
    results: list[VideoMetadata] = []

    for vi in inputs:
        meta = download_video(vi, output_dir, max_retries, retry_delay)
        results.append(meta)

    success = sum(1 for r in results if r.ingestion_status == "success")
    logger.info("Ingestion complete: %d/%d succeeded", success, len(results))
    return results