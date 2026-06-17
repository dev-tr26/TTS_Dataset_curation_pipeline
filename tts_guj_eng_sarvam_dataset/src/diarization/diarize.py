"""
src/diarization/diarize.py
Stage 4 — Sarvam BATCH API for diarization (NOT the real-time endpoint).

Root cause of the previous error:
    "Diarization is not supported in the real-time API.
     Please use the batch API for diarization."

Fix:
    Use the Sarvam Python SDK's `speech_to_text_job.create_job()` workflow:
      1. create_job()  — submit the job with with_diarization=True
      2. upload_files() — upload the WAV file
      3. start()       — trigger processing
      4. wait_until_complete() — poll until done
      5. download_outputs()    — fetch the JSON transcript

Stage 5 — Convert diarization output into TTS-ready candidate segments.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf

from tts_guj_eng_sarvam_dataset.src.schemas import (
    CandidateSegment,
    DiarizationSegment,
    Language,
)

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

# ─── Chunk splitter (kept for large files >2h) ───────────────────────────────

def split_for_diarization(
    audio_path: str | Path,
    output_dir: Path,
    chunk_seconds: int = 3600,   # Batch API supports up to 2h; use full file
) -> list[tuple[Path, float]]:
    """
    Split audio into chunks if > chunk_seconds.
    For the Sarvam Batch API, max file size is 2 hours, so this is rarely needed.
    Returns list of (chunk_path, offset_seconds).
    """
    import soundfile as sf
    
    audio_path = Path(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    info = sf.info(str(audio_path))
    duration = info.frames / info.samplerate
    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]
    
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    chunk_samples = chunk_seconds * sr
    chunks: list[tuple[Path, float]] = []

    for i, start in enumerate(range(0, len(y), chunk_samples)):
        end = min(start + chunk_samples, len(y))
        chunk_path = output_dir / f"{audio_path.stem}_chunk{i}.wav"
        sf.write(str(chunk_path), y[start:end], sr)
        chunks.append((chunk_path, start / sr))

    logger.info("[%s] Split into %d chunk(s)", audio_path.stem, len(chunks))
    return chunks


# ─── Stage 4: Speaker Diarization via Sarvam Batch API ───────────────────────

class SarvamDiarizer:
    """
    Uses the Sarvam SDK batch job API for diarization.

    Install SDK:  pip install sarvamai

    Batch workflow per file:
        job = client.speech_to_text_job.create_job(model=..., with_diarization=True)
        job.upload_files([audio_path])
        job.start()
        job.wait_until_complete(timeout=600)
        outputs = job.download_outputs(output_dir)
        # outputs[0] contains the JSON transcript with diarized_transcript field
    """

    def __init__(self, api_key: str, base_url: str = "https://api.sarvam.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Lazy-init Sarvam SDK client."""
        if self._client is None:
            try:
                from sarvamai import SarvamAI
                self._client = SarvamAI(api_subscription_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "Sarvam SDK not installed. Run: pip install sarvamai"
                )
        return self._client

    def diarize(
        self,
        audio_path: str | Path,
        language: Language,
        min_confidence: float = 0.75,
        num_speakers: int | None = None,
        output_dir: Path | None = None,
    ) -> list[DiarizationSegment]:
        """
        Submit audio to Sarvam Batch API for diarization.
        Blocks until the job completes (polls every 10s, timeout 10 min).
        Returns list of single-speaker DiarizationSegments.
        """
        audio_path = Path(audio_path)
        video_id = audio_path.stem
        output_dir = output_dir or Path("temp_diarization") / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split if > 2h (7200s). Normally a single chunk.
        chunks = split_for_diarization(audio_path, output_dir / "chunks", chunk_seconds=7200)
        all_segments: list[DiarizationSegment] = []

        client = self._get_client()

        for chunk_path, offset in chunks:
            logger.info("[%s] Submitting batch diarization job for %s", video_id, chunk_path.name)

            try:
                # Step 1: Create job
                job_kwargs: dict[str, Any] = {
                    "model": "saaras:v3",
                    "mode": "transcribe",
                    "language_code": language.value,
                    "with_diarization": True,
                }
                if num_speakers:
                    job_kwargs["num_speakers"] = num_speakers

                job = client.speech_to_text_job.create_job(**job_kwargs)

                # Step 2: Upload file
                job.upload_files([str(chunk_path)])

                # Step 3: Start processing
                job.start()
                logger.info("[%s] Batch job started: %s", video_id, job.job_id)

                # status = job.get_status()
                # logger.info("[%s] Job %s status=%s",video_id,job.job_id,status)
                
                # Step 4: Wait for completion (poll, 10-min timeout)
                job.wait_until_complete(timeout=3600, poll_interval=15)
                logger.info("[%s] final status=%s",video_id,job.get_status())
                logger.info("[%s] Batch job completed", video_id)

                # Step 5: Download results
                result_dir = output_dir / f"results_{chunk_path.stem}"
                result_dir.mkdir(exist_ok=True)
                
                job.download_outputs(str(result_dir))
                json_files = list(result_dir.rglob("*.json"))
                
                # downloaded = job.download_outputs(str(result_dir))
                logger.info("[%s] download_outputs type=%s value=%s",video_id,len(json_files), [str(f) for f in json_files],)
                # if isinstance(downloaded, bool):
                #     downloaded = list(result_dir.glob("*.json"))
                #     downloaded = [str(p) for p in downloaded]

                # Step 6: Parse JSON output
                chunk_segments = self._parse_batch_output(
                    [str(f) for f in json_files], video_id, offset, min_confidence
                )
                all_segments.extend(chunk_segments)

            except Exception as exc:
                logger.error("[%s] Batch diarization failed: %s", video_id, exc)
                continue

        logger.info(
            "[%s] Diarization done: %d segments (confidence>=%.2f)",
            video_id, len(all_segments), min_confidence
        )
        return all_segments

    def _parse_batch_output(
        self,
        downloaded_files: list[str],
        video_id: str,
        offset: float,
        min_confidence: float,
    ) -> list[DiarizationSegment]:
        """
        Parse Sarvam batch output JSON.

        The batch API returns a JSON file with a `diarized_transcript` field:
        {
          "diarized_transcript": [
            {
              "speaker": "SPEAKER_00",
              "start": 1.2,
              "end": 5.8,
              "transcript": "Hello world",
              "confidence": 0.92,
              "words": [...]
            },
            ...
          ]
        }
        """
        segments: list[DiarizationSegment] = []

        for file_path in downloaded_files:
            if not file_path.endswith(".json"):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("TOP LEVEL KEYS = %s", list(data.keys()))
                    logger.info( "ENTRIES TYPE=%s VALUE SAMPLE=%s", type(data.get("entries")),str(data.get("entries"))[:500])
                    
                    if "entries" in data:
                        logger.info("entries count=%d first =%s", len(data["entries"]), data["entries"][0] if data["entries"] else None)
                    logger.info( "Loaded JSON file=%s type=%s",file_path,type(data))
                    logger.info("JSON loaded successfully (%d chars)",len(str(data)))
            except Exception as exc:
                logger.warning("Could not parse batch output %s: %s", file_path, exc)
                continue

            # Handle both diarized_transcript and segments fields
            items = []
            if not isinstance(data, dict):
                logger.warning("Unexpected JSON type: %s", type(data))
                continue

            if "entries" in data:
                items = data["entries"]

            elif "diarized_transcript" in data:
                diarized = data["diarized_transcript"]

                if isinstance(diarized, list):
                    items = diarized

                elif isinstance(diarized, dict):
                    items = diarized.get("entries", [])

            elif "segments" in data:
                items = data["segments"]

            logger.info("entries_type=%s entries_len=%s",type(items),len(items) if hasattr(items, "__len__") else "N/A",)
            logger.info("entries_sample=%s",str(items)[:500])
            
            if len(items)> 0:
                logger.info("FIRST RNTRY KEYS=%s", list(items[0].keys()))
                
            for item in items:
                if not isinstance(item, dict):
                    logger.warning("Unexpected item type=%s value=%s",type(item),item,)
                    continue
                start = float(item.get("start_time_seconds",  item.get("start", 0.0))) + offset
                end   = float(item.get("end_time_seconds", item.get("end", 0.0))) + offset
                duration = end - start

                # speaker field varies: "SPEAKER_00", "speaker_0", etc.
                speaker = str(item.get("speaker_id", item.get("speaker",item.get("speaker_label","SPEAKER_00"))))

                # Batch API confidence — not always present; default to 0.8
                # confidence = float(item.get("confidence", 0.8))
                confidence = float(item.get("confidence", 1.0))

                if duration < 1.0:
                    continue
                if confidence < min_confidence:
                    continue

                # Only accept segments with a single unambiguous speaker label
                segments.append(DiarizationSegment(
                    video_id=video_id,
                    speaker_id=speaker,
                    start_time=start,
                    end_time=end,
                    duration=duration,
                    confidence=confidence,
                    is_single_speaker=True,
                ))

        return segments


def select_dominant_speaker(
    segments: list[DiarizationSegment],
) -> tuple[str, list[DiarizationSegment]]:
    """
    Identify the dominant speaker (most total speaking time).
    For TTS we want a single consistent voice throughout the dataset.
    Returns (dominant_speaker_id, filtered_segments).
    """
    from collections import defaultdict
    speaker_time: dict[str, float] = defaultdict(float)
    for seg in segments:
        speaker_time[seg.speaker_id] += seg.duration

    if not speaker_time:
        return ("UNKNOWN", [])

    dominant = max(speaker_time, key=lambda k: speaker_time[k])
    filtered  = [s for s in segments if s.speaker_id == dominant]
    logger.info(
        "Dominant speaker: %s (%.1fs of %.1fs total)",
        dominant,
        speaker_time[dominant],
        sum(speaker_time.values()),
    )
    return dominant, filtered


# ─── Stage 5: Candidate Segment Generation ───────────────────────────────────

def _merge_nearby_segments(
    segments: list[DiarizationSegment],
    max_gap: float = 0.3,
) -> list[DiarizationSegment]:
    """
    Merge contiguous segments from the same speaker if gap < max_gap seconds.
    Rationale: small pauses within a sentence should not cause splits.
    """
    if not segments:
        return []

    merged: list[DiarizationSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap  = seg.start_time - prev.end_time
        if seg.speaker_id == prev.speaker_id and gap < max_gap:
            merged[-1] = DiarizationSegment(
                video_id=prev.video_id,
                speaker_id=prev.speaker_id,
                start_time=prev.start_time,
                end_time=seg.end_time,
                duration=seg.end_time - prev.start_time,
                confidence=min(prev.confidence, seg.confidence),
                is_single_speaker=True,
            )
        else:
            merged.append(seg)
    return merged


def _split_long_segment(
    seg: DiarizationSegment,
    max_duration: float,
) -> list[tuple[float, float]]:
    """Split a segment that exceeds max_duration into equal sub-windows."""
    windows: list[tuple[float, float]] = []
    t = seg.start_time
    while t < seg.end_time:
        end = min(t + max_duration, seg.end_time)
        windows.append((t, end))
        t = end
    return windows


def generate_candidate_segments(
    audio_path: str | Path,
    diarization_segments: list[DiarizationSegment],
    language: Language,
    output_dir: Path,
    segment_prefix: str,
    min_duration: float = 5.0,
    max_duration: float = 20.0,
    pad_start: float = 0.05,
    pad_end: float = 0.05,
) -> list[CandidateSegment]:
    """
    Stage 5: Convert diarization segments into TTS-ready audio clips.

    Rules:
    - Min 5s:  TTS models struggle with very short clips.
    - Max 20s: Longer clips risk attention drift in seq2seq TTS.
    - 50ms padding: avoid clipping first/last phoneme.
    """
    from tts_guj_eng_sarvam_dataset.src.preprocessing.audio import extract_audio_segment

    audio_path = Path(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged    = _merge_nearby_segments(diarization_segments, max_gap=0.3)
    candidates: list[CandidateSegment] = []
    counter   = 0

    for seg in merged:
        time_windows = _split_long_segment(seg, max_duration)

        for (t_start, t_end) in time_windows:
            duration = t_end - t_start
            if duration < min_duration or duration > max_duration:
                continue

            padded_start = max(0.0, t_start - pad_start)
            padded_end   = t_end + pad_end

            counter += 1
            seg_id   = f"{segment_prefix}_{counter:04d}"
            out_file = output_dir / f"{seg_id}.wav"

            try:
                extract_audio_segment(audio_path, out_file, padded_start, padded_end)
            except Exception as exc:
                logger.warning("Failed to cut segment %s: %s", seg_id, exc)
                continue

            candidates.append(CandidateSegment(
                segment_id=seg_id,
                video_id=seg.video_id,
                speaker_id=seg.speaker_id,
                language=language,
                start_time=padded_start,
                end_time=padded_end,
                duration=padded_end - padded_start,
                audio_path=str(out_file),
            ))

    logger.info(
        "[%s] Generated %d candidate segments from %d diarization segments",
        audio_path.stem, len(candidates), len(merged)
    )
    return candidates
