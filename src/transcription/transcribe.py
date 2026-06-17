"""
Stage 6 — Sarvam ASR: transcribe candidate segments with word timestamps.
Stage 7 — 4-layer transcript validation + Sarvam LLM quality check.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests

from src.schemas import (
    CandidateSegment,
    Language,
    TranscriptionResult,
    ValidationResult,
    WordTimestamp,
)

logger = logging.getLogger(__name__)


# ─── Stage 6: Transcription ───────────────────────────────────────────────────

class SarvamASR:
    """Sarvam Speech-to-Text client."""

    def __init__(self, api_key: str, base_url: str = "https://api.sarvam.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"api-subscription-key": api_key})

    def transcribe(
        self,
        audio_path: str | Path,
        language: Language,
        with_timestamps: bool = True,
    ) -> TranscriptionResult | None:
        """
        Transcribe a single audio segment using Sarvam ASR.
        Returns None on failure so the pipeline can skip the segment.
        """
        audio_path = Path(audio_path)
        seg_id = audio_path.stem

        try:
            with open(audio_path, "rb") as f:
                response = self.session.post(
                    f"{self.base_url}/speech-to-text",
                    files={"file": (audio_path.name, f, "audio/wav")},
                    data={
                        "model": "saarika:v2.5",
                        "language_code": language.value,
                        "with_timestamps": str(with_timestamps).lower(),
                        "with_disfluencies": "false",
                    },
                    timeout=120,
                )
            if response.status_code != 200:
                logger.error("Sarvam response body: %s", response.text)
            
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("[%s] ASR API failed: %s", seg_id, exc)
            return None

        data = response.json()
        return self._parse_response(data, seg_id, language)

    def _parse_response(
        self,
        data: dict[str, Any],
        seg_id: str,
        language: Language,
    ) -> TranscriptionResult:
        transcript = data.get("transcript", "").strip()
        confidence = float(data.get("confidence", 0.0))

        word_timestamps: list[WordTimestamp] = []
        for w in data.get("words", []):
            word_timestamps.append(WordTimestamp(
                word=w.get("text", ""),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                confidence=float(w.get("confidence", confidence)),
            ))

        return TranscriptionResult(
            segment_id=seg_id,
            transcript=transcript,
            language=language,
            asr_confidence=confidence,
            word_timestamps=word_timestamps,
        )

    def _transcribe_via_batch_api(self, audio_path, language):
        from sarvamai import SarvamAI
        from pathlib import Path
        import json,time

        audio_path = Path(audio_path)
        seg_id = audio_path.stem
        client = SarvamAI(api_subscription_key=self.api_key)

        result_dir = Path("temp_asr") / seg_id
        result_dir.mkdir(parents=True, exist_ok=True)

        try:
            job = client.speech_to_text_job.create_job(
                model="saaras:v3",
                mode="transcribe",
                language_code=language.value,
                with_timestamps=True,
            )
            job.upload_files([str(audio_path)])
            job.start()
            
            # Poll manually — same way diarization works
            timeout = 300
            poll_interval = 10
            elapsed = 0
            while elapsed < timeout:
                status = job.get_status()
                if status.job_state == "Completed":
                    break
                if status.job_state == "Failed":
                    logger.error("[%s] Batch ASR job failed", seg_id)
                    return None
                time.sleep(poll_interval)
                elapsed += poll_interval
            else:
                logger.error("[%s] Batch ASR timed out", seg_id)
                return None
            
            job.download_outputs(str(result_dir))
            json_files = list(result_dir.rglob("*.json"))
            if not json_files:
                return None

            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)

            transcript = (data.get("transcript") or "").strip()
            if not transcript:
                return None

            return TranscriptionResult(
                segment_id=seg_id,
                transcript=transcript,
                language=language,
                asr_confidence=0.85,
               word_timestamps=[],
                asr_model="saaras:v3-batch",
            )
        except Exception as exc:
            logger.error("[%s] Batch ASR failed: %s", seg_id, exc)
        return None


    def transcribe_batch(self,segments: list[CandidateSegment],min_confidence: float = 0.70,) -> dict[str, TranscriptionResult]:
        """Process all segments; return dict keyed by segment_id."""
        results: dict[str, TranscriptionResult] = {}
        for seg in segments:
            if seg.duration >= 29.0:
                result = self._transcribe_via_batch_api(seg.audio_path, seg.language)
            else:
                result = self.transcribe(seg.audio_path, seg.language)
            
            if result and result.asr_confidence >= min_confidence:
                results[seg.segment_id] = result
                logger.debug("[%s] conf=%.2f: %s", seg.segment_id, result.asr_confidence,
                             result.transcript[:60])
            else:
                logger.info("[%s] Dropped (low confidence or error)", seg.segment_id)
        return results