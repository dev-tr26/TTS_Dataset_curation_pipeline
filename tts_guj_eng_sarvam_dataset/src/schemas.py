"""
src/schemas.py
Central Pydantic schemas shared across all pipeline stages.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class Language(str, Enum):
    EN_IN = "en-IN"
    HI_IN = "hi-IN"
    BN_IN = "bn-IN"
    TA_IN = "ta-IN"
    TE_IN = "te-IN"
    KN_IN = "kn-IN"
    MR_IN = "mr-IN"
    GU_IN = "gu-IN"


class ReviewStatus(str, Enum):
    PENDING  = "pending"
    ACCEPTED = "accepted"
    REVIEW   = "needs_review"
    REJECTED = "rejected"


class Emotion(str, Enum):
    NEUTRAL       = "neutral"
    HAPPY         = "happy"
    SAD           = "sad"
    EXCITED       = "excited"
    ANGRY         = "angry"
    FORMAL        = "formal"
    CONVERSATIONAL = "conversational"
    STORYTELLING  = "storytelling"
    SERIOUS       = "serious"


# ─── Stage 1: Ingestion ───────────────────────────────────────────────────────

class VideoInput(BaseModel):
    video_url: str
    language: Language

    @field_validator("video_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError(f"Not a YouTube URL: {v}")
        return v.strip()


class VideoMetadata(BaseModel):
    video_id: str
    title: str
    channel: str
    duration_seconds: float
    url: str
    language: Language
    raw_audio_path: Optional[str] = None
    ingestion_status: str = "pending"
    error_message: Optional[str] = None


# ─── Stage 3: Quality Filter ──────────────────────────────────────────────────

class AudioFilterResult(BaseModel):
    video_id: str
    passed: bool
    speech_ratio: float
    music_energy_ratio: float
    filtered_segments: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (start_sec, end_sec) of speech-only regions"
    )
    rejection_reason: Optional[str] = None


# ─── Stage 4: Diarization ─────────────────────────────────────────────────────

class DiarizationSegment(BaseModel):
    video_id: str
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    is_single_speaker: bool = True


# ─── Stage 5: Candidate Segment ───────────────────────────────────────────────

class CandidateSegment(BaseModel):
    segment_id: str            # e.g. en_001
    video_id: str
    speaker_id: str
    language: Language
    start_time: float
    end_time: float
    duration: float
    audio_path: str


# ─── Stage 6: Transcription ───────────────────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: float


class TranscriptionResult(BaseModel):
    segment_id: str
    transcript: str
    language: Language
    asr_confidence: float
    word_timestamps: list[WordTimestamp] = Field(default_factory=list)
    asr_model: str = "saarika:v2"


# ─── Stage 7: Validation ──────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    segment_id: str
    layer1_confidence_ok: bool
    layer2_normalized_transcript: str
    layer3_language_ok: bool
    layer4_llm_score: float        # 0–1 from Sarvam LLM
    layer4_llm_issues: list[str]   = Field(default_factory=list)
    transcript_quality_score: float  # 0–1 composite
    passed: bool



# ─── Stage 8: Audio Quality ───────────────────────────────────────────────────

class AudioQualityMetrics(BaseModel):
    segment_id: str
    duration: float
    rms_energy_db: float
    clipping_ratio: float
    silence_ratio: float
    estimated_snr_db: float
    speech_rate_wpm: float
    audio_quality_score: float     # 0–1 weighted
    passed: bool


# ─── Stage 9: Emotion ─────────────────────────────────────────────────────────

class EmotionResult(BaseModel):
    segment_id: str
    emotion: Emotion
    emotion_confidence: float
    style: str                     # e.g. "formal", "conversational"
    acoustic_scores: dict[str, float] = Field(default_factory=dict)
    llm_scores: dict[str, float]      = Field(default_factory=dict)


# ─── Stage 11: Final Score ────────────────────────────────────────────────────

class QualityScore(BaseModel):
    segment_id: str
    audio_quality_score: float
    transcript_quality_score: float
    speaker_purity_score: float
    emotion_confidence_score: float
    final_score: float
    decision: ReviewStatus


# ─── Stage 12: Final Dataset Sample ──────────────────────────────────────────

class DatasetSample(BaseModel):
    """Schema for a single sample in the published TTS dataset."""
    audio: str                    # Relative path within the dataset
    language: Language
    speaker_id: str
    duration: float
    transcript: str
    emotion: Emotion
    emotion_confidence: float
    style: str
    source: str                   # YouTube video_id
    asr_confidence: float
    # Extended metadata
    segment_id: str
    channel: str
    video_title: str
    rms_energy_db: float
    estimated_snr_db: float
    speech_rate_wpm: float
    clipping_ratio: float
    silence_ratio: float
    word_timestamps: list[WordTimestamp] = Field(default_factory=list)
    llm_quality_score: float
    final_score: float
    review_status: ReviewStatus = ReviewStatus.ACCEPTED