"""
src/quality/scorer.py
Stage 11 — Final quality scoring framework.
Combines scores from Stages 7, 8, 9 + speaker purity into a single weighted score.
Decides Accept / Needs Review / Reject.
"""
from __future__ import annotations

import logging

from tts_guj_eng_sarvam_dataset.src.schemas import (
    AudioQualityMetrics,
    EmotionResult,
    QualityScore,
    ReviewStatus,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# ─── Scoring Weights ──────────────────────────────────────────────────────────
#
# These weights reflect the relative importance of each quality dimension
# for a commercial TTS model:
#
# Audio Quality (35%):
#   The single most important factor. Low SNR, clipping, or excess silence
#   directly degrades TTS vocoder quality and is unrecoverable.
#
# Transcript Quality (35%):
#   Equally critical. Mismatched text/audio pairs are the #1 cause of
#   unintelligible TTS output. Accurate transcriptions train good attention.
#
# Speaker Purity (20%):
#   Multi-speaker segments confuse the model into voice averaging.
#   Diarization confidence feeds this score.
#
# Emotion Confidence (10%):
#   Less critical for basic TTS, but important for expressive TTS.
#   Low-confidence emotion labels add noise to style conditioning.

WEIGHTS = {
    "audio_quality":      0.35,
    "transcript_quality": 0.35,
    "speaker_purity":     0.20,
    "emotion_confidence": 0.10,
}

THRESHOLDS = {
    "accept": 0.75,   # High-quality, include in final dataset
    "review": 0.55,   # Borderline — flag for human review
    # Below 0.55 → auto-reject
}


def compute_final_score(
    segment_id: str,
    audio_metrics: AudioQualityMetrics,
    validation: ValidationResult,
    diarization_confidence: float,
    emotion: EmotionResult,
) -> QualityScore:
    """
    Compute the final weighted quality score.

    Formula:
        final = 0.35 * audio_quality_score
              + 0.35 * transcript_quality_score
              + 0.20 * speaker_purity_score
              + 0.10 * emotion_confidence

    Hard gates (any failure → cap score at 0.4):
    - Clipping ratio > 5%
    - ASR confidence < 0.5
    - Language mismatch
    """
    audio_score      = audio_metrics.audio_quality_score
    transcript_score = validation.transcript_quality_score
    speaker_purity   = diarization_confidence         # 0–1 from Sarvam
    emotion_conf     = emotion.emotion_confidence

    final = (
        WEIGHTS["audio_quality"]      * audio_score
        + WEIGHTS["transcript_quality"] * transcript_score
        + WEIGHTS["speaker_purity"]     * speaker_purity
        + WEIGHTS["emotion_confidence"] * emotion_conf
    )

    # Hard gate: clipping
    if audio_metrics.clipping_ratio > 0.05:
        final = min(final, 0.40)
        logger.debug("[%s] Hard gate: clipping too high", segment_id)

    # Hard gate: language mismatch
    if not validation.layer3_language_ok:
        final = min(final, 0.40)
        logger.debug("[%s] Hard gate: language mismatch", segment_id)

    # Hard gate: very low LLM transcript score
    if validation.layer4_llm_score < 0.4:
        final = min(final, 0.40)
        logger.debug("[%s] Hard gate: low LLM transcript score", segment_id)

    # Decision
    if final >= THRESHOLDS["accept"]:
        decision = ReviewStatus.ACCEPTED
    elif final >= THRESHOLDS["review"]:
        decision = ReviewStatus.REVIEW
    else:
        decision = ReviewStatus.REJECTED

    logger.info(
        "[%s] final=%.3f (%s) [aud=%.2f trx=%.2f spk=%.2f emo=%.2f]",
        segment_id, final, decision.value,
        audio_score, transcript_score, speaker_purity, emotion_conf
    )

    return QualityScore(
        segment_id=segment_id,
        audio_quality_score=audio_score,
        transcript_quality_score=transcript_score,
        speaker_purity_score=speaker_purity,
        emotion_confidence_score=emotion_conf,
        final_score=final,
        decision=decision,
    )