"""
tests/test_pipeline.py
Unit tests for key pipeline components.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─── Schema tests ─────────────────────────────────────────────────────────────

def test_video_input_valid():
    from tts_guj_eng_sarvam_dataset.src.schemas import Language, VideoInput
    vi = VideoInput(video_url="https://youtube.com/watch?v=dQw4w9WgXcQ", language=Language.EN_IN)
    assert vi.video_url == "https://youtube.com/watch?v=dQw4w9WgXcQ"


def test_video_input_invalid_url():
    from pydantic import ValidationError
    from tts_guj_eng_sarvam_dataset.src.schemas import Language, VideoInput
    with pytest.raises(ValidationError):
        VideoInput(video_url="https://vimeo.com/12345", language=Language.EN_IN)


def test_dataset_sample_schema():
    from tts_guj_eng_sarvam_dataset.src.schemas import DatasetSample, Emotion, Language, ReviewStatus
    sample = DatasetSample(
        audio="clips/english/en_001.wav",
        language=Language.EN_IN,
        speaker_id="eng_spk_01",
        duration=12.5,
        transcript="Welcome to today's session.",
        emotion=Emotion.NEUTRAL,
        emotion_confidence=0.85,
        style="formal",
        source="dQw4w9WgXcQ",
        asr_confidence=0.92,
        segment_id="en_001",
        channel="Test Channel",
        video_title="Test Video",
        rms_energy_db=-18.0,
        estimated_snr_db=25.0,
        speech_rate_wpm=140.0,
        clipping_ratio=0.0,
        silence_ratio=0.1,
        llm_quality_score=0.9,
        final_score=0.87,
    )
    assert sample.emotion == Emotion.NEUTRAL
    assert sample.language == Language.EN_IN


# ─── CSV loading tests ─────────────────────────────────────────────────────────

def test_load_video_inputs():
    from tts_guj_eng_sarvam_dataset.src.ingestion.ingest import load_video_inputs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("video_url,language\n")
        f.write("https://youtube.com/watch?v=abc123defgh,en-IN\n")
        f.write("https://youtube.com/watch?v=xyz987uvwxy,hi-IN\n")
        csv_path = f.name
    inputs = load_video_inputs(csv_path)
    assert len(inputs) == 2
    assert inputs[0].language.value == "en-IN"
    assert inputs[1].language.value == "hi-IN"


def test_load_video_inputs_skips_invalid():
    from tts_guj_eng_sarvam_dataset.src.ingestion.ingest import load_video_inputs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("video_url,language\n")
        f.write("https://youtube.com/watch?v=abc123defgh,en-IN\n")
        f.write("https://vimeo.com/12345,en-IN\n")   # invalid → skipped
        csv_path = f.name
    inputs = load_video_inputs(csv_path)
    assert len(inputs) == 1


# ─── Audio quality tests ───────────────────────────────────────────────────────

def test_audio_quality_clean_signal():
    """Clean synthetic signal should pass quality checks."""
    from tts_guj_eng_sarvam_dataset.src.quality.assess import compute_audio_quality
    import soundfile as sf

    sr = 16000
    t  = np.linspace(0, 5, sr * 5)
    y  = 0.3 * np.sin(2 * np.pi * 200 * t)  # Clean 200 Hz tone

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        tmp_path = f.name

    aqm = compute_audio_quality(
        audio_path=tmp_path,
        segment_id="test_001",
        transcript="hello world this is a test",
        duration=5.0,
    )
    # Clean sine wave should have good SNR and no clipping
    assert aqm.clipping_ratio < 0.01
    assert aqm.estimated_snr_db > 5.0


def test_audio_quality_clipped_signal():
    """Heavily clipped signal should have high clipping ratio."""
    from tts_guj_eng_sarvam_dataset.src.quality.assess import compute_audio_quality
    import soundfile as sf

    sr = 16000
    t  = np.linspace(0, 5, sr * 5)
    y  = np.clip(5.0 * np.sin(2 * np.pi * 200 * t), -1.0, 1.0)  # Saturated

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        tmp_path = f.name

    aqm = compute_audio_quality(
        audio_path=tmp_path,
        segment_id="test_clip",
        transcript="test transcript",
        duration=5.0,
    )
    # Should detect significant clipping
    assert aqm.clipping_ratio > 0.05


# ─── Transcript validation tests ─────────────────────────────────────────────

def test_transcript_normalization_english():
    from tts_guj_eng_sarvam_dataset.src.schemas import Language, TranscriptionResult
    from tts_guj_eng_sarvam_dataset.src.validation.validate import TranscriptValidator

    validator = TranscriptValidator(api_key="dummy")
    result = validator._normalize("  hello   world  uh um  ", Language.EN_IN)
    assert "uh" not in result
    assert "um" not in result
    assert result.strip() == result
    assert "  " not in result


def test_language_check_english():
    from tts_guj_eng_sarvam_dataset.src.schemas import Language
    from tts_guj_eng_sarvam_dataset.src.validation.validate import TranscriptValidator
    v = TranscriptValidator(api_key="dummy")
    assert v._check_language("Hello, how are you today?", Language.EN_IN)
    assert not v._check_language("नमस्ते", Language.EN_IN)


def test_language_check_hindi():
    from tts_guj_eng_sarvam_dataset.src.schemas import Language
    from tts_guj_eng_sarvam_dataset.src.validation.validate import TranscriptValidator
    v = TranscriptValidator(api_key="dummy")
    assert v._check_language("नमस्ते आप कैसे हैं", Language.HI_IN)
    assert not v._check_language("Hello how are you", Language.HI_IN)


# ─── Quality scorer tests ─────────────────────────────────────────────────────

def test_quality_scorer_accept():
    from tts_guj_eng_sarvam_dataset.src.quality.scorer import compute_final_score
    from tts_guj_eng_sarvam_dataset.src.schemas import (
        AudioQualityMetrics,
        Emotion,
        EmotionResult,
        ReviewStatus,
        ValidationResult,
    )

    aqm = AudioQualityMetrics(
        segment_id="s1", duration=10.0, rms_energy_db=-18.0,
        clipping_ratio=0.0, silence_ratio=0.05,
        estimated_snr_db=30.0, speech_rate_wpm=140.0,
        audio_quality_score=0.9, passed=True,
    )
    val = ValidationResult(
        segment_id="s1", layer1_confidence_ok=True,
        layer2_normalized_transcript="Hello world.",
        layer3_language_ok=True, layer4_llm_score=0.95,
        transcript_quality_score=0.92, passed=True,
    )
    emo = EmotionResult(
        segment_id="s1", emotion=Emotion.NEUTRAL,
        emotion_confidence=0.85, style="formal",
    )
    score = compute_final_score("s1", aqm, val, 0.90, emo)
    assert score.decision == ReviewStatus.ACCEPTED
    assert score.final_score >= 0.75


def test_quality_scorer_reject_clipping():
    from tts_guj_eng_sarvam_dataset.src.quality.scorer import compute_final_score
    from tts_guj_eng_sarvam_dataset.src.schemas import (
        AudioQualityMetrics,
        Emotion,
        EmotionResult,
        ReviewStatus,
        ValidationResult,
    )

    aqm = AudioQualityMetrics(
        segment_id="s2", duration=10.0, rms_energy_db=-5.0,
        clipping_ratio=0.15,           # Heavily clipped!
        silence_ratio=0.05,
        estimated_snr_db=30.0, speech_rate_wpm=140.0,
        audio_quality_score=0.3, passed=False,
    )
    val = ValidationResult(
        segment_id="s2", layer1_confidence_ok=True,
        layer2_normalized_transcript="Hello world.",
        layer3_language_ok=True, layer4_llm_score=0.9,
        transcript_quality_score=0.88, passed=True,
    )
    emo = EmotionResult(
        segment_id="s2", emotion=Emotion.NEUTRAL,
        emotion_confidence=0.8, style="formal",
    )
    score = compute_final_score("s2", aqm, val, 0.9, emo)
    # Hard gate: clipping > 5% caps at 0.40 → must be rejected
    assert score.final_score <= 0.40
    assert score.decision == ReviewStatus.REJECTED


# ─── Diarization segment merging tests ────────────────────────────────────────

def test_merge_nearby_segments():
    from tts_guj_eng_sarvam_dataset.src.diarization.diarize import _merge_nearby_segments
    from tts_guj_eng_sarvam_dataset.src.schemas import DiarizationSegment

    segs = [
        DiarizationSegment(video_id="v1", speaker_id="A",
                           start_time=0.0, end_time=5.0, duration=5.0, confidence=0.9),
        DiarizationSegment(video_id="v1", speaker_id="A",
                           start_time=5.2, end_time=10.0, duration=4.8, confidence=0.85),
        DiarizationSegment(video_id="v1", speaker_id="A",
                           start_time=15.0, end_time=20.0, duration=5.0, confidence=0.9),
    ]
    merged = _merge_nearby_segments(segs, max_gap=0.3)
    # First two should merge (gap = 0.2s < 0.3s); third stays separate
    assert len(merged) == 2
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])