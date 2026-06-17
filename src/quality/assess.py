from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Any
 
import librosa
import numpy as np
import requests
 
from src.schemas import (
    AudioQualityMetrics,
    CandidateSegment,
    Emotion,
    EmotionResult,
    Language,
    TranscriptionResult,
    ValidationResult,
)
 
logger = logging.getLogger(__name__)
 

import numpy as np

def estimate_snr(y: np.ndarray):
    signal_power = np.mean(y ** 2)

    if signal_power == 0:
        return 0.0
    window = 16

    # Estimate noise from deviation from smoothed signal
    smooth = np.convolve(
        y,
        np.ones(window) / window,
        mode="same"
    )

    noise = y - smooth
    noise_power = np.mean(noise ** 2)

    if noise_power <= 1e-12:
        return 100.0

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


 
# ─── Stage 8: Audio Quality Assessment ───────────────────────────────────────
 
def compute_audio_quality(
    audio_path: str | Path,
    segment_id: str,
    transcript: str,
    duration: float,
    max_clipping_ratio: float = 0.01,
    min_snr_db: float = 15.0,
    max_silence_ratio: float = 0.3,
    min_speech_rate_wpm: float = 80.0,
    max_speech_rate_wpm: float = 220.0,
) -> AudioQualityMetrics:
    """
    Compute audio quality metrics for a TTS candidate segment.
 
    Metrics:
    - RMS energy dB:   overall loudness level
    - Clipping ratio:  fraction of samples at ±full-scale (causes distortion)
    - Silence ratio:   fraction of frames below energy floor (dead air)
    - Estimated SNR:   signal-to-noise ratio heuristic
    - Speech rate wpm: natural speech pace check
 
    Scoring weights:
    - SNR:             30% (most critical for TTS)
    - No clipping:     25% (clipping is unrecoverable)
    - Low silence:     20% (silence wastes training data)
    - RMS in range:    15% (adequate recording level)
    - Speech rate:     10% (too fast/slow → unnatural prosody)
    """
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
 
    # ── RMS energy ──────────────────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_db = float(librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0])
 
    # ── Clipping ratio ───────────────────────────────────────────────────────
    # PCM int16 full scale is ±1.0 after librosa normalisation
    clip_threshold = 0.99
    clipping_ratio = float(np.mean(np.abs(y) >= clip_threshold))
 
    # ── Silence ratio ────────────────────────────────────────────────────────
    frame_rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    silence_db_floor = -40.0
    frame_db = librosa.amplitude_to_db(frame_rms, ref=1.0)
    silence_ratio = float(np.mean(frame_db < silence_db_floor))
 
    # ── Estimated SNR (simplified) ───────────────────────────────────────────
    # Sort frames by energy; bottom 10% ≈ noise floor, top 50% ≈ signal
    sorted_energy = np.sort(frame_rms)
    noise_floor   = np.mean(sorted_energy[: max(1, len(sorted_energy) // 10)]) + 1e-10
    signal_level  = np.mean(sorted_energy[len(sorted_energy) // 2 :]) + 1e-10
    snr_db        = estimate_snr(y)
 
    # ── Speech rate wpm ──────────────────────────────────────────────────────
    word_count = len(transcript.split()) if transcript else 0
    speech_rate_wpm = float((word_count / max(duration, 0.1)) * 60.0)
 
    # ── Composite score ──────────────────────────────────────────────────────
    def _norm(val: float, lo: float, hi: float) -> float:
        """Clamp-and-normalise to [0, 1]."""
        return max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-6)))
 
    snr_score     = _norm(snr_db, 10.0, 40.0)
    clip_score    = 1.0 - _norm(clipping_ratio, 0.0, max_clipping_ratio * 2)
    silence_score = 1.0 - _norm(silence_ratio, 0.0, max_silence_ratio * 2)
    rms_score     = _norm(rms_db, -30.0, -5.0)   # -20 dBFS is typical broadcast level
 
    # Speech rate: penalise both extremes
    if speech_rate_wpm < min_speech_rate_wpm or speech_rate_wpm > max_speech_rate_wpm:
        rate_score = 0.3
    else:
        rate_score = 1.0
 
    audio_quality_score = (
        0.30 * snr_score
        + 0.25 * clip_score
        + 0.20 * silence_score
        + 0.15 * rms_score
        + 0.10 * rate_score
    )
 
    passed = (
        clipping_ratio <= max_clipping_ratio
        and snr_db >= min_snr_db
        and silence_ratio <= max_silence_ratio
        and min_speech_rate_wpm <= speech_rate_wpm <= max_speech_rate_wpm
    )
 
    logger.debug(
        "[%s] snr=%.1fdB clip=%.4f sil=%.2f rate=%.0fwpm score=%.2f %s",
        segment_id, snr_db, clipping_ratio, silence_ratio, speech_rate_wpm,
        audio_quality_score, "PASS" if passed else "FAIL"
    )
 
    return AudioQualityMetrics(
        segment_id=segment_id,
        duration=duration,
        rms_energy_db=rms_db,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        estimated_snr_db=snr_db,
        speech_rate_wpm=speech_rate_wpm,
        audio_quality_score=audio_quality_score,
        passed=passed,
    )
 