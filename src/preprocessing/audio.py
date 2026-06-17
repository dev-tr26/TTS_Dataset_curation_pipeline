"""
Stage 2 — ffmpeg-based WAV extraction (16kHz mono).
Stage 3 — Librosa-based quality filtering: remove music, jingles, silence-heavy sections.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import librosa
import numpy as np

from src.schemas import AudioFilterResult, VideoMetadata

logger = logging.getLogger(__name__)


# ─── Stage 2: Audio Extraction ────────────────────────────────────────────────

def extract_audio(
    raw_path: str | Path,
    out_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """
    Convert raw audio/video to 16kHz mono WAV using ffmpeg.
    Idempotent: skips if output already exists.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.debug("Audio already extracted: %s", out_path)
        return out_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-ac", "1",                # mono
        "-ar", str(sample_rate),   # 16kHz
        "-sample_fmt", "s16",      # 16-bit PCM
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {raw_path}: {result.stderr}")

    logger.info("Extracted audio -> %s", out_path)
    return out_path


def extract_audio_segment(
    audio_path: str | Path,
    out_path: str | Path,
    start: float,
    end: float,
    sample_rate: int = 16000,
) -> Path:
    """Cut a precise time segment from a WAV file using ffmpeg."""
    audio_path = Path(audio_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t",  str(duration),
        "-i",  str(audio_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment cut failed: {result.stderr}")
    return out_path


# ─── Stage 3: Audio Quality Filtering ────────────────────────────────────────

def _spectral_flatness_frames(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512
) -> np.ndarray:
    """
    Compute per-frame spectral flatness.
    High flatness → noise-like signal (background noise, music).
    Low flatness  → tonal/harmonic structure (speech has medium flatness).
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    eps = 1e-10
    geometric_mean = np.exp(np.mean(np.log(S + eps), axis=0))
    arithmetic_mean = np.mean(S, axis=0)
    flatness = geometric_mean / (arithmetic_mean + eps)
    return flatness


def _estimate_speech_ratio(
    y: np.ndarray,
    sr: int,
    rms_floor_db: float = -50.0,
    flatness_threshold: float = 0.4,
    hop: int = 512,
    n_fft: int = 2048,
) -> float:
    """
    Heuristic speech ratio: frames that are:
      - Above the RMS floor (not silence)
      - Below the spectral flatness threshold (not pure noise)
    are classified as speech. Returns ratio 0–1.

    Rationale:
      - Pure silence has very low RMS → not speech
      - Music/background noise has high spectral flatness → not speech
      - Speech has medium flatness and adequate energy
    """
    # RMS per frame
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    energy_mask = rms_db > rms_floor_db

    flatness = _spectral_flatness_frames(y, sr, n_fft=n_fft, hop=hop)
    speech_mask_flatness = flatness < flatness_threshold

    n_frames = min(len(energy_mask), len(speech_mask_flatness))
    speech_frames = np.logical_and(
        energy_mask[:n_frames], speech_mask_flatness[:n_frames]
    )
    return float(speech_frames.sum()) / max(n_frames, 1)


def _estimate_music_energy_ratio(
    y: np.ndarray,
    sr: int,
    hop: int = 512,
    n_fft: int = 2048,
) -> float:
    """
    Estimate music energy ratio using chroma energy variance.
    Music has high chroma energy (tonal harmony), speech has low chroma.
    Returns ratio 0–1.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    chroma_energy = chroma.sum(axis=0)
    # Frames where chroma is strong AND uniformly spread → music
    chroma_variance = np.var(chroma, axis=0)
    high_chroma_frames = (chroma_energy > np.percentile(chroma_energy, 75)) & (
        chroma_variance < np.percentile(chroma_variance, 30)
    )
    return float(high_chroma_frames.sum()) / max(len(high_chroma_frames), 1)


def detect_speech_regions(
    audio_path: str | Path,
    skip_intro_sec: float = 30.0,
    skip_outro_sec: float = 20.0,
    min_speech_ratio: float = 0.6,
    max_music_ratio: float = 0.3,
    rms_floor_db: float = -50.0,
    flatness_threshold: float = 0.4,
) -> AudioFilterResult:
    """
    Stage 3 main function.

    Strategy:
    1. Skip fixed intro/outro (credits, jingles, music beds).
    2. Compute speech ratio and music energy ratio over the remaining audio.
    3. If the file passes both thresholds, return the usable time range.
    4. Otherwise mark as failed with a reason.

    Returns AudioFilterResult with (start, end) speech windows.
    """
    audio_path = Path(audio_path)
    video_id = audio_path.stem

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception as exc:
        logger.error("[%s] Failed to load audio: %s", video_id, exc)
        return AudioFilterResult(
            video_id=video_id,
            passed=False,
            speech_ratio=0.0,
            music_energy_ratio=1.0,
            rejection_reason=f"load_error: {exc}",
        )

    total_dur = len(y) / sr

    # Trim fixed intro/outro
    start_sample = int(min(skip_intro_sec, total_dur * 0.2) * sr)
    end_sample   = max(int((total_dur - skip_outro_sec) * sr), start_sample + sr)
    y_core = y[start_sample:end_sample]

    if len(y_core) < sr * 5:
        return AudioFilterResult(
            video_id=video_id,
            passed=False,
            speech_ratio=0.0,
            music_energy_ratio=0.0,
            rejection_reason="audio_too_short_after_trim",
        )

    speech_ratio = _estimate_speech_ratio(
        y_core, sr, rms_floor_db=rms_floor_db, flatness_threshold=flatness_threshold
    )
    music_ratio = _estimate_music_energy_ratio(y_core, sr)

    passed = speech_ratio >= min_speech_ratio and music_ratio <= max_music_ratio

    rejection_reason: str | None = None
    if not passed:
        if speech_ratio < min_speech_ratio:
            rejection_reason = f"low_speech_ratio={speech_ratio:.2f}"
        elif music_ratio > max_music_ratio:
            rejection_reason = f"high_music_ratio={music_ratio:.2f}"

    logger.info(
        "[%s] speech=%.2f music=%.2f -> %s",
        video_id, speech_ratio, music_ratio, "PASS" if passed else f"FAIL({rejection_reason})"
    )

    usable_start = start_sample / sr
    usable_end   = end_sample / sr

    return AudioFilterResult(
        video_id=video_id,
        passed=passed,
        speech_ratio=speech_ratio,
        music_energy_ratio=music_ratio,
        filtered_segments=[(usable_start, usable_end)] if passed else [],
        rejection_reason=rejection_reason,
    )