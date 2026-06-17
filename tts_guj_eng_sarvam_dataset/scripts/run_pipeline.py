"""
scripts/run_pipeline.py
Main orchestrator — runs all 13 pipeline stages end-to-end.

Usage:
    python scripts/run_pipeline.py videos.csv

Environment variables required:
    SARVAM_API_KEY  — Sarvam API key from dashboard.sarvam.ai
    HF_TOKEN        — HuggingFace write token
    HF_REPO_ID      — e.g. "username/indian-tts-dataset"
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import sys 
import yaml

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
    
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")
(ROOT / "logs").mkdir(exist_ok=True)


def load_config() -> dict:
    cfg_path = ROOT / "configs" / "pipeline.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_env(cfg: dict) -> dict:
    """Replace ${oc.env:VAR} placeholders with environment variables."""
    import re
    def _resolve(val):
        if isinstance(val, str):
            m = re.match(r"\$\{oc\.env:(\w+)(?:,(.+))?\}", val)
            if m:
                env_var, default = m.group(1), m.group(2)
                return os.environ.get(env_var, default or "")
        if isinstance(val, dict):
            return {k: _resolve(v) for k, v in val.items()}
        return val
    return {k: _resolve(v) for k, v in cfg.items()}


def main(csv_path: str) -> None:
    cfg  = resolve_env(load_config())
    cfg_paths  = cfg["paths"]
    cfg_sarvam = cfg["sarvam"]

    sarvam_key  = cfg_sarvam["api_key"]
    sarvam_url  = cfg_sarvam["base_url"]
    audio_dir   = Path(cfg_paths["audio_dir"])
    seg_dir     = Path(cfg_paths["segments_dir"])
    dataset_dir = Path(cfg_paths["dataset_dir"])

    if not sarvam_key:
        logger.error("SARVAM_API_KEY not set. Export it and retry.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Ingestion
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — Video Ingestion")
    from tts_guj_eng_sarvam_dataset.src.ingestion.ingest import ingest_all
    video_metas = ingest_all(
        csv_path=csv_path,
        output_dir=audio_dir,
        max_retries=cfg["ingestion"]["max_retries"],
        retry_delay=cfg["ingestion"]["retry_delay_seconds"],
    )
    successful = [m for m in video_metas if m.ingestion_status == "success"]
    logger.info("Ingested %d/%d videos", len(successful), len(video_metas))

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Audio Extraction
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — Audio Extraction")
    from tts_guj_eng_sarvam_dataset.src.preprocessing.audio import extract_audio
    for meta in successful:
        raw = Path(meta.raw_audio_path)
        clean_wav = audio_dir / f"{meta.video_id}_clean.wav"
        try:
            extract_audio(raw, clean_wav, sample_rate=cfg["audio"]["sample_rate"])
            meta.raw_audio_path = str(clean_wav)
        except Exception as exc:
            logger.warning("[%s] Extraction failed: %s", meta.video_id, exc)
            meta.ingestion_status = "audio_failed"

    successful = [m for m in successful if m.ingestion_status == "success"]

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Audio Quality Filtering
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — Audio Quality Filtering")
    from tts_guj_eng_sarvam_dataset.src.preprocessing.audio import detect_speech_regions
    qf_cfg = cfg["quality_filter"]
    filter_results = {}
    passed_metas = []
    for meta in successful:
        result = detect_speech_regions(
            audio_path=meta.raw_audio_path,
            skip_intro_sec=qf_cfg["skip_intro_seconds"],
            skip_outro_sec=qf_cfg["skip_outro_seconds"],
            min_speech_ratio=qf_cfg["min_speech_ratio"],
            max_music_ratio=qf_cfg["max_music_energy_ratio"],
        )
        filter_results[meta.video_id] = result
        if result.passed:
            passed_metas.append(meta)
        else:
            logger.info("[%s] Filtered out: %s", meta.video_id, result.rejection_reason)
    logger.info("%d/%d videos passed audio quality filter", len(passed_metas), len(successful))

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4: Speaker Diarization
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — Speaker Diarization")
    from tts_guj_eng_sarvam_dataset.src.diarization.diarize import SarvamDiarizer, select_dominant_speaker
    diarizer = SarvamDiarizer(api_key=sarvam_key, base_url=sarvam_url)

    diarization_map: dict = {}  # video_id → (dominant_speaker, segments)
    for meta in passed_metas:
        segs = diarizer.diarize(
            audio_path=meta.raw_audio_path,
            language=meta.language,
            min_confidence=cfg["diarization"]["min_confidence"],
        )
        dominant, filtered_segs = select_dominant_speaker(segs)
        diarization_map[meta.video_id] = (dominant, filtered_segs)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5: Candidate Segment Generation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 — Candidate Segment Generation")
    from tts_guj_eng_sarvam_dataset.src.diarization.diarize import generate_candidate_segments
    seg_cfg = cfg["segmentation"]
    all_candidates = []
    lang_counters = {"en-IN": 0, "hi-IN": 0, "gu-IN": 0}

    for meta in passed_metas:
        _, dia_segs = diarization_map.get(meta.video_id, (None, []))
        if not dia_segs:
            continue

        lang_key = meta.language.value.replace("-", "").lower()[:2]
        prefix = f"{lang_key}_{meta.video_id[:6]}"

        candidates = generate_candidate_segments(
            audio_path=meta.raw_audio_path,
            diarization_segments=dia_segs,
            language=meta.language,
            output_dir=seg_dir / lang_key,
            segment_prefix=prefix,
            min_duration=seg_cfg["min_duration"],
            max_duration=seg_cfg["max_duration"],
            pad_start=seg_cfg["pad_start_seconds"],
            pad_end=seg_cfg["pad_end_seconds"],
        )
        all_candidates.extend(candidates)
        lang_counters[meta.language.value] = (
            lang_counters.get(meta.language.value, 0) + len(candidates)
        )

    logger.info("Generated %d candidate segments: %s", len(all_candidates), lang_counters)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 6: Transcription
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 6 — Transcription")
    from tts_guj_eng_sarvam_dataset.src.transcription.transcribe import SarvamASR
    asr = SarvamASR(api_key=sarvam_key, base_url=sarvam_url)
    transcriptions = asr.transcribe_batch(
        all_candidates,
        min_confidence=cfg["transcription"]["min_asr_confidence"],
    )
    logger.info("Transcribed %d/%d segments", len(transcriptions), len(all_candidates))

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 7: Transcript Validation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 7 — Transcript Validation")
    from tts_guj_eng_sarvam_dataset.src.validation.validate import TranscriptValidator
    val_cfg = cfg["validation"]
    validator = TranscriptValidator(
        api_key=sarvam_key,
        base_url=sarvam_url,
        min_asr_confidence=val_cfg["min_transcript_confidence"],
        min_word_count=val_cfg["min_word_count"],
        max_word_count=val_cfg["max_word_count"],
    )
    validations = {}
    for seg_id, txn in transcriptions.items():
        validations[seg_id] = validator.validate(txn)

    passed_validations = {k: v for k, v in validations.items() if v.passed}
    logger.info("Validation passed: %d/%d", len(passed_validations), len(validations))

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 8: Audio Quality Assessment
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 8 — Audio Quality Assessment")
    from tts_guj_eng_sarvam_dataset.src.quality.assess import compute_audio_quality
    aq_cfg = cfg["audio_quality"]
    audio_metrics = {}
    candidate_map = {c.segment_id: c for c in all_candidates}

    for seg_id in passed_validations:
        candidate = candidate_map.get(seg_id)
        if not candidate:
            continue
        txn = transcriptions[seg_id]
        aqm = compute_audio_quality(
            audio_path=candidate.audio_path,
            segment_id=seg_id,
            transcript=txn.transcript,
            duration=candidate.duration,
            max_clipping_ratio=aq_cfg["max_clipping_ratio"],
            min_snr_db=aq_cfg["min_snr_db"],
            max_silence_ratio=aq_cfg["max_silence_ratio"],
            min_speech_rate_wpm=aq_cfg["min_speech_rate_wpm"],
            max_speech_rate_wpm=aq_cfg["max_speech_rate_wpm"],
        )
        audio_metrics[seg_id] = aqm

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 9: Emotion / Style Labeling
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 9 — Emotion / Style Labeling")
    from tts_guj_eng_sarvam_dataset.src.emotion.label import EmotionLabeler
    emo_cfg = cfg["emotion"]
    labeler = EmotionLabeler(
        api_key=sarvam_key,
        base_url=sarvam_url,
        acoustic_weight=emo_cfg["acoustic_weight"],
        llm_weight=emo_cfg["llm_weight"],
        min_confidence=emo_cfg["min_confidence"],
    )
    emotions = {}
    for seg_id in audio_metrics:
        candidate = candidate_map[seg_id]
        txn = transcriptions[seg_id]
        emotions[seg_id] = labeler.label(candidate, txn)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 11: Quality Scoring
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 11 — Quality Scoring")
    from tts_guj_eng_sarvam_dataset.src.quality.scorer import compute_final_score
    from tts_guj_eng_sarvam_dataset.src.schemas import ReviewStatus

    quality_scores = {}
    for seg_id in emotions:
        _, dia_segs = diarization_map.get(
            candidate_map[seg_id].video_id, (None, [])
        )
        dia_conf = max((d.confidence for d in dia_segs), default=0.5)
        quality_scores[seg_id] = compute_final_score(
            segment_id=seg_id,
            audio_metrics=audio_metrics[seg_id],
            validation=validations[seg_id],
            diarization_confidence=dia_conf,
            emotion=emotions[seg_id],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 10: Build Human Review Queue
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 10 — Building Human Review Queue")
    review_queue_path = Path(cfg_paths["output_dir"]) / "review_queue.jsonl"
    review_items = []

    for seg_id, qs in quality_scores.items():
        candidate  = candidate_map[seg_id]
        txn        = transcriptions[seg_id]
        val        = validations[seg_id]
        aqm        = audio_metrics[seg_id]
        emo        = emotions[seg_id]

        # Find video meta
        meta = next((m for m in passed_metas if m.video_id == candidate.video_id), None)

        item = {
            "segment_id":             seg_id,
            "audio_path":             candidate.audio_path,
            "language":               candidate.language.value,
            "duration":               candidate.duration,
            "transcript":             val.layer2_normalized_transcript,
            "emotion":                emo.emotion.value,
            "emotion_confidence":     emo.emotion_confidence,
            "style":                  emo.style,
            "asr_confidence":         txn.asr_confidence,
            "llm_quality_score":      val.layer4_llm_score,
            "llm_issues":             val.layer4_llm_issues,
            "estimated_snr_db":       aqm.estimated_snr_db,
            "clipping_ratio":         aqm.clipping_ratio,
            "silence_ratio":          aqm.silence_ratio,
            "speech_rate_wpm":        aqm.speech_rate_wpm,
            "audio_quality_score":    aqm.audio_quality_score,
            "transcript_quality_score": val.transcript_quality_score,
            "speaker_purity_score":   qs.speaker_purity_score,
            "final_score":            qs.final_score,
            "auto_decision":          qs.decision.value,
            "source":                 candidate.video_id,
            "video_title":            meta.title if meta else "unknown",
            "channel":                meta.channel if meta else "unknown",
        }
        review_items.append(item)

    with open(review_queue_path, "w") as f:
        for item in review_items:
            f.write(json.dumps(item) + "\n")

    logger.info("Review queue written: %s (%d items)", review_queue_path, len(review_items))
    logger.info("Run Streamlit dashboard: streamlit run src/review/dashboard.py -- %s",
                review_queue_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 12 & 13: Package + Publish (after human review)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("After human review, run:")
    logger.info("  python scripts/package_dataset.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pipeline.py videos.csv")
        sys.exit(1)
    main(sys.argv[1])
