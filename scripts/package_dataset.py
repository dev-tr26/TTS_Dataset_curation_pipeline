"""
scripts/package_dataset.py
Run AFTER human review is complete to package and publish the dataset.

Usage:
    python scripts/package_dataset.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("package")


def main() -> None:
    import yaml
    from src.packaging.publish import (
        build_dataset_dataframe,
        generate_analytics,
        generate_dataset_card,
        publish_to_huggingface,
    )

    with open(ROOT / "configs" / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)

    output_dir  = Path(cfg["paths"]["output_dir"])
    dataset_dir = Path(cfg["paths"]["dataset_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load review queue
    queue_path = output_dir / "review_queue.jsonl"
    decisions_path = output_dir / "review_decisions.jsonl"

    if not queue_path.exists():
        logger.error("No review queue found. Run run_pipeline.py first.")
        sys.exit(1)

    queue: list[dict] = []
    with open(queue_path) as f:
        for line in f:
            if line.strip():
                queue.append(json.loads(line))

    # Load human review decisions
    human_decisions: dict[str, str] = {}
    if decisions_path.exists():
        with open(decisions_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    human_decisions[rec["segment_id"]] = rec["decision"]

    # Determine accepted set:
    # Human decision takes priority; fallback to auto-decision
    accepted_ids: set[str] = set()
    for item in queue:
        seg_id = item["segment_id"]
        decision = human_decisions.get(seg_id, item.get("auto_decision", "rejected"))
        if decision == "accepted":
            accepted_ids.add(seg_id)

    logger.info(
        "Accepted: %d / %d (%.1f%%)",
        len(accepted_ids), len(queue), len(accepted_ids) / max(len(queue), 1) * 100
    )

    # Build DatasetSample objects for accepted items
    from src.schemas import DatasetSample, Emotion, Language, ReviewStatus
    samples: list[DatasetSample] = []
    for item in queue:
        if item["segment_id"] not in accepted_ids:
            continue
        try:
            sample = DatasetSample(
                audio=item["audio_path"],
                language=Language(item["language"]),
                speaker_id=item.get("source", "unknown")[:6] + "_spk",
                duration=item["duration"],
                transcript=item["transcript"],
                emotion=Emotion(item["emotion"]),
                emotion_confidence=item["emotion_confidence"],
                style=item["style"],
                source=item["source"],
                asr_confidence=item["asr_confidence"],
                segment_id=item["segment_id"],
                channel=item["channel"],
                video_title=item["video_title"],
                rms_energy_db=item.get("rms_energy_db", 0.0),
                estimated_snr_db=item["estimated_snr_db"],
                speech_rate_wpm=item["speech_rate_wpm"],
                clipping_ratio=item["clipping_ratio"],
                silence_ratio=item["silence_ratio"],
                word_timestamps=[],
                llm_quality_score=item["llm_quality_score"],
                final_score=item["final_score"],
                review_status=ReviewStatus.ACCEPTED,
            )
            samples.append(sample)
        except Exception as exc:
            logger.warning("Skipping %s: %s", item["segment_id"], exc)

    if not samples:
        logger.error("No accepted samples found. Check review decisions.")
        sys.exit(1)

    # Stage 12: Build DataFrame + copy files
    logger.info("Building dataset DataFrame...")
    df = build_dataset_dataframe(samples, accepted_ids, dataset_dir)

    # Save metadata CSV and JSONL
    df.to_csv(dataset_dir / "metadata.csv", index=False)
    df.to_json(dataset_dir / "metadata.jsonl", orient="records", lines=True)
    logger.info("Metadata saved: %s", dataset_dir / "metadata.csv")

    # Generate dataset card
    generate_dataset_card(df, dataset_dir / "README.md")

    # Stage 13: Analytics
    logger.info("Generating analytics...")
    generate_analytics(df, dataset_dir)

    # Publish to HuggingFace
    import os
    hf_token  = os.environ.get("HF_TOKEN", "")
    hf_repo   = os.environ.get("HF_REPO_ID", "")
    if hf_token and hf_repo:
        logger.info("Publishing to HuggingFace: %s", hf_repo)
        publish_to_huggingface(df, dataset_dir, hf_repo, hf_token)
    else:
        logger.warning("HF_TOKEN or HF_REPO_ID not set; skipping HuggingFace upload")
        logger.info("Dataset files ready at: %s", dataset_dir)


if __name__ == "__main__":
    main()
