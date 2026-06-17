"""
src/packaging/publish.py
Stage 12 — Package the reviewed dataset and publish to HuggingFace Hub.
Stage 13 — Generate dataset analytics figures.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from tts_guj_eng_sarvam_dataset.src.schemas import DatasetSample, ReviewStatus

logger = logging.getLogger(__name__)


# ─── Stage 12: Dataset Packaging ─────────────────────────────────────────────

DATASET_CARD_TEMPLATE = """\
---
language:
  - en
  - hi
license: cc-by-4.0
task_categories:
  - text-to-speech
  - automatic-speech-recognition
tags:
  - indian-english
  - hindi
  - tts
  - speech
  - emotion
  - sarvam
pretty_name: Indian TTS Dataset (en-IN + hi-IN)
size_categories:
  - 1K<n<10K
---

# Indian TTS Dataset

A high-quality Text-to-Speech training dataset of ~60 minutes total duration:
- ~30 minutes **Indian English** (en-IN)
- ~30 minutes **Hindi** (hi-IN)

## Dataset Details

| Field | Value |
|---|---|
| Total samples | {total_samples} |
| Total duration | {total_duration_min:.1f} minutes |
| en-IN samples | {en_samples} ({en_duration:.1f} min) |
| hi-IN samples | {hi_samples} ({hi_duration:.1f} min) |
| Accept rate | {accept_rate:.1%} |

## Schema

Each sample contains:

```python
{{
    "audio":              AudioFile,      # WAV, 16kHz mono
    "language":           str,            # "en-IN" or "hi-IN"
    "speaker_id":         str,            # e.g. "eng_spk_01"
    "duration":           float,          # seconds
    "transcript":         str,            # cleaned text
    "emotion":            str,            # emotion label
    "emotion_confidence": float,          # 0–1
    "style":              str,            # speaking style
    "source":             str,            # YouTube video ID
    "asr_confidence":     float,          # Sarvam ASR confidence
    "segment_id":         str,
    "channel":            str,
    "video_title":        str,
    "rms_energy_db":      float,
    "estimated_snr_db":   float,
    "speech_rate_wpm":    float,
    "clipping_ratio":     float,
    "silence_ratio":      float,
    "llm_quality_score":  float,
    "final_score":        float,
}}
```

## Pipeline

Built with the **TTS Dataset Pipeline** using:
- `yt-dlp` for audio ingestion
- `ffmpeg` for WAV extraction (16kHz mono)
- **Sarvam Diarization API** for speaker separation
- **Sarvam ASR (saarika:v2)** for transcription
- **Sarvam LLM (sarvam-m)** for transcript QA and emotion labeling
- `librosa` for acoustic feature extraction
- Human review via Streamlit dashboard

## Quality Scoring

| Component | Weight |
|---|---|
| Audio quality (SNR, clipping, silence) | 35% |
| Transcript quality (ASR + LLM QA) | 35% |
| Speaker purity (diarization confidence) | 20% |
| Emotion confidence | 10% |

**Accept threshold:** ≥ 0.75  
**Review threshold:** ≥ 0.55  

## Emotion Distribution

{emotion_table}

## Citation

```bibtex
@dataset{{indian_tts_dataset,
    title={{Indian TTS Dataset (en-IN + hi-IN)}},
    year={{2024}},
    note={{Created with the TTS Dataset Pipeline using Sarvam APIs}}
}}
```
"""


def build_dataset_dataframe(
    samples: list[DatasetSample],
    accepted_ids: set[str],
    dataset_dir: Path,
) -> pd.DataFrame:
    """
    Convert accepted DatasetSamples to a Pandas DataFrame,
    copying audio files into the dataset directory structure.
    """
    rows: list[dict] = []
    lang_dirs = {"en-IN": dataset_dir / "clips" / "english",
                 "hi-IN": dataset_dir / "clips" / "hindi"}
    for ld in lang_dirs.values():
        ld.mkdir(parents=True, exist_ok=True)

    for s in samples:
        if s.segment_id not in accepted_ids:
            continue

        lang_key = s.language.value
        dest_dir = lang_dirs.get(lang_key, dataset_dir / "clips" / "other")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy audio file with dataset-relative path
        src = Path(s.audio)
        dst = dest_dir / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

        relative_audio = str(dst.relative_to(dataset_dir))

        row = s.model_dump()
        row["audio"] = relative_audio
        row["language"] = s.language.value
        row["emotion"] = s.emotion.value
        row["review_status"] = s.review_status.value
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_dataset_card(
    df: pd.DataFrame,
    output_path: Path,
) -> str:
    """Generate and write the HuggingFace dataset card (README.md)."""
    en_df = df[df["language"] == "en-IN"]
    hi_df = df[df["language"] == "hi-IN"]

    # Emotion distribution table
    emotion_counts = df["emotion"].value_counts()
    emotion_table = "| Emotion | Count | % |\n|---|---|---|\n"
    for emo, cnt in emotion_counts.items():
        pct = cnt / len(df) * 100
        emotion_table += f"| {emo} | {cnt} | {pct:.1f}% |\n"

    card = DATASET_CARD_TEMPLATE.format(
        total_samples=len(df),
        total_duration_min=df["duration"].sum() / 60,
        en_samples=len(en_df),
        en_duration=en_df["duration"].sum() / 60,
        hi_samples=len(hi_df),
        hi_duration=hi_df["duration"].sum() / 60,
        accept_rate=len(df) / max(len(df) + 1, 1),
        emotion_table=emotion_table,
    )

    output_path.write_text(card, encoding="utf-8")
    logger.info("Dataset card written to %s", output_path)
    return card


def publish_to_huggingface(
    df: pd.DataFrame,
    dataset_dir: Path,
    repo_id: str,
    hf_token: str,
    private: bool = False,
) -> None:
    """
    Upload dataset to HuggingFace Hub.
    Requires: `pip install datasets huggingface_hub`
    """
    try:
        from datasets import Audio, Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("Install 'datasets' and 'huggingface_hub': pip install datasets huggingface_hub")
        return

    api = HfApi(token=hf_token)

    # Create or ensure repo exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        logger.info("HF repo: %s", repo_id)
    except Exception as exc:
        logger.error("Failed to create HF repo: %s", exc)
        return

    # Build HF Dataset with audio column
    # Audio paths must be absolute for datasets to resolve them
    df_upload = df.copy()
    df_upload["audio"] = df_upload["audio"].apply(
        lambda p: str((dataset_dir / p).resolve())
    )

    # Drop word_timestamps for parquet compatibility (nested list of dicts)
    if "word_timestamps" in df_upload.columns:
        df_upload["word_timestamps"] = df_upload["word_timestamps"].apply(json.dumps)

    hf_dataset = Dataset.from_pandas(df_upload)
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split by language
    en_ds = hf_dataset.filter(lambda x: x["language"] == "en-IN")
    hi_ds = hf_dataset.filter(lambda x: x["language"] == "hi-IN")

    dataset_dict = DatasetDict({
        "english": en_ds,
        "hindi":   hi_ds,
        "all":     hf_dataset,
    })

    dataset_dict.push_to_hub(
        repo_id,
        token=hf_token,
        private=private,
    )

    # Upload dataset card
    card_path = dataset_dir / "README.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

    logger.info("✅ Dataset published: https://huggingface.co/datasets/%s", repo_id)


# ─── Stage 13: Analytics ──────────────────────────────────────────────────────

def generate_analytics(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate publication-ready analytics figures using matplotlib.
    Saves PNG files to output_dir/figures/.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning("matplotlib not installed; skipping analytics figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    # 1. Language distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    lang_counts = df["language"].value_counts()
    bars = ax.bar(lang_counts.index, lang_counts.values, color=["#1D9E75", "#7F77DD"])
    ax.bar_label(bars)
    ax.set_title("Language distribution")
    ax.set_ylabel("Segments")
    fig.tight_layout()
    fig.savefig(fig_dir / "language_distribution.png")
    plt.close(fig)

    # 2. Emotion distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    emotion_counts = df["emotion"].value_counts()
    bars = ax.barh(emotion_counts.index, emotion_counts.values, color="#534AB7")
    ax.bar_label(bars, padding=3)
    ax.set_title("Emotion distribution")
    ax.set_xlabel("Segments")
    fig.tight_layout()
    fig.savefig(fig_dir / "emotion_distribution.png")
    plt.close(fig)

    # 3. Duration distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["duration"], bins=20, color="#1D9E75", edgecolor="white")
    ax.set_title("Segment duration distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Frequency")
    ax.axvline(df["duration"].mean(), color="orange", linestyle="--",
               label=f"Mean: {df['duration'].mean():.1f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "duration_distribution.png")
    plt.close(fig)

    # 4. Quality score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["final_score"], bins=20, color="#534AB7", edgecolor="white")
    ax.axvline(0.75, color="green",  linestyle="--", label="Accept (0.75)")
    ax.axvline(0.55, color="orange", linestyle="--", label="Review (0.55)")
    ax.set_title("Final quality score distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "quality_distribution.png")
    plt.close(fig)

    # 5. SNR distribution per language
    fig, ax = plt.subplots(figsize=(6, 4))
    for lang, color in [("en-IN", "#1D9E75"), ("hi-IN", "#534AB7")]:
        subset = df[df["language"] == lang]["estimated_snr_db"]
        if len(subset):
            ax.hist(subset, bins=15, alpha=0.7, label=lang, color=color, edgecolor="white")
    ax.set_title("SNR distribution by language")
    ax.set_xlabel("Estimated SNR (dB)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "snr_by_language.png")
    plt.close(fig)

    logger.info("Analytics figures saved to %s", fig_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total accepted samples:   {len(df)}")
    print(f"Total duration:           {df['duration'].sum() / 60:.1f} minutes")
    print(f"Avg segment duration:     {df['duration'].mean():.1f}s")
    print(f"Avg ASR confidence:       {df['asr_confidence'].mean():.3f}")
    print(f"Avg quality score:        {df['final_score'].mean():.3f}")
    print(f"Avg SNR:                  {df['estimated_snr_db'].mean():.1f} dB")
    print("\nEmotion distribution:")
    for emo, cnt in df["emotion"].value_counts().items():
        print(f"  {emo:<18} {cnt:>4} ({cnt/len(df)*100:.1f}%)")
    print("=" * 60)