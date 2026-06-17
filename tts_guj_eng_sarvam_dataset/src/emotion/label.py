from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from sarvamai import SarvamAI
import requests
import os 
from dotenv import load_dotenv

# =============================================================================
# CONFIG
# =============================================================================

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")

BASE_URL = "https://api.sarvam.ai"

ROOT_DIR = Path(".")

GUJARATI_JSON = Path(
    r"C:\Desktop\github_projects\tts_pipeline\transcripts\gujarati\gu_transcripts.json"
)

ENGLISH_JSON = Path(
    r"C:\Desktop\github_projects\tts_pipeline\transcripts\english\en_transcripts.json"
)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# LABELS
# =============================================================================

EMOTION_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "excited",
    "serious",
    "conversational",
    "storytelling",
    "inspirational",
    "humorous",
]

STYLE_LABELS = [
    "formal",
    "conversational",
    "storytelling",
    "instructional",
    "argumentative",
]


# =============================================================================
# PROMPT
# =============================================================================

EMOTION_PROMPT = """
You are an expert annotator and speech analyst for a multilingual TTS (Text-to-Speech) dataset.

Analyse the following transcript and determine the speaker's emotion and speaking style.
Task:
Determine the most appropriate emotion and speaking style.

Language: {language}

Transcript:
{transcript}

Emotion labels:
{emotion_labels}

Style labels:
formal
conversational
storytelling
instructional
argumentative

Return ONLY valid JSON.

{{
  "emotion": "<emotion>",
  "style": "<style>",
  "confidence": <float between 0 and 1>
}}
"""


# =============================================================================
# LABELER
# =============================================================================

class EmotionLabeler:

    def __init__(self, api_key: str):

        self.client = SarvamAI(
            api_subscription_key=api_key
        )

    def label(
        self,
        transcript: str,
        language: str,
    ) -> dict:

        prompt = EMOTION_PROMPT.format(
            language=language,
            transcript=transcript,
            emotion_labels=", ".join(EMOTION_LABELS),
        )

        try:

            response = self.client.chat.completions(
                model="sarvam-105b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            content = response.choices[0].message.content.strip()

            content = (
                content.replace("```json", "")
                .replace("```", "")
                .strip()
            )

            parsed = json.loads(content)

            return {
                "emotion": parsed.get(
                    "emotion",
                    "neutral",
                ),
                "emotion_confidence": float(
                    parsed.get(
                        "confidence",
                        0.5,
                    )
                ),
                "style": parsed.get(
                    "style",
                    "conversational",
                ),
            }

        except Exception as exc:

            logger.exception(
                "Labeling failed"
            )

            return {
                "emotion": "neutral",
                "emotion_confidence": 0.3,
                "style": "conversational",
            }

# =============================================================================
# PROCESS FILE
# =============================================================================

def process_json_file(
    json_path: Path,
    language: str,
    labeler: EmotionLabeler,
):

    logger.info(
        "Processing %s",
        json_path,
    )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    entries = data["diarized_transcript"]["entries"]

    total = len(entries)

    logger.info(
        "Found %d entries",
        total,
    )

    for idx, entry in enumerate(entries, start=1):

        if (
            "emotion" in entry
            and "emotion_confidence" in entry
            and "style" in entry
        ):
            continue

        transcript = entry.get(
            "transcript",
            "",
        ).strip()

        if not transcript:
            continue

        result = labeler.label(
            transcript=transcript,
            language=language,
        )

        entry["emotion"] = result["emotion"]
        entry["emotion_confidence"] = result[
            "emotion_confidence"
        ]
        entry["style"] = result["style"]

        logger.info(
            "[%d/%d] %s -> %s (%.2f)",
            idx,
            total,
            entry.get(
                "segment_id",
                "unknown",
            ),
            result["emotion"],
            result["emotion_confidence"],
        )

        # Save progress every 20 entries
        if idx % 20 == 0:

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(
                "Progress saved (%d/%d)",
                idx,
                total,
            )

        time.sleep(0.5)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Completed %s",
        json_path,
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    labeler = EmotionLabeler(
        api_key=API_KEY,
    )
    logger.info("Testing Sarvam...")

    print(labeler.label(transcript="Hello, how are you today?",language="English"))

    process_json_file(
        json_path=GUJARATI_JSON,
        language="Gujarati",
        labeler=labeler,
    )

    process_json_file(
        json_path=ENGLISH_JSON,
        language="English",
        labeler=labeler,
    )

    logger.info("All done.")


if __name__ == "__main__":
    main()