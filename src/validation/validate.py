
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


# ─── Stage 7: Transcript Validation ──────────────────────────────────────────
 
# LLM prompt for Sarvam transcript quality check
TRANSCRIPT_QA_PROMPT = """You are a professional linguist and speech data quality analyst.
 
You are given:
1. A transcript of a short audio clip (30–60 seconds)
2. The language of the speaker: {language}
 
Your task is to evaluate the transcript quality for training a Text-to-Speech (TTS) model.
 
Transcript:
"{transcript}"
 
Please respond in JSON (no markdown, no explanation outside the JSON):
{{
  "score": <float 0.0–1.0>,
  "is_complete_sentence": <true/false>,
  "has_asr_errors": <true/false>,
  "has_code_switching": <true/false>,
  "has_hallucination_risk": <true/false>,
  "issues": [<list of issue strings, empty if none>],
  "cleaned_transcript": "<corrected transcript or same if no corrections needed>"
}}
 
Scoring guide:
- 1.0: Perfect transcript, complete sentence, no issues
- 0.8–0.9: Minor issues (minor normalisation needed), still usable
- 0.5–0.7: Moderate issues (incomplete sentence, 1–2 word errors)
- 0.0–0.4: Major issues (heavy errors, hallucinations, wrong language)
"""
 
# Language detection patterns
LANG_PATTERNS = {
    Language.EN_IN: re.compile(r"[a-zA-Z]"),
    Language.HI_IN: re.compile(r"[\u0900-\u097F]"),
    Language.GU_IN: re.compile(r"[\u0A80-\u0AFF]"),
    Language.TA_IN: re.compile(r"[\u0B80-\u0BFF]"),
    Language.TE_IN: re.compile(r"[\u0C00-\u0C7F]"), 
    Language.KN_IN: re.compile(r"[\u0C80-\u0CFF]"),  
}
 
 
class TranscriptValidator:
    """4-layer transcript validation pipeline."""
 
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sarvam.ai",
        min_asr_confidence: float = 0.70,
        min_word_count: int = 5,
        max_word_count: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.min_asr_confidence = min_asr_confidence
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.session = requests.Session()
        self.session.headers.update({"api-subscription-key": api_key})
 
    # ── Layer 1: ASR Confidence ──────────────────────────────────────────────
 
    def _check_confidence(self, asr_confidence: float) -> bool:
        return asr_confidence >= self.min_asr_confidence
 
    # ── Layer 2: Transcript Normalization ────────────────────────────────────
 
    def _normalize(self, transcript: str, language: Language) -> str:
        """
        Normalize transcript for consistency:
        - Strip leading/trailing whitespace
        - Collapse multiple spaces
        - Normalize unicode (NFC form)
        - Remove disfluency markers if present (uh, um for English)
        - Ensure sentence ends with punctuation
        """
        text = transcript.strip()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text)
 
        if language == Language.EN_IN:
            # Remove common disfluencies that Sarvam may transcribe
            text = re.sub(r"\b(uh+|um+|hmm+|ah+)\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            # Ensure ends with terminal punctuation
            if text and text[-1] not in ".!?,;:":
                text += "."
 
        return text
 
    # ── Layer 3: Language Consistency ───────────────────────────────────────
 
    def _check_language(self, transcript: str, expected_language: Language) -> bool:
        """
        Check whether the transcript script matches the expected language.
        For en-IN: must contain Latin characters.
        For hi-IN: must contain Devanagari characters.
        Allows a small amount of the other script (code-switching is common in India).
        """
        pattern = LANG_PATTERNS.get(expected_language)
        if pattern is None:
            return True  # Unknown language: skip check
 
        primary_chars = len(pattern.findall(transcript))
        total_alpha   = len(re.findall(r"\w", transcript))
        if total_alpha == 0:
            return False
        ratio = primary_chars / total_alpha
        return ratio >= 0.6  # At least 60% expected-language characters
 
    # ── Layer 4: Sarvam LLM Quality Check ───────────────────────────────────
 
    def _llm_check(
        self, transcript: str, language: Language
    ) -> tuple[float, list[str], str]:
        """
        Call Sarvam LLM to evaluate transcript quality.
        Returns (score, issues, cleaned_transcript).
        """
        prompt = TRANSCRIPT_QA_PROMPT.format(
            language=language.value,
            transcript=transcript,
        )
 
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "sarvam-m",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.05,
                },
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
 
            # Parse JSON response
            import json
            parsed = json.loads(content.strip())
            score  = float(parsed.get("score", 0.5))
            issues = parsed.get("issues", [])
            cleaned = parsed.get("cleaned_transcript", transcript)
            return score, issues, cleaned
 
        except Exception as exc:
            logger.warning("LLM check failed: %s", exc)
            return 0.5, ["llm_check_failed"], transcript
 
    # ── Composite validation ─────────────────────────────────────────────────
 
    def validate(
        self,
        transcription: TranscriptionResult,
    ) -> ValidationResult:
        """Run all 4 layers and return a ValidationResult."""
        seg_id = transcription.segment_id
        transcript = transcription.transcript
        language   = transcription.language
 
        # Layer 1
        l1_ok = self._check_confidence(transcription.asr_confidence)
 
        # Layer 2
        normalized = self._normalize(transcript, language)
 
        # Word count checks
        word_count = len(normalized.split())
        if word_count < self.min_word_count or word_count > self.max_word_count:
            l1_ok = False
 
        # Layer 3
        l3_ok = self._check_language(normalized, language)
 
        # Layer 4
        llm_score, llm_issues, cleaned = self._llm_check(normalized, language)
 
        # Composite transcript quality score:
        # Layer 1 is a hard gate; if it fails, score is capped at 0.4
        l1_contrib = 1.0 if l1_ok else 0.0
        l3_contrib = 1.0 if l3_ok else 0.0
        transcript_score = (
            0.3 * l1_contrib
            + 0.1 * l3_contrib
            + 0.6 * llm_score
        )
        if not l1_ok:
            transcript_score = min(transcript_score, 0.4)
 
        passed = (
            l1_ok
            and l3_ok
            and llm_score >= 0.5
        )
 
        logger.debug(
            "[%s] validation: l1=%s l3=%s llm=%.2f → %.2f %s",
            seg_id, l1_ok, l3_ok, llm_score, transcript_score,
            "PASS" if passed else "FAIL"
        )
 
        return ValidationResult(
            segment_id=seg_id,
            layer1_confidence_ok=l1_ok,
            layer2_normalized_transcript=cleaned,
            layer3_language_ok=l3_ok,
            layer4_llm_score=llm_score,
            layer4_llm_issues=llm_issues,
            transcript_quality_score=transcript_score,
            passed=passed,
        )