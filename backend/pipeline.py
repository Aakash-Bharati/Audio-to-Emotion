"""
Analysis pipeline — orchestrates STT → Emotion Classification.

This is the single entry point for processing an audio file end-to-end.
It coordinates the STT and Emotion modules and returns a structured
result ready for the API layer to serialise as JSON.

EXTENSION POINTS
----------------
* **Speaker diarization** — Insert a diarization step between STT output
  and emotion classification. The `SegmentResult` dataclass already has
  an optional `speaker` field. See the marked location below.
* **PDF / report export** — After `analyze()` returns, a serialisation
  layer can convert `AnalysisResult` into a styled PDF. The structured
  dataclass makes this straightforward (e.g. with reportlab or weasyprint).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from stt import SpeechToText, TranscriptSegment
from emotion import EmotionClassifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------
@dataclass
class SegmentResult:
    """A single transcript segment enriched with emotion data."""

    start: float
    end: float
    text: str
    emotion: str
    confidence: float
    speaker: Optional[str] = field(default=None)  # future: diarization


@dataclass
class AnalysisResult:
    """Full analysis output returned to the frontend."""

    segments: list[SegmentResult]
    duration: float                         # total audio duration in seconds
    dominant_emotion: str                   # most frequent emotion label
    emotion_distribution: dict[str, float]  # emotion → percentage (0–100)


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------
class AnalysisPipeline:
    """
    Orchestrates the full transcription + emotion-classification flow.

    Both the STT engine and the emotion classifier are injected via
    constructor, making it easy to swap implementations or mock for tests.
    """

    def __init__(
        self,
        stt: Optional[SpeechToText] = None,
        emotion: Optional[EmotionClassifier] = None,
    ) -> None:
        self.stt = stt or SpeechToText()
        self.emotion = emotion or EmotionClassifier()

    def analyze(self, audio_path: str) -> AnalysisResult:
        """
        Run the full pipeline on an audio file.

        Steps
        -----
        1. **Transcribe** — Convert audio to timestamped text segments.
        2. *(Future: Diarize — assign speaker IDs to segments.)*
        3. **Classify** — Tag each segment with its dominant emotion.
        4. **Aggregate** — Compute summary statistics.

        Parameters
        ----------
        audio_path : str
            Path to the uploaded audio file.

        Returns
        -------
        AnalysisResult
        """

        # === STEP 1: Speech-to-Text ===
        logger.info("Step 1/3 — Transcribing audio …")
        transcript_segments: list[TranscriptSegment] = self.stt.transcribe(
            audio_path
        )

        if not transcript_segments:
            logger.warning("STT returned zero segments.")
            return AnalysisResult(
                segments=[],
                duration=0.0,
                dominant_emotion="neutral",
                emotion_distribution={"neutral": 100.0},
            )

        # === STEP 2 (FUTURE): Speaker Diarization ===
        # ----------------------------------------------------------
        # A diarization model (e.g. pyannote/speaker-diarization-3.1)
        # would run here, taking `audio_path` + `transcript_segments`
        # and populating each segment's `.speaker` field.
        #
        # from diarization import Diarizer  # future module
        # diarizer = Diarizer()
        # transcript_segments = diarizer.assign_speakers(
        #     audio_path, transcript_segments
        # )
        # ----------------------------------------------------------

        # === STEP 3: Emotion Classification ===
        logger.info(
            "Step 2/3 — Classifying emotions for %d segments …",
            len(transcript_segments),
        )
        texts = [seg.text for seg in transcript_segments]
        emotion_results = self.emotion.classify_batch(texts)

        # Merge STT + emotion results into final segments.
        segments: list[SegmentResult] = []
        for tseg, emo in zip(transcript_segments, emotion_results):
            segments.append(
                SegmentResult(
                    start=tseg.start,
                    end=tseg.end,
                    text=tseg.text,
                    emotion=emo.emotion,
                    confidence=emo.confidence,
                    speaker=tseg.speaker,
                )
            )

        # === STEP 4: Aggregate Statistics ===
        logger.info("Step 3/3 — Computing summary statistics …")
        duration = segments[-1].end if segments else 0.0

        # Emotion distribution as percentages.
        counter = Counter(seg.emotion for seg in segments)
        total = len(segments)
        emotion_distribution = {
            emo: round((count / total) * 100, 1)
            for emo, count in counter.most_common()
        }
        dominant_emotion = counter.most_common(1)[0][0]

        # === STEP 5 (FUTURE): PDF / Report Export ===
        # ----------------------------------------------------------
        # An export module could take the AnalysisResult and render
        # it as a styled PDF here. The structured dataclass makes
        # serialisation straightforward.
        #
        # from export import PdfExporter  # future module
        # PdfExporter().export(result, output_path="report.pdf")
        # ----------------------------------------------------------

        result = AnalysisResult(
            segments=segments,
            duration=duration,
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_distribution,
        )

        logger.info(
            "Analysis complete — %.1fs audio, %d segments, dominant=%s",
            result.duration,
            len(result.segments),
            result.dominant_emotion,
        )
        return result
