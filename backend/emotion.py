"""
Emotion classification wrapper using a Hugging Face transformer model.

This module owns all emotion-detection logic and exposes a single
`EmotionClassifier` class. The pipeline code interacts only through its
public interface, so the underlying model can be swapped (different HF
checkpoint, a custom fine-tuned model, or even a cloud API) without
touching orchestration logic.

EXTENSION POINTS
----------------
* **Fine-tuning** — After training a custom model on domain-specific data
  (e.g. customer-service call transcripts), point the `EMOTION_MODEL`
  config value to the local checkpoint path. The HuggingFace pipeline
  loader accepts both Hub identifiers and local directories.
* **Multi-label classification** — The current setup uses `top_k=1` for
  a single dominant emotion per segment. To support multi-label output
  (e.g. "frustrated AND sad"), switch to a BCE-trained model and change
  `top_k` / apply a threshold to the returned scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from transformers import pipeline as hf_pipeline

import config

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Classification result for a single text segment."""

    emotion: str      # e.g. "anger", "joy", "neutral"
    confidence: float  # 0.0 – 1.0


class EmotionClassifier:
    """
    Thin wrapper around a HuggingFace text-classification pipeline
    configured for emotion detection.

    The model is loaded lazily on the first call to `classify()`.
    """

    def __init__(self, model_name: str = config.EMOTION_MODEL) -> None:
        self._model_name = model_name
        self._pipe = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load the HuggingFace classification pipeline on first use."""
        if self._pipe is None:
            logger.info("Loading emotion model '%s' …", self._model_name)
            self._pipe = hf_pipeline(
                "text-classification",
                model=self._model_name,
                top_k=1,          # we only need the dominant emotion
                truncation=True,  # auto-truncate inputs > model max length
            )
            logger.info("Emotion model loaded successfully.")
        return self._pipe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, text: str) -> EmotionResult:
        """
        Classify a single text string into an emotion label.

        Parameters
        ----------
        text : str
            The transcript segment text to classify.

        Returns
        -------
        EmotionResult
            The dominant emotion label and its confidence score.
        """
        pipe = self._load_model()

        if not text or not text.strip():
            return EmotionResult(emotion="neutral", confidence=1.0)

        results = pipe(text)
        # `top_k=1` returns [[{label, score}]]
        top = results[0][0]
        return EmotionResult(
            emotion=top["label"].lower(),
            confidence=round(top["score"], 4),
        )

    def classify_batch(self, texts: list[str]) -> list[EmotionResult]:
        """
        Classify a batch of texts for better throughput.

        Parameters
        ----------
        texts : list[str]
            List of transcript segment texts.

        Returns
        -------
        list[EmotionResult]
            Emotion results in the same order as input texts.
        """
        pipe = self._load_model()

        # Replace empty strings with a fallback so the model doesn't choke.
        cleaned = [t if t and t.strip() else "." for t in texts]

        batch_results = pipe(cleaned, batch_size=16)

        out: list[EmotionResult] = []
        for i, results in enumerate(batch_results):
            if not texts[i] or not texts[i].strip():
                out.append(EmotionResult(emotion="neutral", confidence=1.0))
            else:
                top = results[0]
                out.append(
                    EmotionResult(
                        emotion=top["label"].lower(),
                        confidence=round(top["score"], 4),
                    )
                )
        return out
