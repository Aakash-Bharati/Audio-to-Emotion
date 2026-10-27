"""
Speech-to-Text wrapper using faster-whisper (CTranslate2 backend).

This module owns all STT logic and exposes a single `SpeechToText` class.
The rest of the pipeline interacts only through its public interface, so
the underlying STT engine can be swapped (e.g. to whisper.cpp, cloud APIs)
without touching pipeline code.

EXTENSION POINTS
----------------
* **Speaker diarization** — After `transcribe()` returns raw segments, a
  diarization step (e.g. pyannote-audio) can be inserted to tag each
  segment with a `speaker_id`. The `TranscriptSegment` dataclass already
  reserves a slot for this field.
* **Real-time streaming** — Replace the file-based `transcribe()` call
  with a chunked VAD + partial-result loop. faster-whisper supports this
  via its `transcribe()` generator, which yields segments as they are
  decoded. A WebSocket endpoint in main.py would drive this mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single timestamped chunk of transcribed text."""

    start: float  # seconds from audio start
    end: float
    text: str
    # Reserved for future diarization — will hold "Speaker 1", etc.
    speaker: Optional[str] = field(default=None)


class SpeechToText:
    """
    Thin wrapper around faster-whisper's WhisperModel.

    The model is loaded lazily on the first call to `transcribe()` so that
    importing this module is cheap during startup.
    """

    def __init__(
        self,
        model_name: str = config.STT_MODEL,
        device: str = config.STT_DEVICE,
        compute_type: str = config.STT_COMPUTE_TYPE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------
    def _load_model(self) -> WhisperModel:
        """Load the Whisper model on first use, with automatic CPU fallback."""
        if self._model is None:
            logger.info(
                "Loading STT model '%s' on device=%s, compute_type=%s …",
                self._model_name,
                self._device,
                self._compute_type,
            )
            try:
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                logger.info("STT model loaded successfully.")
            except (RuntimeError, Exception) as exc:
                # Catch CUDA/cuBLAS failures like:
                #   RuntimeError: Library cublas64_12.dll is not found
                if config.STT_CPU_FALLBACK and self._device != "cpu":
                    logger.warning(
                        "GPU/CUDA load failed (%s). Falling back to "
                        "device='cpu', compute_type='int8' …",
                        exc,
                    )
                    self._device = "cpu"
                    self._compute_type = "int8"
                    self._model = WhisperModel(
                        self._model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                    logger.info(
                        "STT model loaded successfully on CPU (int8 fallback)."
                    )
                else:
                    raise
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        """
        Transcribe an audio file into timestamped text segments.

        Parameters
        ----------
        audio_path : str
            Absolute path to the audio file (.mp3, .wav, .m4a).

        Returns
        -------
        list[TranscriptSegment]
            Ordered list of segments with start/end times and text.
        """
        model = self._load_model()

        try:
            segments_iter, info = self._run_transcribe(model, audio_path)
        except RuntimeError as exc:
            # CUDA/cuBLAS failures surface here (e.g. cublas64_12.dll missing)
            # because model.encode() is called lazily during transcription.
            if config.STT_CPU_FALLBACK and self._device != "cpu":
                logger.warning(
                    "GPU inference failed (%s). Reloading model on CPU …", exc
                )
                self._device = "cpu"
                self._compute_type = "int8"
                self._model = None  # force reload
                model = self._load_model()
                segments_iter, info = self._run_transcribe(model, audio_path)
            else:
                raise

        logger.info(
            "Detected language '%s' (prob %.2f), duration %.1fs",
            info.language,
            info.language_probability,
            info.duration,
        )

        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg.text.strip(),
                )
            )

        # -----------------------------------------------------------
        # EXTENSION: Diarization would run here.
        # Given `segments` + the original audio, a diarization model
        # (e.g. pyannote/speaker-diarization-3.1) would assign a
        # `speaker` label to each segment.
        # -----------------------------------------------------------

        logger.info("Transcription complete — %d segments.", len(segments))
        return segments

    @staticmethod
    def _run_transcribe(model: WhisperModel, audio_path: str):
        """Run model.transcribe with standard parameters.

        Separated so it can be retried on CPU fallback without
        duplicating beam_size / VAD settings.
        """
        # beam_size=5 balances quality vs speed; VAD filter removes silence.
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        # Force-materialise at least the language detection (which triggers
        # the first encoder call) so CUDA errors surface immediately.
        _ = info.language
        return segments_iter, info

    @property
    def duration_info(self) -> Optional[float]:
        """Return the duration of the last transcribed audio (set by info)."""
        return None  # caller gets duration from pipeline.py via info
