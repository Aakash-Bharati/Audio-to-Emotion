"""
Centralized configuration for the Transcript Emotion Mapper backend.

All tuneable values are loaded from environment variables (or a .env file)
so the ML models, device settings, and server options can be changed without
touching code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above /backend)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Speech-to-Text (faster-whisper) settings
# ---------------------------------------------------------------------------
# Model identifier — points to a CTranslate2-compatible Whisper checkpoint on
# Hugging Face. "large-v3-turbo" is the current best speed/accuracy tradeoff.
STT_MODEL: str = os.getenv(
    "STT_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2"
)

# Device for STT inference: "cpu", "cuda", or "auto" (auto-detect).
STT_DEVICE: str = os.getenv("STT_DEVICE", "auto")

# Compute type for CTranslate2: "float16", "int8", "int8_float16", or "auto".
STT_COMPUTE_TYPE: str = os.getenv("STT_COMPUTE_TYPE", "float16")

# Whether to automatically fall back to CPU + int8 if CUDA/cuBLAS is unavailable.
STT_CPU_FALLBACK: bool = os.getenv("STT_CPU_FALLBACK", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Emotion classification settings
# ---------------------------------------------------------------------------
# Hugging Face model identifier for the emotion classifier.
# Default: j-hartmann/emotion-english-distilroberta-base (7-class Ekman model).
# Easy to swap — just change this env var to another HF model name.
EMOTION_MODEL: str = os.getenv(
    "EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
)

# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# Maximum upload size in megabytes.
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "100"))

# Allowed audio file extensions.
ALLOWED_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a"}
