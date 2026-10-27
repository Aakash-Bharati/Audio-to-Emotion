"""
FastAPI application — REST API for the Transcript Emotion Mapper.

Endpoints
---------
POST /api/analyze   Upload an audio file → get emotion-tagged transcript.
GET  /api/health    Simple health check.

EXTENSION POINTS
----------------
* **WebSocket /api/stream** — For real-time streaming transcription,
  add a WebSocket endpoint that accepts audio chunks and yields partial
  transcript segments as they are decoded. The `SpeechToText` class
  already supports generator-based transcription via faster-whisper.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import config
from pipeline import AnalysisPipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Transcript Emotion Mapper API",
    version="1.0.0",
    description="Transcribe audio and classify each segment's emotion.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline singleton (lazy — models loaded on first request)
# ---------------------------------------------------------------------------
_pipeline: AnalysisPipeline | None = None


def get_pipeline() -> AnalysisPipeline:
    """Return (and lazily create) the singleton analysis pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AnalysisPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Health check — returns 200 if the server is up."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Sample audio — serve the bundled reference clip
# ---------------------------------------------------------------------------
_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "reference"
_SAMPLE_FILE = _SAMPLE_DIR / "freesound_community-female-voice-and-this-is-how-it-ends-26030.mp3"


@app.get("/api/sample")
async def get_sample():
    """Serve the bundled sample audio file for the 'Try a Sample Clip' button."""
    if not _SAMPLE_FILE.exists():
        raise HTTPException(status_code=404, detail="Sample audio not found.")
    return FileResponse(
        path=str(_SAMPLE_FILE),
        media_type="audio/mpeg",
        filename=_SAMPLE_FILE.name,
    )


@app.post("/api/analyze-sample")
async def analyze_sample():
    """
    Run the full analysis pipeline on the bundled sample audio.
    No upload needed — used by the 'Try a Sample Clip' button.
    """
    if not _SAMPLE_FILE.exists():
        raise HTTPException(status_code=404, detail="Sample audio not found.")

    try:
        size_mb = _SAMPLE_FILE.stat().st_size / (1024 * 1024)
        logger.info(
            "Analyzing sample '%s' (%.1f MB) …", _SAMPLE_FILE.name, size_mb
        )
        pipeline = get_pipeline()
        result = pipeline.analyze(str(_SAMPLE_FILE))
        return JSONResponse(content=asdict(result))

    except Exception as exc:
        logger.exception("Pipeline failed for sample '%s'", _SAMPLE_FILE.name)
        raise HTTPException(
            status_code=500,
            detail=f"Sample analysis failed: {exc}",
        ) from exc


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept an audio file upload, run the full STT + emotion pipeline,
    and return the structured analysis result as JSON.

    Request
    -------
    Content-Type: multipart/form-data
    Field: file (audio file, .mp3 / .wav / .m4a)

    Response (200)
    --------
    {
      "segments": [ { "start", "end", "text", "emotion", "confidence" } ],
      "duration": float,
      "dominant_emotion": str,
      "emotion_distribution": { emotion: percentage }
    }
    """

    # --- Validate file extension ---
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
            ),
        )

    # --- Read and validate file size ---
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed: {config.MAX_UPLOAD_MB} MB."
            ),
        )

    # --- Save to temp file (faster-whisper needs a file path) ---
    tmp_dir = tempfile.mkdtemp(prefix="emotion_mapper_")
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        logger.info(
            "Received '%s' (%.1f MB) — starting analysis …", filename, size_mb
        )

        # --- Run pipeline ---
        pipeline = get_pipeline()
        result = pipeline.analyze(tmp_path)

        return JSONResponse(content=asdict(result))

    except Exception as exc:
        logger.exception("Pipeline failed for '%s'", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    finally:
        # Clean up temp file.
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point (for `python main.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
