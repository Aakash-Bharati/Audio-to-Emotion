# Transcript Emotion Mapper

An AI-powered tool that transcribes audio recordings (customer service calls, podcasts, meetings) and renders the transcript **color-coded by the speaker's shifting emotion over time**, alongside an **emotional-arc chart** of the conversation.

![EmotionMapper](./reference/ui-reference.png.png)

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Frontend                    │
│        Next.js · TypeScript · Tailwind       │
│                                              │
│  Upload ──→ fetch POST /api/analyze ──→ UI   │
│              (multipart/form-data)           │
└─────────────────┬───────────────────────────┘
                  │ HTTP
┌─────────────────▼───────────────────────────┐
│                  Backend                     │
│               FastAPI (Python)               │
│                                              │
│  main.py ──→ pipeline.py                     │
│                 ├─ stt.py (faster-whisper)    │
│                 └─ emotion.py (HF transformer)│
└─────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 15 (App Router), TypeScript, Tailwind CSS, Recharts, lucide-react |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **STT** | faster-whisper (`large-v3-turbo` via CTranslate2) |
| **Emotion** | HuggingFace Transformers (`j-hartmann/emotion-english-distilroberta-base`) |

## Prerequisites

- **Python** 3.10 or later
- **Node.js** 18 or later
- **NVIDIA GPU** with CUDA (recommended) — or CPU (will be slower)
- ~4 GB disk space for ML models (downloaded on first run)

## Setup & Run

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy and edit config
cp ../.env.example .env

# Start the API server
python main.py
# → Server runs at http://localhost:8000
# → First request downloads ML models (~2-3 GB)
```

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
# → App runs at http://localhost:3000
```

### 3. Use the App

1. Open **http://localhost:3000** in your browser.
2. Click **"Upload Audio"** and select an `.mp3`, `.wav`, or `.m4a` file.
3. Wait for the transcription + emotion classification to complete.
4. View the **color-coded transcript**, **emotional-arc chart**, and **emotion distribution**.

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Whisper model (CTranslate2 format) |
| `STT_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `STT_COMPUTE_TYPE` | `float16` | `float16`, `int8`, `int8_float16`, `auto` |
| `EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | HuggingFace emotion classifier |
| `API_PORT` | `8000` | Backend server port |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend → Backend URL |

## Emotion Color Map

| Emotion | Color | Hex |
|---------|-------|-----|
| Anger | 🔴 Red | `#ef4444` |
| Disgust | 🟢 Lime | `#84cc16` |
| Fear | 🟣 Purple | `#a855f7` |
| Joy | 🟡 Yellow | `#eab308` |
| Neutral | ⚪ Gray | `#6b7280` |
| Sadness | 🔵 Blue | `#3b82f6` |
| Surprise | 🟠 Orange | `#f97316` |

## Project Structure

```
├── backend/
│   ├── main.py         # FastAPI app + routes
│   ├── pipeline.py     # Orchestrates STT → Emotion flow
│   ├── stt.py          # Speech-to-Text wrapper (faster-whisper)
│   ├── emotion.py      # Emotion classifier wrapper (HuggingFace)
│   ├── config.py       # Settings from environment
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # Main page
│   │   │   ├── layout.tsx     # Root layout
│   │   │   └── globals.css    # Design tokens + base styles
│   │   ├── components/
│   │   │   ├── Navbar.tsx
│   │   │   ├── HeroBadges.tsx
│   │   │   ├── HeroSection.tsx
│   │   │   ├── EmotionPulseCard.tsx
│   │   │   ├── StepStrip.tsx
│   │   │   ├── UploadModal.tsx
│   │   │   ├── TranscriptPanel.tsx
│   │   │   ├── EmotionArcChart.tsx
│   │   │   ├── EmotionLegend.tsx
│   │   │   └── EmotionDistribution.tsx
│   │   ├── lib/
│   │   │   └── constants.ts   # Colors, API URL, helpers
│   │   └── types/
│   │       └── index.ts       # TypeScript interfaces
│   └── package.json
├── .env.example
└── README.md
```

## Future Extensions (Scaffolded)

The codebase is structured so these can be added without a rewrite:

- **Speaker Diarization** — Extension point in `stt.py` and `pipeline.py` (between STT and emotion classification)
- **Real-time Streaming** — Extension point in `stt.py` (chunked VAD + WebSocket in `main.py`)
- **Fine-tuning** — `emotion.py` accepts any local model checkpoint path
- **PDF Report Export** — Extension point in `pipeline.py` (post-analysis serialisation)

## License

MIT
