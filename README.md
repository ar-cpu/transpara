# Transpara

Political bias and extremism detection system using machine learning.

## Overview

Transpara analyzes text, audio, video, and documents to detect political bias (left/center/right) and extremist rhetoric. It combines a Random Forest classifier for political leaning with pattern-based extremism detection.

## Features

- **Text Analysis**: Direct text input analysis
- **Audio/Video Processing**: Speech-to-text via Whisper, then analysis
- **Document Support**: PDF and DOCX extraction
- **Real-time Recording**: Live audio/video capture and analysis
- **Extremism Detection**: Pattern matching for violent rhetoric, hate speech, conspiracy theories

## Project Structure

```
transpara/
├── backend/           # flask api server
│   ├── app/          # application code
│   ├── models/       # ml model loading
│   ├── config/       # configuration
│   └── tests/        # backend tests
├── frontend/          # angular 19 spa
│   ├── src/          # source code
│   └── public/       # static assets
├── training/          # model training scripts
│   ├── model_main_tr.py       # political bias classifier
│   └── model_extremism_tr.py  # extremism detector
├── data/              # datasets (gitignored)
├── models/            # trained models (gitignored)
├── tests/             # test files and examples
├── utils/             # utility scripts
├── docker/            # docker configurations
├── nginx/             # nginx config
├── infrastructure/    # terraform/deployment
├── scripts/           # setup scripts
├── docs/              # documentation
└── assets/            # images and static files
```

## Tech Stack

**Backend**
- Python 3.11+
- Flask + Gunicorn
- scikit-learn (Random Forest)
- OpenAI Whisper (transcription)
- PostgreSQL + Redis

**Frontend**
- Angular 19
- Tailwind CSS
- TypeScript

**Infrastructure**
- Docker Compose
- Nginx reverse proxy

## Quick Start

```bash
# clone the repo
git clone https://github.com/yourusername/transpara.git
cd transpara

# start with docker
docker-compose up -d

# access the app
open http://localhost:4200
```

## Development

```bash
# backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on windows
pip install -r requirements.txt
flask run

# frontend
cd frontend
npm install
npm start
```

## Training Models

```bash
cd training

# train political bias classifier
python model_main_tr.py

# train extremism detector
python model_extremism_tr.py
```

Training data should be placed in `data/` as CSV files with `text` and `label` columns.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/text` | POST | Analyze text input |
| `/analyze/audio` | POST | Analyze audio file |
| `/analyze/video` | POST | Analyze video file |
| `/analyze/pdf` | POST | Analyze PDF document |
| `/analyze/docx` | POST | Analyze Word document |

## Response Format

```json
{
  "prediction": "left",
  "confidence": 0.85,
  "probabilities": {
    "left": 0.85,
    "center": 0.10,
    "right": 0.05
  },
  "is_extremist": false,
  "extremist_content": [],
  "transcription": "analyzed text here"
}
```

## Model Details

**Political Bias Classifier**
- Random Forest with 900 estimators
- TF-IDF + character n-grams + handcrafted features
- Three-class output: left, center, right

**Extremism Detector**
- Pattern matching for known extremist phrases
- Categories: violent rhetoric, hate speech, conspiracy theories, religious extremism, racial extremism
- Overrides political classification when detected

## License

MIT

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request
