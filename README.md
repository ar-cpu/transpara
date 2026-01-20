# Transpara

A machine learning system that analyzes political content to detect bias and extremism. Transpara reads text (or transcribes audio/video) and classifies whether the content leans left, center, or right politically, and whether it contains extremist rhetoric.

## Overview

Transpara uses two separate machine learning models working together:

**The Political Bias Detector** analyzes text and classifies it as left-leaning, centrist, or right-leaning. It's trained on a comprehensive dataset of political statements and uses a Random Forest classifier with 900 decision trees. It analyzes linguistic patterns, sentiment, writing style, argumentation structure, and readability metrics to make its prediction.

**The Extremism Detector** scans for anti-American or extremist content. This model specifically looks for violent rhetoric, hate speech, conspiracy theories, religious extremism, racial supremacy language, and anti-government sedition. It uses pattern matching combined with machine learning to identify dangerous content that goes beyond normal political discourse.

Both models work on any text input, direct entry, extracted from PDFs or Word documents, or transcribed from audio/video using OpenAI's Whisper.

## How the Models Work

### Political Bias Model (model_main_tr.py)

This is a sophisticated text classifier that uses multiple layers of feature extraction:

- **Lexical Analysis**: Identifies words and phrases associated with different political ideologies
  - Left indicators: "social justice", "climate change", "universal healthcare", "workers rights"
  - Right indicators: "free market", "border security", "traditional values", "fiscal responsibility"
  - Center indicators: "bipartisan", "data shows", "evidence suggests", "cost benefit"

- **Semantic Features**: Analyzes how ideas are framed
  - Economic framing (inequality vs opportunity)
  - Social framing (systemic issues vs personal responsibility)
  - Government framing (public good vs overreach)

- **Stylometric Analysis**: Examines writing style
  - Sentence complexity and length
  - Vocabulary richness
  - Punctuation patterns
  - Function word usage

- **Discourse Analysis**: Studies argument structure
  - Causal markers ("because", "therefore")
  - Contrasts ("but", "however")
  - Evidence presentation
  - Claim-making patterns

The model combines all these features using TF-IDF vectorization with character n-grams, then feeds them into a Random Forest classifier. It's trained on thousands of political statements labeled as left/center/right and achieves high accuracy through cross-validation.

### Extremism Model (model_extremism_tr.py)

This model is specifically designed to catch anti-American and extremist content with comprehensive detection capabilities:

- **Violence Vocabulary**: Tracks 200+ words related to killing, destruction, weapons, and bloodshed
- **Fear and Threat Language**: Monitors panic words, invasion rhetoric, and doomsday scenarios
- **Dehumanization**: Detects language that treats groups as subhuman, vermin, or diseases
- **Us-vs-Them Tribalism**: Identifies divisive othering language
- **Certainty and Absolutism**: Spots dogmatic, black-and-white thinking
- **Call to Action**: Recognizes mobilization and violent coordination
- **Extremist Ideologies**:
  - Religious extremism (jihad, caliphate, holy war, crusade)
  - Racial extremism (Aryan supremacy, eugenics, purity, genocide)
  - Conspiracy theories (deep state, globalist cabal, false flags)
  - Anti-government sedition (tyranny, overthrow, revolution, armed uprising)

The model analyzes its  density, co-occurrence patterns, and context. It uses:
- Word embeddings (100-dimensional vectors)
- N-gram analysis (unigrams, bigrams, trigrams, character n-grams)
- Semantic density scoring
- Syntactic pattern recognition
- Sentiment analysis
- Readability metrics
- Context window analysis (examines sentences in groups)

All features are scaled, selected, and fed into a Random Forest with 1,000 trees. The model classifies content as "american" or "anti_american".

## Tech Stack

**Backend**:
- Python 3.11+ with Flask
- scikit-learn for machine learning (Random Forests, TF-IDF, feature selection)
- OpenAI Whisper for speech-to-text transcription
- joblib for model serialization
- NumPy and Pandas for data processing
- PostgreSQL for data storage, Redis for caching

**Frontend**:
- Angular 19 (latest)
- TypeScript for type safety
- Tailwind CSS for styling

**Infrastructure**:
- Docker Compose for orchestration
- Nginx as reverse proxy
- Gunicorn for serving Python

## Project Structure

```
transpara/
├── training/                    # Machine learning training scripts
│   ├── model_main_tr.py        # Trains the political bias classifier (left/center/right)
│   └── model_extremism_tr.py   # Trains the extremism detector (american/anti-american)
├── models/                      # Trained models are saved here
│   ├── bias_detector_model.pkl
│   └── extremism_detector.pkl
├── data/                        # Training datasets (CSV files with text + label columns)
├── backend/                     # Flask API server
│   ├── app/                    # Application logic
│   ├── models/                 # Model loading and inference
│   └── config/                 # Configuration files
├── frontend/                    # Angular web interface
│   ├── src/                    # Source code
│   └── public/                 # Static assets
├── tests/                       # Test files and example inputs
├── utils/                       # Utility scripts
├── docker/                      # Docker configurations
├── nginx/                       # Nginx reverse proxy config
└── infrastructure/              # Terraform/deployment scripts
```

## Getting Started

The fastest way to run Transpara is with Docker:

```bash
git clone https://github.com/ar-cpu/transpara.git
cd transpara
docker-compose up -d
```

Then open your browser to `http://localhost:4200`

## Development Setup

For local development without Docker:

**Backend**:
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # on Windows
source venv/bin/activate       # on Linux/Mac
pip install -r requirements.txt
flask run
```

**Frontend**:
```bash
cd frontend
npm install
npm start
```

The frontend will be at `http://localhost:4200` and communicates with the backend at `http://localhost:5000`

## Training the Models

Both models require training data in CSV format with two columns: `text` and `label`. Place your CSV files in the `data/` directory.

**Train the Political Bias Model**:
```bash
cd training
python model_main_tr.py
```

This creates `models/bias_detector_model.pkl`. The training process:
1. Loads all CSV files from `../data`
2. Extracts hundreds of linguistic features
3. Trains a Random Forest with 900 trees
4. Evaluates with cross-validation
5. Saves the model

**Train the Extremism Detector**:
```bash
cd training
python model_extremism_tr.py
```

This creates `models/extremism_detector.pkl`. The training is similar but even more comprehensive with 1,000 trees and additional feature extractors.

Training can take several minutes depending on your dataset size. You'll see progress output showing feature extraction steps, cross-validation scores, and performance metrics.

## Using the API

Once the backend is running, you can analyze content through these endpoints:

**Analyze Text**:
```bash
POST /analyze/text
Content-Type: application/json

{
  "text": "your text here"
}
```

**Analyze Audio File**:
```bash
POST /analyze/audio
Content-Type: multipart/form-data

file: <audio file>
```

**Analyze Video File**:
```bash
POST /analyze/video
Content-Type: multipart/form-data

file: <video file>
```

**Analyze PDF Document**:
```bash
POST /analyze/pdf
Content-Type: multipart/form-data

file: <pdf file>
```

**Analyze Word Document**:
```bash
POST /analyze/docx
Content-Type: multipart/form-data

file: <docx file>
```

### Response Format

All endpoints return the same structure:

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

The `prediction` field shows the political bias (left/center/right). The `confidence` represents the model's certainty. The `probabilities` object breaks down the likelihood for each class.

If extremist content is detected, `is_extremist` becomes true and `extremist_content` lists the specific patterns found. The `transcription` field is only populated for audio/video files.

## Model Performance

The political bias classifier typically achieves:
- Accuracy: 85-90% on test data
- F1-Score: 0.84-0.88 weighted average

The extremism detector is more conservative (fewer false positives):
- Accuracy: 90-95% on test data
- F1-Score: 0.88-0.94 weighted average

Both models use stratified k-fold cross-validation during training to ensure robust performance across different types of content.

## Media Type Support

**Text**: Analyzed directly

**Audio/Video**: First transcribed using OpenAI's Whisper model (automatic speech recognition), then the transcription is analyzed

**PDF/DOCX**: Text is extracted using specialized parsers, then analyzed

The backend handles all file processing and feeds clean text to the models.

## Contributing

To improve the models:

1. Add more training data to the `data/` directory
2. Retrain the models with `python model_main_tr.py` or `python model_extremism_tr.py`
3. Test the new models with sample inputs
4. Submit a pull request with your improvements

The models will automatically improve with more diverse training data.

## License

MIT - Use this however you want
