# Transpara - Political Bias & Extremism Detection System

## Overview

Transpara is a machine learning-powered content analysis platform designed to detect political bias and extremist rhetoric in text, audio, video, and document formats. The system combines a trained Random Forest classifier for political leaning detection with a comprehensive pattern-based extremism detector.

## Architecture

### Backend (Flask/Python)
- **Framework**: Flask with Gunicorn WSGI server
- **ML Model**: Random Forest classifier trained on political text data
- **Audio/Video Processing**: Whisper for speech-to-text transcription
- **Document Processing**: PDF and DOCX text extraction
- **Database**: PostgreSQL for user data, Redis for caching

### Frontend (Angular 19)
- **Framework**: Angular 19 with standalone components
- **Styling**: Tailwind CSS
- **State Management**: Angular Signals

### Infrastructure
- **Containerization**: Docker Compose
- **Reverse Proxy**: Nginx
- **File Size Limits**: 100MB for uploads

---

## Model Design & Fine-Tuning Rationale

### 1. Political Bias Classifier (BiasDetector)

#### Model Choice: Random Forest
We chose Random Forest over deep learning approaches for several reasons:
- **Interpretability**: Random Forest provides feature importance scores, allowing us to understand which words/phrases drive predictions
- **Training efficiency**: Works well with limited labeled political text data
- **Robustness**: Less prone to overfitting on small datasets compared to neural networks
- **Inference speed**: Fast predictions suitable for real-time analysis

#### Confidence Calibration
The raw model outputs often showed low confidence (50-60%) even for clearly partisan text. This created a poor user experience where obvious left or right-leaning content appeared uncertain. We implemented confidence calibration with the following approach:

```python
def apply_confidence_boost(probabilities, boost_factor=2.2, min_confidence=0.80):
    # Temperature scaling - raises confident predictions, suppresses uncertain ones
    scaled = np.power(probabilities, boost_factor)
    scaled = scaled / np.sum(scaled)  # Renormalize

    # Ensure minimum confidence threshold
    if scaled[max_idx] < min_confidence:
        # Redistribute probability from losing classes

    # Cap at 92% to avoid overconfidence
    if scaled[max_idx] > 0.92:
        scaled[max_idx] = 0.92
```

**Why these parameters?**
- `boost_factor=2.2`: Aggressive enough to separate confident predictions from uncertain ones, while preserving relative ordering
- `min_confidence=0.80`: Users expect decisive classifications; 80%+ feels authoritative
- `max_cap=0.92`: Prevents 99%+ confidence which would appear overconfident and reduce trust

#### Three-Class Classification (Left/Center/Right)
We deliberately chose a simplified three-class system rather than a political spectrum score because:
- Users intuitively understand left/center/right
- Reduces model complexity and improves accuracy
- Avoids false precision of numerical scales

---

### 2. Extremism Detector (ExtremismDetector)

#### Design Philosophy
The extremism detector uses pattern matching rather than ML classification. This was a deliberate choice:

**Why Pattern Matching over ML?**
1. **Explainability**: When flagging content as extremist, users need to know exactly WHY. Pattern matching shows the exact phrase that triggered detection.
2. **No false negatives on known threats**: ML models might miss known extremist phrases due to context. Pattern matching guarantees detection of specific terms.
3. **Immediate updates**: New extremist terminology can be added instantly without retraining
4. **Transparency**: The pattern list serves as documentation of what the system considers extremist

#### Pattern Categories & Rationale

We organized patterns into specific categories to provide detailed, actionable feedback:

| Category | Examples | Rationale |
|----------|----------|-----------|
| **violent anti-american rhetoric** | "death to america", "destroy america" | Direct calls for violence against the US |
| **critique of US economy** | "us economic system", "american capitalism" | Economic criticism - flagged but distinct from violence |
| **critique of US culture** | "american culture", "consumerism" | Cultural criticism |
| **radicalist ideology** | "white supremacy", "ethnostate", "race realism" | Racial/ethnic extremism |
| **antisemitism** | "jewish conspiracy", "zog", "protocols of zion" | Anti-Jewish hatred |
| **religious extremism** | "sharia law", "caliphate", "deus vult" | Religious-based extremism |
| **violence advocacy** | "kill all", "exterminate", "lynch" | Direct calls for violence |
| **sedition** | "overthrow the government", "armed revolution" | Anti-government extremism |
| **conspiracy extremism** | "false flag", "crisis actor", "chemtrails" | Conspiracy theories that radicalize |
| **hate speech** | Slurs and dehumanizing language | Discriminatory language |

#### Why So Many Patterns (~500+)?

Extremism manifests in many forms. A minimalist approach would miss:
- **Dog whistles**: Terms like "1488", "14 words", "based and redpilled" that seem innocuous but signal extremism
- **Evolving language**: Online communities constantly create new coded terms
- **Subtle variants**: "amerikkka" vs "america", coded spellings

#### Category Granularity

Early versions used broad categories like "anti-american" for everything. We refined this because:
- Users asked "why is this flagged?" - generic labels don't answer that
- Distinguishing "violent rhetoric" from "economic critique" provides context
- Researchers need detailed categorization for analysis
- Reduces perception of political bias in the tool itself

**Example of category evolution:**
```
Before: ('american capitalism', 'anti-american')
After:  ('american capitalism', 'critique of US capitalism')

Before: ('death to america', 'anti-american')
After:  ('death to america', 'violent anti-american rhetoric')
```

---

## Override Logic

When extremist content is detected, it **overrides** the political bias classification. This design choice reflects that:

1. Extremism transcends the left-right spectrum
2. Violent content shouldn't be normalized as merely "political"
3. Users need clear warning that content is problematic regardless of political lean

The UI displays a prominent red warning banner with:
- The overall classification changed to "ANTI-AMERICAN" or the relevant category
- Each flagged phrase with its specific category
- The exact pattern that matched (for transparency)

---

## Confidence Display

The system shows confidence percentages for the three-class prediction:
- Left: X%
- Center: Y%
- Right: Z%

This probability distribution helps users understand:
- How decisive the classification is
- Whether content has mixed signals
- The relative strength of each classification

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/text` | POST | Analyze text input |
| `/analyze/audio` | POST | Analyze uploaded audio file |
| `/analyze/video` | POST | Analyze uploaded video file |
| `/analyze/live-audio` | POST | Analyze recorded audio |
| `/analyze/live-video` | POST | Analyze recorded video |
| `/analyze/pdf` | POST | Analyze PDF document |
| `/analyze/docx` | POST | Analyze Word document |

---

## Response Format

```json
{
  "prediction": "left|center|right",
  "confidence": 0.85,
  "probabilities": {
    "left": 0.85,
    "center": 0.10,
    "right": 0.05
  },
  "interpretation": "Human-readable analysis",
  "transcription": "Original or transcribed text",
  "is_extremist": true,
  "extremist_content": [
    {
      "text": "The flagged sentence",
      "category": "critique of US economy",
      "pattern_matched": "us economic system",
      "start_pos": 0,
      "end_pos": 45
    }
  ],
  "sentence_analysis": [
    {
      "text": "Individual sentence",
      "prediction": "left",
      "confidence": 0.82
    }
  ]
}
```

---

## Future Improvements

1. **Contextual analysis**: Some patterns are extremist only in certain contexts
2. **Multi-language support**: Currently English-only
3. **Severity scoring**: Not all extremism is equally severe
4. **Source attribution**: Linking patterns to known extremist movements
5. **Temporal analysis**: Tracking how language evolves over time

---

## Running the Application

```bash
# Start all services
docker-compose up -d

# Frontend development
cd frontend && npm start

# Access the application
http://localhost:4200
```

---

## Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bias model | Random Forest | Interpretability, speed, works with limited data |
| Extremism detection | Pattern matching | Explainability, no false negatives, instant updates |
| Confidence calibration | Temperature scaling + floor | User trust, decisive predictions |
| Categories | 30+ specific categories | Detailed feedback, transparency |
| Override behavior | Extremism overrides bias | Safety prioritization |
| Three-class system | Left/Center/Right | User intuition, reduced complexity |

---

## License

Proprietary - Transpara Educational Tools

## Contact

For questions about the model or methodology, contact the development team.
