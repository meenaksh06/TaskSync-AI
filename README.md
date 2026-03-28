# TaskFlow AI

An intelligent, multi-modal AI assistant built with state-of-the-art NLP models for intent classification, entity extraction, and speech-to-text capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)
![spaCy](https://img.shields.io/badge/spaCy-3.7-purple.svg)

---

## Project Overview

This project implements a comprehensive AI assistant that can:

- **Understand Natural Language** - Fine-tuned DistilBERT intent classifier
- **Extract Entities** - spaCy NER + custom patterns for names, dates, emails
- **Process Voice Commands** - OpenAI Whisper speech-to-text
- **Manage Tasks** - Schedule meetings, set reminders, send messages
- **Remember Context** - Multi-user session management

### Key Improvements Over Baseline

| Component             | Baseline           | Enhanced                 |
| --------------------- | ------------------ | ------------------------ |
| Intent Classification | Zero-shot (75-85%) | Fine-tuned (~95%)        |
| Entity Extraction     | Regex only         | spaCy NER + Patterns     |
| Speed                 | ~500ms/query       | ~50ms/query              |
| Voice Support         | Not integrated     | Full Whisper integration |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│                    (Text or Voice/Audio)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPEECH-TO-TEXT (if audio)                     │
│                    OpenAI Whisper Base                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INTENT CLASSIFICATION                          │
│              Fine-tuned DistilBERT Classifier                    │
│                    (8 Intent Classes)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENTITY EXTRACTION                             │
│              spaCy NER + Custom Pattern Matching                 │
│         (Names, Emails, Phones, Dates, Organizations)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ACTION EXECUTION                              │
│     Schedule Meeting | Set Reminder | Send Email | etc.          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE GENERATION                           │
│              + Session Memory Update                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AI:ML Project 4/
├── Ai Agent/
│   ├── app_enhanced.py          # Main FastAPI application
│   ├── app.py                   # Original baseline app
│   ├── train_intent_classifier.py  # Model training script
│   ├── entity_extractor.py      # Advanced entity extraction
│   ├── evaluate_models.py       # Model comparison & evaluation
│   ├── setup_and_run.py         # Automated setup script
│   ├── requirements.txt         # Python dependencies
│   ├── data/
│   │   └── intent_dataset.json  # Training dataset (320 examples)
│   ├── models/
│   │   └── intent_classifier/   # Trained model files
│   ├── reports/
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   └── model_comparison_report.txt
│   └── static/
│       ├── index.html           # Web UI
│       ├── styles.css           # Styling
│       └── app.js               # Frontend logic
├── speech_to_text/              # STT module
├── AIML Project Proposal.pdf
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd "Ai Agent"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the Model (Optional - pre-trained available)

```bash
python train_intent_classifier.py
```

This will:

- Train a fine-tuned DistilBERT classifier
- Generate evaluation metrics
- Save model to `models/intent_classifier/`

### 3. Run the Application

```bash
uvicorn app_enhanced:app --reload --port 8000
```

### 4. Access the Assistant

- **Web UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## API Endpoints

### POST `/infer`

Process text input and return assistant response.

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "text": "Schedule a meeting with John at 3pm tomorrow"}'
```

**Response:**

```json
{
  "intent": "schedule_meeting",
  "confidence": 0.97,
  "model_type": "fine_tuned",
  "entities": {
    "names": ["John"],
    "datetime": "2024-12-08T15:00:00"
  },
  "response": "  Meeting scheduled with John at 2024-12-08T15:00:00."
}
```

### POST `/voice`

Process audio input via speech-to-text.

```bash
curl -X POST "http://localhost:8000/voice?user_id=user123" \
  -F "audio=@recording.wav"
```

### GET `/context/{user_id}`

Retrieve user session data.

### GET `/reset/{user_id}`

Clear user session.

### GET `/history/{user_id}`

Get conversation history.

---

## Supported Intents

| Intent             | Example Commands                            |
| ------------------ | ------------------------------------------- |
| `schedule_meeting` | "Schedule a meeting with John at 3pm"       |
| `send_email`       | "Email sarah@company.com about the project" |
| `check_calendar`   | "What meetings do I have today?"            |
| `cancel_meeting`   | "Cancel my 2pm appointment"                 |
| `add_reminder`     | "Remind me to call mom at 5pm"              |
| `check_reminders`  | "What are my reminders?"                    |
| `send_message`     | "Tell Mike I'll be late"                    |
| `general_query`    | "Hello", "What can you do?"                 |

---

## Model Performance

### Intent Classification Results

After training on 320 examples across 8 intents:

| Metric         | Fine-tuned | Zero-shot Baseline |
| -------------- | ---------- | ------------------ |
| Accuracy       | ~95%       | ~78%               |
| F1 Score       | ~0.94      | ~0.75              |
| Inference Time | ~50ms      | ~500ms             |

### Entity Extraction

The enhanced entity extractor handles:

- **Names**: "John", "Dr. Smith", "john" (case-insensitive)
- **Emails**: Various formats
- **Times**: "3pm", "15:00", "noon", "5ish", "in 2 hours"
- **Dates**: "tomorrow", "next Monday", "Dec 15"
- **Organizations**: Company names via NER
- **Phone Numbers**: Multiple formats

---

## Technical Details

### Models Used

1. **Intent Classification**: `distilbert-base-uncased` fine-tuned
2. **Entity Extraction**: `en_core_web_sm` (spaCy) + custom patterns
3. **Speech-to-Text**: `openai/whisper-base`
4. **Fallback**: `typeform/distilbert-base-uncased-mnli` (zero-shot)

### Training Configuration

```python
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 64,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1
}
```

---

## Web Interface Features

- **Modern Dark Theme** with light mode toggle
- **Voice Recording** with browser microphone
- **Real-time Chat** interface
- **Intent Analysis** panel showing confidence scores
- **Entity Highlighting** for extracted information
- **Session Management** (view context, clear history)
- **Quick Action Buttons** for common tasks

---

## Future Improvements

- [ ] Add more training data for edge cases
- [ ] Implement conversation context for follow-ups
- [ ] Add calendar integration (Google Calendar, Outlook)
- [ ] Implement actual email sending (SMTP)
- [ ] Add notification system for reminders
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support

---

## Authors

- Meenaksh Singhania
- Eranki Sai Vikas
- Prerak Arya

---

## License

This project is for educational purposes as part of AI/ML coursework.

---

## Acknowledgments

- Hugging Face Transformers
- spaCy NLP
- OpenAI Whisper
- FastAPI Framework
