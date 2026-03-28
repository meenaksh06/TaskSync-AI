# 🚀 Quick Start Guide

## Terminal Commands to Run the Project

### Prerequisites
- Python 3.9+ installed
- Node.js 18+ installed
- `credentials.json` in `Ai Agent/` folder (for Google Calendar integration)

---

## Option 1: Run Both Services (Recommended)

### Terminal 1 - Backend (FastAPI)
```bash
cd " AI:ML Project 4/Ai Agent"
python3 -m uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 - Frontend (React)
```bash
cd " AI:ML Project 4/Ai Agent/frontend"
npm run dev
```

---

## Option 2: One-Line Commands (Background)

### Start Backend in Background
```bash
cd " AI:ML Project 4/Ai Agent" && python3 -m uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000 &
```

### Start Frontend in Background
```bash
cd " AI:ML Project 4/Ai Agent/frontend" && npm run dev &
```

---

## First Time Setup

### 1. Install Python Dependencies
```bash
cd " AI:ML Project 4/Ai Agent"
pip3 install -r requirements.txt
```

### 2. Download spaCy Model
```bash
python3 -m spacy download en_core_web_sm
```

### 3. Install Frontend Dependencies
```bash
cd " AI:ML Project 4/Ai Agent/frontend"
npm install
```

### 4. Train Intent Classifier (if not already trained)
```bash
cd " AI:ML Project 4/Ai Agent"
python3 train_intent_classifier.py
```

---

## Access URLs

Once both services are running:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:3000 | Main chat interface |
| **Backend API** | http://localhost:8000 | FastAPI server |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |
| **Health Check** | http://localhost:8000/health | Server status |

---

## Stop Services

### Stop Backend
```bash
pkill -f "uvicorn app_enhanced"
```

### Stop Frontend
```bash
pkill -f "vite"
```

### Stop Both
```bash
pkill -f "uvicorn app_enhanced"; pkill -f "vite"
```

---

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Kill existing process: `kill -9 $(lsof -t -i:8000)`

### Frontend won't start
- Check if port 3000 is already in use: `lsof -i :3000`
- Kill existing process: `kill -9 $(lsof -t -i:3000)`

### Missing dependencies
- Backend: `pip3 install -r requirements.txt`
- Frontend: `cd frontend && npm install`

### Model not found
- Train the model: `python3 train_intent_classifier.py`
- Check if `models/intent_classifier_best.pt` exists

---

## Development Commands

### Backend
```bash
# Run with auto-reload (development)
python3 -m uvicorn app_enhanced:app --reload

# Run without reload (production)
python3 -m uvicorn app_enhanced:app --host 0.0.0.0 --port 8000

# Check API health
curl http://localhost:8000/health
```

### Frontend
```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## Quick Test

### Test Backend
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","text":"hello"}'
```

### Test Frontend
Open http://localhost:3000 in your browser

---

## Notes

- Backend must be running before frontend can connect
- Google Calendar integration requires `credentials.json` in `Ai Agent/` folder
- First run may take longer as models load into memory
- Voice features require microphone permissions in browser

