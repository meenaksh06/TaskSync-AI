#!/bin/bash

# Start Backend and Frontend Servers
# Usage: ./start.sh

echo "🚀 Starting AI Assistant..."

# Start Backend
echo "📦 Starting Backend Server..."
cd "Ai Agent"
python3 -m uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start Frontend
echo "🎨 Starting Frontend Server..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Servers started!"
echo "📡 Backend: http://localhost:8000 (PID: $BACKEND_PID)"
echo "🌐 Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

