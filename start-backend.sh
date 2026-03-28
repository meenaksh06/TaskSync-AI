#!/bin/bash
# Backend Server Startup Script
cd "Ai Agent"
python3 -m uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000

