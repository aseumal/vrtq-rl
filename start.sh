#!/bin/bash
# VRTQ-RL: Start both backend and frontend
# Usage: ./start.sh

echo "Starting VRTQ-RL..."

# Start FastAPI backend
echo "[1/2] Starting FastAPI backend on port 8002..."
uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload &
BACKEND_PID=$!

# Start React frontend
echo "[2/2] Starting React dashboard on port 3001..."
cd dashboard/react-app && npm run dev &
FRONTEND_PID=$!

echo ""
echo "VRTQ-RL running:"
echo "  Backend:   http://localhost:8002"
echo "  Frontend:  http://localhost:3001"
echo "  API docs:  http://localhost:8002/docs"
echo ""
echo "Press Ctrl+C to stop both."

wait $BACKEND_PID $FRONTEND_PID
