#!/bin/bash
# LTX-2 Studio — Startup Script
# Kills any existing processes on ports 8000 and 5173, then boots the backend and frontend.

echo "========================================"
echo " Starting LTX-2 Studio"
echo "========================================"

# Function to cleanly kill processes attached to a port
cleanup_port() {
    PORT=$1
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "Cleaning up dangling process on port $PORT (PID: $PID)..."
        kill -9 $PID 2>/dev/null
    fi
}

# Cleanup known ports
cleanup_port 8080
cleanup_port 5173

echo "Starting FastAPI Backend (Port 8080)..."
# Start backend in background
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8080 --reload > backend.log 2>&1 &
BACKEND_PID=$!

echo "Starting Vite Frontend (Port 5173)..."
# Start frontend in background
cd app/frontend && npm run dev -- --host --port 5173 > ../../frontend.log 2>&1 &
FRONTEND_PID=$!

# Trap SIGINT (Ctrl+C) to gracefully stop both servers when exited
trap "echo -e '\nStopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

echo "========================================"
echo " Servers are running!"
echo " -> API Backend: http://0.0.0.0:8080"
echo " -> UI Frontend: http://localhost:5173"
echo " (Press Ctrl+C to stop both servers)"
echo "========================================"

# Wait for background processes so the script stays active
wait $BACKEND_PID $FRONTEND_PID
