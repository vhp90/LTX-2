 #!/bin/bash
# ==============================================================================
#  LTX-2 Studio — Startup Script
#  Runs environment setup (idempotent), then launches backend + frontend.
# ==============================================================================

echo "========================================"
echo " Starting LTX-2 Studio"
echo "========================================"

# ------------------------------------------------------------------
#  Cleanup any dangling processes on our ports
# ------------------------------------------------------------------
cleanup_port() {
    PORT=$1
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "Cleaning up process on port $PORT (PID: $PID)..."
        kill -9 $PID 2>/dev/null
    fi
}

cleanup_port 8080
cleanup_port 5173

# ------------------------------------------------------------------
#  Environment setup (installs deps + downloads models if needed)
# ------------------------------------------------------------------
echo ""
echo "Running environment setup..."
bash setup_environment.sh
echo ""

# ------------------------------------------------------------------
#  Launch servers
# ------------------------------------------------------------------
echo "Starting FastAPI Backend (Port 8080)..."
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8080 --reload > backend.log 2>&1 &
BACKEND_PID=$!

echo "Starting Vite Frontend (Port 5173)..."
npm --prefix app/frontend run dev -- --host --port 5173 > frontend.log 2>&1 &
FRONTEND_PID=$!

trap "echo -e '\nStopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

echo "========================================"
echo " Servers are running!"
echo " -> API Backend: http://0.0.0.0:8080"
echo " -> UI Frontend: http://localhost:5173"
echo " (Press Ctrl+C to stop both servers)"
echo "========================================"

wait $BACKEND_PID $FRONTEND_PID
