#!/bin/bash

# Start backend and frontend in separate terminal windows.
# Usage: bash scripts/start.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting backend..."
mintty -t "Backend (FastAPI) — http://localhost:8000" --hold always -e bash -c "
cd '$PROJECT_DIR' &&
source .venv_BMAD/Scripts/activate &&
uvicorn backend.api:app --reload --port 8000
" &

# Wait for backend to be ready before opening frontend
echo "Waiting for backend to be ready..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is ready."
        break
    fi
    sleep 1
done

echo "Starting frontend..."
mintty -t "Frontend (Streamlit) — http://localhost:8501" --hold always -e bash -c "
cd '$PROJECT_DIR' &&
source .venv_BMAD/Scripts/activate &&
streamlit run frontend/app.py
" &

echo ""
echo "Both services started in separate windows."
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:8501"
