#!/bin/bash
echo "Starting ContentGuard..."
echo ""
echo "Starting backend on http://localhost:8000"
cd "$(dirname "$0")/backend" && uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo ""
echo "Starting frontend on http://localhost:5173"
cd "$(dirname "$0")/frontend" && npm run dev &
FRONTEND_PID=$!
echo ""
echo "Both servers running. Press Ctrl+C to stop."

trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait
