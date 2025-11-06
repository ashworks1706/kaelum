#!/bin/bash

# Kaelum Demo Startup Script
echo "🚀 Starting Kaelum Demo..."
echo ""

# Check if .kaelum directory exists
if [ ! -d ".kaelum" ]; then
    echo "📁 Creating .kaelum directory structure..."
    mkdir -p .kaelum/{routing,cache,cache_validation,calibration}
fi

# Start backend in background
echo "🔧 Starting Flask API backend on port 5000..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo "✅ Backend API is running (PID: $BACKEND_PID)"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo "🎨 Starting Next.js frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✨ Kaelum Demo is running!"
echo ""
echo "📍 Backend API:  http://localhost:5000"
echo "📍 Frontend UI:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Trap Ctrl+C and cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # Kill any remaining python/node processes from this script
    pkill -P $$ 2>/dev/null
    echo "✅ Stopped successfully"
    exit 0
}

trap cleanup INT TERM

# Wait for user interrupt
wait
