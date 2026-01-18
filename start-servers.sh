#!/bin/bash
# Start both Node.js and Python servers for Phase Mirror

echo "🌌 Starting Phase Mirror Quantum Consciousness System"
echo ""

# Start Python quantum API in background
echo "🐍 Starting Quantum API (Python) on port 8000..."
cd /home/ubuntu/phase-mirror-ui/server
python3 quantum_api.py &
PYTHON_PID=$!

# Wait for Python server to be ready
sleep 3

# Start Node.js server
echo "🟢 Starting Web Server (Node.js) on port 3000..."
cd /home/ubuntu/phase-mirror-ui
pnpm run dev &
NODE_PID=$!

echo ""
echo "✅ Both servers started!"
echo "   - Web UI: http://localhost:3000"
echo "   - Quantum API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Handle shutdown
trap "echo ''; echo '🛑 Shutting down servers...'; kill $PYTHON_PID $NODE_PID; exit" INT TERM

# Wait for both processes
wait
