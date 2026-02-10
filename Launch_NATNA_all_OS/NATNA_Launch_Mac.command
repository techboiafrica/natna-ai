#!/bin/bash
echo "[FIRE] LAUNCHING BULLETPROOF NATNA AI DESKTOP [FIRE]"
echo "============================================="
echo "[FOLDER] Sacred files directory: $(pwd)"
echo "[DB] Database: massive_tigrinya_database.db (265,972+ entries)"
echo "[BRAIN] AI Engine: intelligent_translator.py (DeepSeek + SQL fallback)"
echo "[GLOBE] Interface: NatnaUI.py with 6-Layer Defense System"
echo ""

# Resolve script directory (works even with symlinks/spaces)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATNA_ROOT="$SCRIPT_DIR/.."

# Detect architecture and pick bundled Python
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    PYTHON_BIN="$NATNA_ROOT/SACRED/runtime/python/mac-arm64/python/bin/python3.11"
else
    PYTHON_BIN="$NATNA_ROOT/SACRED/runtime/python/mac-x64/python/bin/python3"
fi

# Fall back to system python3 if bundled not found
if [ ! -x "$PYTHON_BIN" ]; then
    echo "  [WARN] Bundled Python not found at $PYTHON_BIN, falling back to system python3..."
    PYTHON_BIN="$(command -v python3 || echo python3)"
fi

# Set PYTHONPATH for bundled packages
export PYTHONPATH="$NATNA_ROOT/SACRED/runtime/python/packages:$NATNA_ROOT/SACRED/APP_Production/dependencies${PYTHONPATH:+:$PYTHONPATH}"

echo "  [PYTHON] Using: $PYTHON_BIN"

# STEP 1: Start Ollama Server with NATNA Drive Models
echo "[AI] Step 1: Starting Ollama server with NATNA drive models..."
export OLLAMA_MODELS="$NATNA_ROOT/SACRED/runtime/models"
export OLLAMA_HOST="127.0.0.1:11434"

# [LAUNCH] MAXIMUM PERFORMANCE CONFIGURATION
echo "  [FIRE] Configuring Ollama for MAXIMUM resource utilization..."

# [FIRE] FORCE MAXIMUM PERFORMANCE - HARD-CODED VALUES FOR YOUR M1 PRO
echo "  [MEM] FORCING: 16GB RAM, 8 CPU cores (M1 Pro Maximum Performance Mode)"
echo "  [LAUNCH] Configuring for ABSOLUTE MAXIMUM resource utilization..."

export OLLAMA_MAX_LOADED_MODELS=1        # Keep ONLY 1 model in memory at a time
export OLLAMA_NUM_PARALLEL=8             # Use ALL 8 CPU cores
export OLLAMA_FLASH_ATTENTION=true       # Enable fast attention mechanism
export OLLAMA_CONTEXT_LENGTH=8192        # Longer conversations
export OLLAMA_MAX_QUEUE=1024            # Handle more concurrent requests
export OLLAMA_KEEP_ALIVE=10m            # Keep models loaded longer

echo "  [OK] MAXIMUM PERFORMANCE Ollama configuration applied"
echo "  [FIRE] Parallel processing: 8 cores, Models in memory: 4, Context: 8192, Queue: 1024"

# Start portable Ollama from NATNA drive
OLLAMA_PATH="$NATNA_ROOT/SACRED/runtime/ollama/ollama"
if [ -f "$OLLAMA_PATH" ]; then
    echo "  [LAUNCH] Starting portable Ollama server..."
    "$OLLAMA_PATH" serve &
    OLLAMA_PID=$!
    echo "  [NOTE] Ollama PID: $OLLAMA_PID"
else
    echo "  [WARN] Portable Ollama not found, using system Ollama..."
    ollama serve &
    OLLAMA_PID=$!
fi

# Wait for Ollama to start
echo "  [WAIT] Waiting for Ollama server to initialize..."
sleep 5

# Test Ollama connection and preload Qwen model
echo "  [TEST] Testing Ollama connection and preloading Qwen 3 0.6B..."
for i in {1..10}; do
    if curl -s http://localhost:11434/api/version >/dev/null; then
        echo "  [OK] Ollama server responding!"

        # Preload the default Qwen model
        echo "  [DOWNLOAD] Preloading Qwen 3 0.6B model for instant responses..."
        curl -s -X POST http://localhost:11434/api/generate \
            -H "Content-Type: application/json" \
            -d '{"model": "qwen3:0.6b", "prompt": "Hello", "stream": false}' >/dev/null &
        break
    else
        echo "  [WAIT] Ollama starting... (attempt $i/10)"
        sleep 2
    fi

    if [ $i -eq 10 ]; then
        echo "  [ERROR] Ollama failed to start after 10 attempts"
        exit 1
    fi
done

# STEP 2: Start NATNA Web Interface with Bulletproof System
echo ""
echo "[SHIELD] Step 2: Starting NATNA web interface with 6-Layer Defense System..."

# Find NatnaUI.py in multiple possible locations
SCRIPT_LOCATIONS=(
    "$NATNA_ROOT/SACRED/APP_Production/NatnaUI.py"
    "$SCRIPT_DIR/../SACRED/APP_Production/NatnaUI.py"
    "$SCRIPT_DIR/NATNA AI/SACRED/APP_Production/NatnaUI.py"
    "$SCRIPT_DIR/APP_Production/NatnaUI.py"
    "$SCRIPT_DIR/NatnaUI.py"
    "./NatnaUI.py"
)

SCRIPT_PATH=""
for location in "${SCRIPT_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        SCRIPT_PATH="$location"
        echo "  📍 Found NatnaUI.py at: $SCRIPT_PATH"
        break
    fi
done

if [ -z "$SCRIPT_PATH" ]; then
    echo "  [ERROR] ERROR: NatnaUI.py not found in any expected location!"
    echo "  [LIST] Searched locations:"
    for location in "${SCRIPT_LOCATIONS[@]}"; do
        echo "    - $location"
    done
    exit 1
fi

cd "$(dirname "$SCRIPT_PATH")"

# Start the bulletproof server with error logging
echo "  [LAUNCH] Starting web server with error logging..."
LOG_FILE="/tmp/natna_startup.log"
"$PYTHON_BIN" "$(basename "$SCRIPT_PATH")" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "  [NOTE] Web server PID: $SERVER_PID"
echo "  [DOC] Error log: $LOG_FILE"

# STEP 3: Wait for Web Server and Auto-Open Browser
echo ""
echo "[GLOBE] Step 3: Waiting for web server and opening browser..."

# Wait for web server with better detection (longer timeout for model loading)
WEB_READY=false
for i in {1..30}; do
    if curl -s http://localhost:8080 >/dev/null 2>&1; then
        echo "  [OK] Web server responding!"
        WEB_READY=true
        break
    else
        echo "  [WAIT] Web server starting... (attempt $i/30)"
        sleep 1
    fi
done

if [ "$WEB_READY" = false ]; then
    echo "  [ERROR] Web server failed to start after 30 attempts"
    echo ""
    echo "[SEARCH] DIAGNOSTIC INFORMATION:"
    echo "  [LIST] Web server process status:"
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "    [OK] Process $SERVER_PID is still running (hung during startup)"
    else
        echo "    [ERROR] Process $SERVER_PID has crashed/exited"
    fi
    echo ""
    echo "  [DOC] Error log contents:"
    if [ -f "$LOG_FILE" ]; then
        echo "    --- START ERROR LOG ---"
        tail -20 "$LOG_FILE" | sed 's/^/    /'
        echo "    --- END ERROR LOG ---"
        echo ""
        echo "  📍 Full log file: $LOG_FILE"
    else
        echo "    [ERROR] No log file found at $LOG_FILE"
    fi
    echo ""
    echo "  [PLUG] Port status:"
    echo "    Port 8080: $(lsof -ti :8080 | wc -l | xargs) processes"
    echo ""
    echo "  [MEM] Database status:"
    DB_PATH="../educational_archive/knowledge/massive_wikipedia.db"
    if [ -f "$DB_PATH" ]; then
        echo "    [OK] Database file exists: $DB_PATH"
        echo "    [STATS] Database size: $(ls -lh "$DB_PATH" | awk '{print $5}')"
    else
        echo "    [ERROR] Database not found at: $DB_PATH"
    fi
    echo ""
    echo "[STOP] Cleaning up and exiting..."
    kill $OLLAMA_PID $SERVER_PID 2>/dev/null
    exit 1
fi

# STEP 4: Force Browser Opening with Multiple Attempts
echo ""
echo "[LAUNCH] Step 4: Opening browser to NATNA AI interface..."

# Try multiple browser opening methods
open "http://localhost:8080" 2>/dev/null || \
    /usr/bin/open "http://localhost:8080" 2>/dev/null || \
    osascript -e 'tell application "Safari" to make new document with properties {URL:"http://localhost:8080"}' 2>/dev/null || \
    osascript -e 'tell application "Google Chrome" to make new tab with properties {URL:"http://localhost:8080"}' 2>/dev/null || \
    echo "  [WARN] Auto-browser opening failed - manually go to http://localhost:8080"

# Give browser time to open
sleep 2

# Detect LAN IP for network access (phone/tablet)
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")

echo ""
echo "[SUCCESS] =========================================="
echo "[SUCCESS] NATNA AI DESKTOP SUCCESSFULLY LAUNCHED!"
echo "[SUCCESS] =========================================="
echo ""
echo "[OK] Ollama AI Server: Running (PID $OLLAMA_PID)"
echo "[OK] NATNA Web Interface: Running (PID $SERVER_PID)"
echo "[OK] Models Available: qwen3:0.6b, phi4-mini, deepseek-r1, etc."
echo "[OK] Browser: http://localhost:8080"
if [ -n "$LAN_IP" ]; then
    echo ""
    echo "[MOBILE] =========================================="
    echo "[MOBILE] ACCESS FROM PHONE/TABLET (same Wi-Fi):"
    echo "[MOBILE]   http://$LAN_IP:8080"
    echo "[MOBILE] =========================================="
    echo ""
    echo "[TIP] If it doesn't connect, check System Settings > Network > Firewall"
    echo "   and allow incoming connections for Python."
fi
echo ""
echo "[TEST] Ready to test: Ask 'What is photosynthesis?' to test AI"
echo ""
echo "[STOP] Press Ctrl+C to stop both servers"
echo "═══════════════════════════════════════════"

# Enhanced cleanup trap for both processes
cleanup() {
    echo ""
    echo "[STOP] Shutting down NATNA AI Desktop..."
    echo "  [SHIELD] Using bulletproof cleanup system..."

    # Kill web server first (it has bulletproof cleanup)
    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "  [SIGNAL] Stopping web server (PID $SERVER_PID)..."
        kill -TERM $SERVER_PID 2>/dev/null
        sleep 3
    fi

    # Kill Ollama server
    if [ ! -z "$OLLAMA_PID" ] && kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "  [AI] Stopping Ollama server (PID $OLLAMA_PID)..."
        kill -TERM $OLLAMA_PID 2>/dev/null
        sleep 2

        # Force kill if still running
        if kill -0 $OLLAMA_PID 2>/dev/null; then
            kill -9 $OLLAMA_PID 2>/dev/null
        fi
    fi

    echo "  [OK] All processes stopped cleanly"
    echo ""
    echo "[TARGET] NATNA AI Desktop shutdown complete!"
    exit 0
}

trap cleanup INT TERM

# Keep script running and monitor both processes
while true; do
    # Check if web server is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ERROR] Web server stopped unexpectedly"
        cleanup
    fi

    # Check if Ollama is still running
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "[ERROR] Ollama server stopped unexpectedly"
        cleanup
    fi

    sleep 5
done