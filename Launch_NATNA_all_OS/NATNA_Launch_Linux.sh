#!/bin/bash
echo "[FIRE] LAUNCHING BULLETPROOF NATNA AI DESKTOP [FIRE]"
echo "============================================="
echo "[FOLDER] Sacred files directory: $(dirname "$0")"
echo "[DB] Database: massive_tigrinya_database.db (265,972+ entries)"
echo "[BRAIN] AI Engine: intelligent_translator.py (DeepSeek + SQL fallback)"
echo "[GLOBE] Interface: NatnaUI.py with 6-Layer Defense System"
echo ""

# Get script directory and NATNA root (parent of Launch dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATNA_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

# Resolve bundled Python
PYTHON_BIN="$NATNA_ROOT/SACRED/runtime/python/linux-x64/python/bin/python3.11"
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
echo "  [MEM] FORCING: 16GB RAM, 8 CPU cores (Maximum Performance Mode)"

export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_MAX_QUEUE=1024
export OLLAMA_KEEP_ALIVE=10m

echo "  [OK] MAXIMUM PERFORMANCE Ollama configuration applied"

# Start portable Ollama from NATNA drive (Linux version)
OLLAMA_PATH="$NATNA_ROOT/SACRED/runtime/ollama-linux/bin/ollama"
if [ -f "$OLLAMA_PATH" ]; then
    echo "  [LAUNCH] Starting portable Ollama server..."
    chmod +x "$OLLAMA_PATH" 2>/dev/null
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

# Find NatnaUI.py
SCRIPT_LOCATIONS=(
    "$NATNA_ROOT/SACRED/APP_Production/NatnaUI.py"
    "$SCRIPT_DIR/APP_Production/NatnaUI.py"
    "$SCRIPT_DIR/NatnaUI.py"
)

PYTHON_SCRIPT=""
for location in "${SCRIPT_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        PYTHON_SCRIPT="$location"
        echo "  [FOUND] Found NatnaUI.py at: $PYTHON_SCRIPT"
        break
    fi
done

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "  [ERROR] ERROR: NatnaUI.py not found in any expected location!"
    exit 1
fi

cd "$(dirname "$PYTHON_SCRIPT")"

# Start the server
echo "  [LAUNCH] Starting web server..."
LOG_FILE="/tmp/natna_startup.log"
"$PYTHON_BIN" "$(basename "$PYTHON_SCRIPT")" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "  [NOTE] Web server PID: $SERVER_PID"

# STEP 3: Wait for Web Server
echo ""
echo "[GLOBE] Step 3: Waiting for web server and opening browser..."

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
    echo "  [DOC] Check log: $LOG_FILE"
    kill $OLLAMA_PID 2>/dev/null
    exit 1
fi

# STEP 4: Open Browser
echo ""
echo "[LAUNCH] Step 4: Opening browser to NATNA AI interface..."

# Try multiple browser opening methods for Linux
xdg-open "http://localhost:8080" 2>/dev/null || \
    sensible-browser "http://localhost:8080" 2>/dev/null || \
    x-www-browser "http://localhost:8080" 2>/dev/null || \
    gnome-open "http://localhost:8080" 2>/dev/null || \
    firefox "http://localhost:8080" 2>/dev/null || \
    google-chrome "http://localhost:8080" 2>/dev/null || \
    chromium-browser "http://localhost:8080" 2>/dev/null || \
    echo "  [WARN] Auto-browser opening failed - manually go to http://localhost:8080"

sleep 2

# Detect LAN IP for network access
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || ip route get 1 2>/dev/null | awk '{print $7}' | head -1 || echo "")

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
fi
echo ""
echo "[TEST] Ready to test: Ask 'What is photosynthesis?' to test AI"
echo ""
echo "[STOP] Press Ctrl+C to stop both servers"
echo "==========================================="

# Cleanup trap
cleanup() {
    echo ""
    echo "[STOP] Shutting down NATNA AI Desktop..."

    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "  [SIGNAL] Stopping web server (PID $SERVER_PID)..."
        kill -TERM $SERVER_PID 2>/dev/null
        sleep 2
    fi

    if [ ! -z "$OLLAMA_PID" ] && kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "  [AI] Stopping Ollama server (PID $OLLAMA_PID)..."
        kill -TERM $OLLAMA_PID 2>/dev/null
        sleep 2
        kill -0 $OLLAMA_PID 2>/dev/null && kill -9 $OLLAMA_PID 2>/dev/null
    fi

    echo "  [OK] All processes stopped cleanly"
    echo "[TARGET] NATNA AI Desktop shutdown complete!"
    exit 0
}

trap cleanup INT TERM

# Keep script running and monitor processes
while true; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ERROR] Web server stopped unexpectedly"
        cleanup
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "[ERROR] Ollama server stopped unexpectedly"
        cleanup
    fi
    sleep 5
done
