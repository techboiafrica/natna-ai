#!/bin/bash
# NATNA Process Cleanup Script

echo "[CLEANUP] Killing all NATNA processes..."

# Step 1: Kill Python server and Ollama processes
echo "[KILL] Killing NATNA processes..."
pkill -9 -f "NatnaUI" 2>/dev/null
pkill -9 -f "intelligent_translator" 2>/dev/null
pkill -9 -f "ollama" 2>/dev/null

# Step 2: Clear ports (cross-platform)
echo "[PORTS] Clearing ports..."
# Kill processes on port 8080 and other NATNA ports
if command -v lsof >/dev/null 2>&1; then
    # Unix-like systems (macOS, Linux)
    lsof -ti:8080 | xargs kill -9 2>/dev/null
    lsof -ti:8081 | xargs kill -9 2>/dev/null
    lsof -ti:8082 | xargs kill -9 2>/dev/null
    lsof -ti:11434 | xargs kill -9 2>/dev/null
elif command -v netstat >/dev/null 2>&1; then
    # Windows with netstat
    netstat -ano | grep ":8080" | awk '{print $5}' | xargs -r taskkill /PID /F 2>/dev/null
    netstat -ano | grep ":8081" | awk '{print $5}' | xargs -r taskkill /PID /F 2>/dev/null
    netstat -ano | grep ":8082" | awk '{print $5}' | xargs -r taskkill /PID /F 2>/dev/null
    netstat -ano | grep ":11434" | awk '{print $5}' | xargs -r taskkill /PID /F 2>/dev/null
fi

# Step 3: Wait a moment for cleanup
sleep 2

# Step 4: Verify all processes are killed
echo "[CHECK] Verifying cleanup..."
REMAINING=$(ps aux | grep -E "(NatnaUI|intelligent_translator|ollama)" | grep -v grep | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo ""
    echo "[OK] ALL BACKGROUND PROCESSES KILLED AND CLEARED"
    echo ""
    echo "[RESTART] Now use NATNA_Launch_Mac.command to restart"
    echo ""
else
    echo "[WARN] Warning: $REMAINING processes may still be running"
    echo "You may need to manually kill remaining processes"
fi