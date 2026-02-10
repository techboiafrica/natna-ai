#!/bin/bash
# NATNA Process Cleanup Script - Linux

echo "[CLEANUP] Killing all NATNA processes..."

# Step 1: Kill Python server and Ollama processes
echo "[KILL] Killing NATNA processes..."
pkill -9 -f "NatnaUI" 2>/dev/null
pkill -9 -f "intelligent_translator" 2>/dev/null
pkill -9 -f "ollama" 2>/dev/null

# Step 2: Clear ports
echo "[PORTS] Clearing ports..."
fuser -k 8080/tcp 2>/dev/null
fuser -k 8081/tcp 2>/dev/null
fuser -k 8082/tcp 2>/dev/null
fuser -k 11434/tcp 2>/dev/null

# Step 3: Wait a moment for cleanup
sleep 2

# Step 4: Verify all processes are killed
echo "[CHECK] Verifying cleanup..."
REMAINING=$(ps aux | grep -E "(NatnaUI|intelligent_translator|ollama)" | grep -v grep | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo ""
    echo "[OK] ALL BACKGROUND PROCESSES KILLED AND CLEARED"
    echo ""
    echo "[RESTART] Now use NATNA_Launch_Linux.sh to restart"
    echo ""
else
    echo "[WARN] Warning: $REMAINING processes may still be running"
    echo "You may need to manually kill remaining processes"
fi
