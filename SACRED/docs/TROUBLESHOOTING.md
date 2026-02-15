# Troubleshooting Guide

Solutions to common NATNA issues.

## Startup Issues

### Launcher Won't Run

**Mac - "Cannot be opened because the developer cannot be verified"**
1. Right-click the `.command` file
2. Select "Open"
3. Click "Open" in the dialog

**Windows - "Windows protected your PC"**
1. Click "More info"
2. Click "Run anyway"

**Linux - Permission denied**
```bash
chmod +x NATNA_Launch_Linux.sh
./NATNA_Launch_Linux.sh
```

### "NATNA is already running!"

Another instance is running. Either:
1. Use the kill scripts in `Shutdown NATNA` folder
2. Or manually:
```bash
# Mac/Linux
pkill -f NatnaUI
pkill -f ollama

# Windows
taskkill /F /IM python.exe
taskkill /F /IM ollama.exe
```

### "Address already in use (port 8080)"

Another app is using port 8080:
```bash
# Find the process
# Mac/Linux
lsof -i :8080

# Windows
netstat -ano | findstr :8080
```

Kill the process or change NATNA's port in `NatnaUI.py`.

---

## Browser Issues

### Incognito / Private Browsing Not Supported

NATNA does not work reliably in incognito or private browser windows. Queries may return "Error: No response" even when the server is running and responding normally in regular browser windows.

**Use a regular browser window.** This applies to all browsers (Chrome, Firefox, Safari, Edge). Incognito/private mode enforces stricter local network restrictions that can block communication with the local NATNA server.

---

## Connection Issues

### "Rhizome Server: Disconnected"

Ollama is not running.

**Check if Ollama is running:**
```bash
curl http://localhost:11434
# Should return "Ollama is running"
```

**Manually start Ollama:**
```bash
# Mac
/Volumes/Natna\ v1/NATNA\ AI/SACRED/runtime/ollama/ollama serve

# Windows
"D:\NATNA AI\SACRED\runtime\ollama-windows\ollama.exe" serve

# Linux
/path/to/SACRED/runtime/ollama-linux/bin/ollama serve
```

### Wikipedia Shows "0 articles"

Database not found or warming not complete.

**Check database exists:**
```bash
ls -la "SACRED/educational_archive/knowledge/massive_wikipedia.db"
```

**Wait for warming:**
- Check terminal for "Warming phase X complete"
- Warming can take 30-60 seconds on slow drives

### Can't Connect from Phone

**Same WiFi network?**
- Phone and computer must be on the same WiFi

**Check firewall:**
- Mac: System Settings → Network → Firewall → Allow Python
- Windows: Windows Defender Firewall → Allow an app → Python
- Linux: `sudo ufw allow 8080`

**Get correct IP:**
```bash
# Mac
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'

# Windows
ipconfig | findstr IPv4
```

---

## Response Issues

### Slow First Response

Normal - the AI model is loading into memory. Subsequent responses will be faster.

### No Tigrinya Translation

**Check database connection:**
```bash
sqlite3 "SACRED/organized_data/databases/massive_tigrinya_database.db" ".tables"
```

**Verify path_config.py:**
```python
from path_config import TIGRINYA_DB
print(TIGRINYA_DB)
```

### Gibberish Output

Usually a model issue:
1. Try a different model (Qwen 3 0.6B is most stable)
2. Clear history and try again
3. Restart NATNA

### "Context limit reached"

The conversation is too long for the model's context window.
1. Click "Clear History"
2. Or switch to a model with larger context (16GB models have 16K tokens)

---

## Performance Issues

### High RAM Usage

**Switch to smaller model:**
- Qwen 3 0.6B uses ~522MB
- SmolLM2 360M uses ~726MB

**Check what's using RAM:**
```bash
# Mac/Linux
ps aux | grep -E "python|ollama" | grep -v grep
```

### Queries Taking Too Long

**Check Wikipedia warming:**
- Queries during warming are slower
- Wait for "Wikipedia ready" status

**Try smaller result limits:**
- Wikipedia searches with fewer results are faster

**Check disk speed:**
- USB 3.0+ recommended
- USB 2.0 will be significantly slower

---

## Database Issues

### "No such table: articles"

Database may be corrupted.

**Check integrity:**
```bash
sqlite3 massive_wikipedia.db "PRAGMA integrity_check;"
```

**If corrupted:** Restore from backup in `/Volumes/Natna v1/Archive/`

### "Database is locked"

Multiple processes accessing the database.

**Stop all NATNA processes:**
```bash
pkill -f NatnaUI
pkill -f intelligent_translator
```

### Search Returns No Results

**Rebuild FTS index:**
```bash
sqlite3 massive_wikipedia.db "INSERT INTO articles_fts(articles_fts) VALUES('rebuild');"
```

---

## Platform-Specific Issues

### Mac

**USB drive not found:**
```bash
ls /Volumes/
# Look for "Natna v1" or similar
```

**Security permissions:**
- System Settings → Privacy & Security → Allow apps from identified developers

### Windows

**Path with spaces:**
Always use quotes around paths:
```cmd
cd "D:\NATNA AI\SACRED"
```

**Python not found:**
- Use `py` instead of `python`
- Or reinstall Python with "Add to PATH" checked

### Linux

**Ollama won't start:**
```bash
# Check if already running
pgrep ollama

# Check permissions
chmod +x SACRED/runtime/ollama-linux/bin/ollama
```

**USB mount permissions:**
```bash
sudo chmod -R 755 /media/user/Natna\ v1
```

---

## Getting Debug Information

### Check All Status

Visit: `http://localhost:8080/api/status/all`

### View Startup Log

Look at the terminal window where you launched NATNA.

### Wikipedia Debug Log

```bash
cat /tmp/natna_wiki_debug.txt
```

### Test API Directly

```bash
# Test chat
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "domain": "general", "model": "qwen3:0.6b"}'

# Test Wikipedia
curl -X POST http://localhost:8080/api/search_wikipedia \
  -H "Content-Type: application/json" \
  -d '{"query": "photosynthesis"}'
```

---

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| "No response" in incognito | Use a regular browser window |
| Won't start | Run kill script, try again |
| Ollama disconnected | Check terminal, restart launcher |
| No Wikipedia | Wait for warming, check db path |
| Phone can't connect | Same WiFi, check firewall |
| Slow responses | Smaller model, clear history |
| High RAM | Switch to Qwen 3 0.6B |
| Database locked | Kill all processes |
| No Tigrinya | Check database path |

---

## Still Stuck?

1. Check all status: `/api/status/all`
2. Look at terminal output for errors
3. Try restarting with kill script first
4. Check terminal startup logs for specific error messages
