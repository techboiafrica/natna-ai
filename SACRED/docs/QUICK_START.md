# Quick Start Guide

Get NATNA running in under 2 minutes.

## Launching NATNA

### Step 1: Open the Launch Folder

Navigate to the USB drive and open the `Launch NATNA` folder.

### Step 2: Run Your Platform's Launcher

**Mac:**
- Double-click `NATNA_Launch_Mac.command`
- If blocked, right-click → Open → Open anyway

**Windows:**
- Double-click `NATNA_Launch_Windows.bat`
- If Windows Defender blocks it, click "More info" → "Run anyway"

**Linux:**
- Open terminal in the folder
- Run: `./NATNA_Launch_Linux.sh`
- Or right-click → Run as Program

### Step 3: Wait for Startup

The launcher will:
1. Start the Ollama AI server
2. Preload the default Qwen 3 0.6B model
3. Start the web interface
4. Open your browser automatically

You'll see status messages in the terminal. Wait for:
```
[OK] Web server responding!
[SUCCESS] NATNA AI DESKTOP SUCCESSFULLY LAUNCHED!
```

### Step 4: Start Using NATNA

Your browser opens to `http://localhost:8080`. Type a question and press Enter.

**Try these:**
- "What is photosynthesis?"
- "ብዛዕባ ጥዕና ንገረኒ" (Tell me about health)
- "Explain the water cycle for a 5th grader"

---

## Mobile Access (Phone/Tablet)

NATNA can be accessed from any device on the same WiFi network.

### Method 1: QR Code (Easiest)

1. Look for the **📱 Connect** button in the bottom-right of the web interface
2. Click it to show a QR code
3. Scan with your phone's camera
4. Your phone opens NATNA automatically

### Method 2: Manual IP Entry

1. Look at the terminal window after launch
2. Find the line that says:
   ```
   [MOBILE] ACCESS FROM PHONE/TABLET (same Wi-Fi):
   [MOBILE]   http://192.168.x.x:8080
   ```
3. Type that address into your phone's browser

### Mobile Troubleshooting

**Can't connect from phone?**
- Ensure phone and computer are on the same WiFi network
- Check your computer's firewall:
  - **Mac**: System Settings → Network → Firewall → Allow Python
  - **Windows**: Allow through Windows Defender Firewall
  - **Linux**: `sudo ufw allow 8080`

---

## Interface Overview

### Main Areas

| Area | Description |
|------|-------------|
| **Domain Selector** | Choose knowledge focus (Medical, Education, etc.) |
| **Model Selector** | Choose AI model based on your RAM |
| **Chat Area** | Conversation display |
| **Input Box** | Type questions here |
| **Research Panel** | Wikipedia sources for current query |
| **Status Bar** | Ollama status, Wikipedia status, RAM usage |

### Key Buttons

| Button | Function |
|--------|----------|
| **Expand Answer** | Enhance AI response with Wikipedia context |
| **Clear** | Reset conversation history |
| **Dark/Light** | Toggle dark/light theme |
| **EN/TI** | Toggle between English-only (blue) and bilingual mode (yellow/red) |
| **Sources** (mobile) | Open research panel with Wikipedia articles |

**Note:** Wikipedia search happens automatically when you ask questions - no manual search button needed.

---

## Choosing a Domain

Select a domain to focus the AI on specific knowledge:

| Domain | Best For |
|--------|----------|
| **General** | Broad questions, everyday topics |
| **Medical** | Health, symptoms, treatments (78,000+ articles) |
| **Education** | K-12 learning, basic concepts |
| **College** | University-level topics (24,000+ articles) |
| **Technical** | Engineering, vocational skills (26,000+ articles) |
| **Agriculture** | Farming, crops, livestock |
| **Programming** | Code questions (auto-selects coding model) |

---

## Choosing a Model

Select based on your available RAM:

| RAM | Recommended Model |
|-----|-------------------|
| **4 GB** | Qwen 3 0.6B (default) - fast, good quality |
| **8 GB** | Phi-4 Mini or coding models |
| **16 GB+** | DeepSeek Coder V2, Qwen Coder 14B |

**Uncensored models** (DeepSeek-R1 Abliterated) provide unfiltered responses for sensitive topics.

**Mobile Model Changes:** Changing the AI model from a mobile device requires an admin password. This prevents accidental model changes that affect all connected users. Desktop users can change models freely.

The default admin password is `Mekelle8080`. To change it, set the `NATNA_ADMIN_PW` environment variable before launching NATNA:
```bash
# Mac/Linux
export NATNA_ADMIN_PW='YourNewPassword'

# Windows
set NATNA_ADMIN_PW=YourNewPassword
```
Rate limiting is enforced: max 5 password attempts per minute per device.

---

## Stopping NATNA

### Method 1: Terminal
Press `Ctrl+C` in the terminal window.

### Method 2: Kill Scripts
Use the scripts in `Shutdown NATNA` folder:
- Mac: `KILL_NATNA_Processes_Mac.command`
- Windows: `KILL_NATNA_Processes_Windows.bat`
- Linux: `KILL_NATNA_Processes_Linux.sh`

---

## First Run Checklist

- [ ] Launcher opens without errors
- [ ] Browser shows NATNA interface
- [ ] Status bar shows "Rhizome Server: Connected"
- [ ] Status bar shows "Wikipedia: X articles"
- [ ] Test query returns bilingual response
- [ ] (Optional) Phone can connect via WiFi

---

## Next Steps

- [AI_SYSTEM.md](AI_SYSTEM.md) - Learn about the AI models
- [DATABASE_SYSTEM.md](DATABASE_SYSTEM.md) - Understand the knowledge bases
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - If something isn't working
