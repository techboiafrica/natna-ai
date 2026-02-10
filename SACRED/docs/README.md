# NATNA AI - Bilingual Educational Assistant

NATNA is an offline-capable bilingual AI assistant providing educational content in English and Tigrinya. Designed to run from a USB drive without requiring constant internet connectivity.

## Features

- **Bilingual Responses** - English and Tigrinya (ትግርኛ) side by side
- **13+ AI Models** - From lightweight (522MB) to advanced coding models (16GB+)
- **475,000+ Wikipedia Articles** - Domain-specific knowledge bases
- **265,000+ Tigrinya Translations** - Comprehensive dictionary database
- **7 Knowledge Domains** - Medical, Education, College, Technical, Agriculture, Programming, General
- **Mobile Access** - Connect from phone/tablet via WiFi with QR code
- **Cross-Platform** - Windows, macOS, and Linux support
- **Offline Operation** - Works without internet after initial setup

## Quick Start

1. Open the `Launch NATNA` folder at the drive root
2. Double-click the launcher for your platform:
   - **Mac**: `NATNA_Launch_Mac.command`
   - **Windows**: `NATNA_Launch_Windows.bat`
   - **Linux**: `NATNA_Launch_Linux.sh`
3. Browser opens automatically to `http://localhost:8080`
4. For phone/tablet access, scan the QR code or use the LAN IP shown in terminal

See [QUICK_START.md](QUICK_START.md) for detailed instructions including mobile setup.

## Documentation

| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | Launching the app, mobile access, first steps |
| [AI_SYSTEM.md](AI_SYSTEM.md) | AI models, translation pipeline, Ollama integration |
| [DATABASE_SYSTEM.md](DATABASE_SYSTEM.md) | Wikipedia databases, Tigrinya dictionary, search system |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Code architecture, key files, extending the system |
| [API_REFERENCE.md](API_REFERENCE.md) | HTTP API endpoints |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| Storage | 15 GB free | 20 GB free |
| Python | 3.9+ | 3.11+ |
| OS | Windows 10, macOS 10.15+, Ubuntu 20.04+ | Latest |

## Available AI Models

| Category | Models | RAM Required |
|----------|--------|--------------|
| **Small/Fast** | Qwen 3 0.6B (default), SmolLM2 360M, SmolLM3 3B, Phi-4 Mini | 4 GB |
| **Uncensored** | DeepSeek-R1 Abliterated (4GB/16GB variants) | 4-16 GB |
| **Coding** | Qwen Coder, DeepSeek Coder, CodeGemma, CodeLlama | 8-16 GB |

## Directory Structure

```
NATNA AI/
├── Launch NATNA/              # Platform-specific launchers
│   ├── NATNA_Launch_Mac.command
│   ├── NATNA_Launch_Windows.bat
│   └── NATNA_Launch_Linux.sh
├── Shutdown NATNA/            # Process cleanup scripts
└── SACRED/
    ├── APP_Production/        # Application code
    │   ├── NatnaUI.py
    │   ├── intelligent_translator.py
    │   └── ...
    ├── docs/                  # Documentation (you are here)
    ├── educational_archive/
    │   └── knowledge/         # Wikipedia databases
    ├── organized_data/
    │   └── databases/         # Tigrinya translation databases
    └── runtime/
        ├── models/            # Ollama AI models (62GB)
        ├── ollama/            # Mac binary
        ├── ollama-windows/    # Windows binary
        └── ollama-linux/      # Linux binary
```

## Technology Stack

- **Python 3.9+** - Core application
- **Ollama** - Local AI model inference
- **SQLite + FTS5** - Full-text search across databases
- **HTTP Server** - Built-in Python server on port 8080

## License

Educational use.
