# NATNA AI

**Offline Education AI for Tigray and the Global South**

NATNA is a self-contained AI education platform that runs entirely offline. It combines large language models, a 142,000+ entry Tigrinya-English dictionary, curated Wikipedia knowledge bases, and domain-specific knowledge cards for medicine, agriculture, mental health, and more.

Built for communities with limited or no internet access.

## Quick Start

```bash
git clone https://github.com/techboiafrica/natna-ai.git
cd natna-ai
python installer/natna_setup.py
```

The installer automatically:
1. Downloads knowledge databases (~6 GB) from Hugging Face
2. Installs Ollama (if not already installed)
3. Lets you choose which AI models to pull
4. Verifies everything with SHA-256 checksums

Then launch:
```bash
# macOS
./Launch_NATNA_all_OS/NATNA_Launch_Mac.command

# Linux
./Launch_NATNA_all_OS/NATNA_Launch_Linux.sh

# Windows
Launch_NATNA_all_OS\NATNA_Launch_Windows.bat
```

## What Gets Downloaded

| Component | Size | Source |
|-----------|------|--------|
| Knowledge databases | ~6 GB | Hugging Face (auto) |
| AI models (your choice) | 0.4–65 GB | Ollama registry (auto) |
| Ollama runtime | 29 MB–5 GB | ollama.com (auto) |

**Default install (recommended models):** ~16 GB total
**Full install (all models):** ~71 GB total

## Features

- **Bilingual AI** — English and Tigrinya (ትግርኛ), with 142K+ hand-cleaned translation entries
- **Domain intelligence** — Specialized responses for medicine, agriculture, mental health, education, programming, and more
- **Wikipedia knowledge** — 4.4 GB curated Wikipedia database for context-enriched answers
- **Knowledge cards** — Structured medical, agricultural, and mental health reference data
- **Multiple AI models** — Switch between models in the UI. From 360M parameters to 14B+
- **Fully offline** — Everything runs locally after setup. No cloud, no API keys, no subscriptions
- **Cross-platform** — macOS (Intel & Apple Silicon), Linux, Windows

## Model Choices

| Model | Size | Best For |
|-------|------|----------|
| qwen2.5:0.5b | 397 MB | Fast responses, low RAM |
| smollm2:360m | 692 MB | Compact multilingual |
| qwen3:0.6b | 498 MB | Latest Qwen small |
| alibayram/smollm3 | 1.8 GB | SmolLM3 |
| phi4-mini | 2.4 GB | Microsoft Phi-4 Mini |
| deepseek-r1:1.5b | 1.1 GB | Reasoning |
| deepseek-r1-abliterated | 4.5 GB | Uncensored reasoning |
| deepseek-r1:14b | 9.0 GB | Best quality (needs 16 GB RAM) |
| qwen2.5-coder:7b | 4.7 GB | Coding |
| deepseek-coder:6.7b | 3.8 GB | Coding |

## System Requirements

- **Minimum:** 4 GB RAM, 20 GB disk, Python 3.10+
- **Recommended:** 8 GB RAM, 50 GB disk
- **Full install:** 16 GB RAM, 100 GB disk

## Project Structure

```
natna-ai/
├── SACRED/
│   ├── APP_Production/          # Python source code + UI
│   │   ├── NatnaUI.py           # Main web UI (Flask)
│   │   ├── intelligent_translator.py  # AI engine
│   │   ├── context_manager.py   # Conversation context
│   │   ├── path_config.py       # Cross-platform paths
│   │   ├── qrcodegen.py         # QR code generation
│   │   ├── assets/              # KaTeX math rendering
│   │   └── dependencies/        # Vendored Python packages
│   ├── educational_archive/     # Wikipedia databases (downloaded)
│   ├── organized_data/          # Translation DBs + config
│   ├── docs/                    # Documentation
│   └── tests/                   # Test suite
├── Launch_NATNA_all_OS/         # Platform launch scripts
├── Shutdown_NATNA_all_OS/       # Platform shutdown scripts
├── installer/
│   ├── natna_setup.py           # Bootstrap installer
│   ├── manifest.json            # SHA-256 checksums
│   └── generate_manifest.py     # Manifest generator (dev tool)
└── README.md
```

## Shutting Down

```bash
# macOS
./Shutdown_NATNA_all_OS/KILL_NATNA_Processes_Mac.command

# Linux
./Shutdown_NATNA_all_OS/KILL_NATNA_Processes_Linux.sh

# Windows
Shutdown_NATNA_all_OS\KILL_NATNA_Processes_Windows.bat
```

## Re-running the Installer

The installer is idempotent — re-run it safely at any time:
- Skips already-downloaded databases (verified by checksum)
- Skips already-installed models
- Resumes interrupted downloads

```bash
python installer/natna_setup.py           # Interactive
python installer/natna_setup.py --all     # Install all models
python installer/natna_setup.py --minimal # Smallest model only
```

## License

MIT

## Credits

Built by [techboiafrica](https://github.com/techboiafrica) for Tigray and the Global South.
