#!/usr/bin/env python3
"""
NATNA AI Bootstrap Installer

Downloads knowledge databases from Hugging Face, installs Ollama,
pulls AI models, and sets up the Python runtime — all automatically.

Usage:
    python installer/natna_setup.py          # Interactive (default models)
    python installer/natna_setup.py --all    # Install all models
    python installer/natna_setup.py --minimal # Smallest model only
"""

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO = "techboiafrica/natna-databases"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

BANNER = r"""
╔══════════════════════════════════════╗
║         NATNA AI Setup               ║
║   Offline Education AI for Tigray    ║
╚══════════════════════════════════════╝
"""

# Model presets: (name, size_display, description, default)
MODELS = [
    ("qwen2.5:0.5b",              "397 MB",  "Default, lightweight",       True),
    ("smollm2:360m",              "692 MB",  "Compact multilingual",       True),
    ("qwen3:0.6b",                "498 MB",  "Latest Qwen small",         True),
    ("alibayram/smollm3",         "1.8 GB",  "SmolLM3",                   True),
    ("phi4-mini",                 "2.4 GB",  "Microsoft Phi-4 Mini",      True),
    ("deepseek-r1:1.5b",         "1.1 GB",  "DeepSeek reasoning",        True),
    ("vanilj/deepseek-r1-abliterated:Q4_K_M", "4.5 GB", "Uncensored reasoning", True),
    ("deepseek-r1:14b",          "9.0 GB",  "Needs 16 GB RAM",           False),
    ("qwen2.5-coder:7b",         "4.7 GB",  "Coding assistant",          False),
    ("deepseek-coder:6.7b",      "3.8 GB",  "Coding assistant",          False),
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def detect_platform():
    """Return (os_name, arch) tuple."""
    os_name = platform.system().lower()  # linux, darwin, windows
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine
    return os_name, arch


def human_size(nbytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def sha256_file(filepath, chunk_size=1 << 20):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest, expected_size=None, expected_sha256=None):
    """
    Download a file with progress bar and resume support.

    Uses HTTP Range headers for resuming interrupted downloads.
    Falls back to full download if the server ignores the Range
    request (responds 200 instead of 206 Partial Content).

    Returns True on success, False on failure.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    if dest.exists() and expected_sha256:
        if sha256_file(dest) == expected_sha256:
            return True

    # Resume support: check partial download
    partial = dest.with_suffix(dest.suffix + ".partial")
    start_byte = 0
    if partial.exists():
        start_byte = partial.stat().st_size

    headers = {}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable — file complete
            if partial.exists():
                partial.rename(dest)
            return True
        print(f"\n  Error downloading {url}: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"\n  Error downloading {url}: {e}")
        return False

    # Verify the server actually honored our Range request.
    # HTTP 206 = Partial Content (resume works).
    # HTTP 200 = Full content (server ignored Range header,
    #   e.g. CDN redirect stripped the header or server
    #   doesn't support byte-range requests).
    # If we get 200 when we asked for a range, we must
    # overwrite the partial file instead of appending,
    # otherwise we'd end up with partial + full = corrupt.
    resp_code = resp.getcode()
    if start_byte > 0 and resp_code != 206:
        print(
            f"  Server did not honor resume"
            f" (HTTP {resp_code}). Restarting download."
        )
        start_byte = 0

    # Determine total size
    content_range = resp.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = int(content_range.split("/")[-1])
    elif expected_size:
        total = expected_size
    else:
        cl = resp.headers.get("Content-Length")
        total = int(cl) + start_byte if cl else None

    mode = "ab" if start_byte > 0 else "wb"
    downloaded = start_byte
    chunk_size = 1 << 20  # 1 MB

    try:
        with open(partial, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = downloaded / total
                    bar_len = 30
                    filled = int(bar_len * pct)
                    bar = (
                        "=" * filled
                        + ">"
                        + " " * (bar_len - filled - 1)
                    )
                    print(
                        f"\r  [{bar}] {pct*100:5.1f}%"
                        f"  {human_size(downloaded)}"
                        f"/{human_size(total)}  ",
                        end="", flush=True,
                    )
                else:
                    print(
                        f"\r  Downloaded"
                        f" {human_size(downloaded)}  ",
                        end="", flush=True,
                    )

        print()  # newline after progress bar
    except Exception as e:
        print(f"\n  Download interrupted: {e}")
        print(f"  Partial file saved. Re-run installer to resume.")
        return False

    # Verify checksum
    if expected_sha256:
        actual = sha256_file(partial)
        if actual != expected_sha256:
            print(f"  CHECKSUM MISMATCH for {dest.name}!")
            print(f"    Expected: {expected_sha256}")
            print(f"    Got:      {actual}")
            partial.unlink(missing_ok=True)
            return False

    # Move partial to final
    partial.rename(dest)
    return True


# ---------------------------------------------------------------------------
# Step 1: Download databases from Hugging Face
# ---------------------------------------------------------------------------

def download_databases(sacred_dir, manifest):
    """Download all database files from HF to the SACRED directory."""
    print("\nStep 1/4: Downloading knowledge databases...")
    files = manifest.get("files", {})
    total_size = sum(f["size"] for f in files.values())
    print(f"  {len(files)} files, {human_size(total_size)} total\n")

    succeeded = 0
    skipped = 0
    failed = 0

    for rel_path, info in files.items():
        dest = sacred_dir / rel_path
        name = Path(rel_path).name

        # Already downloaded and verified?
        if dest.exists() and dest.stat().st_size == info["size"]:
            existing_hash = sha256_file(dest)
            if existing_hash == info["sha256"]:
                print(f"  {name} — already downloaded. Skipping.")
                skipped += 1
                continue

        url = f"{HF_BASE_URL}/{rel_path}"
        print(f"  Downloading {name} ({human_size(info['size'])})...")

        ok = download_file(
            url, dest,
            expected_size=info["size"],
            expected_sha256=info["sha256"],
        )
        if ok:
            succeeded += 1
        else:
            failed += 1

    print(f"\n  Databases: {succeeded} downloaded, {skipped} already present, {failed} failed.")
    return failed == 0


# ---------------------------------------------------------------------------
# Step 2: Install Ollama
# ---------------------------------------------------------------------------

def check_ollama():
    """Check if Ollama is installed and return its path, or None."""
    return shutil.which("ollama")


def install_ollama():
    """Install Ollama for the current platform."""
    print("\nStep 2/4: Installing Ollama...")

    if check_ollama():
        print("  Ollama already installed. ✓")
        return True

    os_name, arch = detect_platform()

    if os_name == "darwin":
        print("  Downloading Ollama for macOS...")
        url = "https://ollama.com/download/Ollama-darwin.zip"
        tmp = Path(tempfile.mkdtemp())
        zip_path = tmp / "Ollama-darwin.zip"

        ok = download_file(url, zip_path)
        if not ok:
            print("  Failed to download Ollama. Install manually: https://ollama.com/download")
            return False

        print("  Extracting...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(tmp)], check=True)

        app_src = tmp / "Ollama.app"
        app_dest = Path("/Applications/Ollama.app")
        if app_src.exists():
            if app_dest.exists():
                shutil.rmtree(app_dest)
            shutil.move(str(app_src), str(app_dest))
            print("  Ollama installed to /Applications/Ollama.app ✓")
            # Start Ollama
            subprocess.Popen(["open", str(app_dest)])
            import time
            print("  Waiting for Ollama to start...")
            time.sleep(5)
        shutil.rmtree(tmp, ignore_errors=True)
        return True

    elif os_name == "linux":
        print("  Installing Ollama via official script...")
        try:
            subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=True,
            )
            print("  Ollama installed. ✓")
            return True
        except subprocess.CalledProcessError:
            print("  Failed to install Ollama. Install manually: https://ollama.com/download")
            return False

    elif os_name == "windows":
        print("  Please download Ollama from https://ollama.com/download")
        print("  Run the installer, then re-run this setup script.")
        return False

    else:
        print(f"  Unsupported platform: {os_name}. Install Ollama manually.")
        return False


# ---------------------------------------------------------------------------
# Step 3: Pull AI models
# ---------------------------------------------------------------------------

def get_installed_models():
    """Return set of installed model names."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        installed = set()
        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            if line.strip():
                name = line.split()[0]
                # Normalize: remove :latest suffix for comparison
                installed.add(name.replace(":latest", ""))
        return installed
    except Exception:
        return set()


def select_models(mode="interactive"):
    """Let user select which models to install. Returns list of model names."""
    if mode == "all":
        return [m[0] for m in MODELS]
    elif mode == "minimal":
        return [MODELS[0][0]]  # Just the smallest

    # Interactive selection
    installed = get_installed_models()

    print("\n  Available AI models:")
    print("  " + "-" * 60)
    for i, (name, size, desc, default) in enumerate(MODELS):
        tag = ""
        if name in installed or name.replace(":latest", "") in installed:
            tag = " [installed]"
        marker = "x" if default else " "
        print(f"  [{marker}] {i+1:2d}. {name:<45s} {size:>8s}  {desc}{tag}")

    print()
    print("  Press ENTER to install defaults, or type numbers (e.g. 1,2,5):")
    print("  Type 'all' for everything, 'none' to skip models.")

    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice == "all":
        return [m[0] for m in MODELS]
    elif choice == "none":
        return []
    elif choice == "":
        return [m[0] for m, _ in zip(MODELS, range(len(MODELS))) if m[3]]
    else:
        selected = []
        for part in choice.replace(",", " ").split():
            try:
                idx = int(part) - 1
                if 0 <= idx < len(MODELS):
                    selected.append(MODELS[idx][0])
            except ValueError:
                pass
        return selected if selected else [m[0] for m in MODELS if m[3]]


def pull_models(model_names):
    """Pull selected models via ollama."""
    print("\nStep 3/4: Pulling AI models...")

    if not model_names:
        print("  No models selected. Skipping.")
        return True

    if not check_ollama():
        print("  Ollama not found. Skipping model pull.")
        return False

    installed = get_installed_models()
    all_ok = True

    for name in model_names:
        if name in installed or name.replace(":latest", "") in installed:
            print(f"  {name} — already installed. ✓")
            continue

        print(f"  Pulling {name}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", name],
                timeout=3600,  # 1 hour max per model
            )
            if result.returncode == 0:
                print(f"  {name} ✓")
            else:
                print(f"  {name} — pull failed (exit {result.returncode})")
                all_ok = False
        except subprocess.TimeoutExpired:
            print(f"  {name} — timed out after 1 hour")
            all_ok = False
        except Exception as e:
            print(f"  {name} — error: {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Step 4: Python runtime (bundled dependencies check)
# ---------------------------------------------------------------------------

def setup_python_runtime(sacred_dir):
    """Verify Python dependencies are available."""
    print("\nStep 4/4: Checking Python runtime...")

    # The app bundles its dependencies in APP_Production/dependencies/
    deps_dir = sacred_dir / "APP_Production" / "dependencies"
    if deps_dir.exists():
        print(f"  Bundled dependencies found at {deps_dir} ✓")
    else:
        print(f"  Warning: dependencies/ not found. The app ships its own packages.")

    # Check Python version
    py_version = platform.python_version()
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        print(f"  Python {py_version} ✓")
    else:
        print(f"  Warning: Python {py_version} detected. Python 3.10+ recommended.")

    # Check psutil (needed for RAM counter)
    try:
        import psutil
        print("  psutil available ✓")
    except ImportError:
        print("  Installing psutil for RAM monitoring...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "psutil"],
                capture_output=True, check=True,
            )
            print("  psutil installed ✓")
        except Exception:
            print("  Warning: Could not install psutil. RAM counter may show N/A.")

    return True


# ---------------------------------------------------------------------------
# Step 5: Set launch script permissions
# ---------------------------------------------------------------------------

def fix_permissions(repo_dir):
    """Make launch/shutdown scripts executable."""
    print("\nFinalizing...")
    for script_dir in ("Launch_NATNA_all_OS", "Shutdown_NATNA_all_OS"):
        d = repo_dir / script_dir
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix in (".command", ".sh"):
                f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                print(f"  Made executable: {f.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(BANNER)

    # Detect platform
    os_name, arch = detect_platform()
    os_display = {"darwin": "macOS", "linux": "Linux", "windows": "Windows"}.get(os_name, os_name)
    arch_display = {"arm64": "ARM64 (Apple Silicon)" if os_name == "darwin" else "ARM64",
                    "x86_64": "x86_64 (Intel)"}.get(arch, arch)
    print(f"Detected platform: {os_display} {arch_display}")
    print(f"Python: {platform.python_version()}")

    # Locate repo root (installer/ is inside it)
    installer_dir = Path(__file__).parent.resolve()
    repo_dir = installer_dir.parent

    # SACRED dir is where databases go
    sacred_dir = repo_dir / "SACRED"
    sacred_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = installer_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"\nError: {manifest_path} not found.")
        print("The installer package appears incomplete.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Parse CLI args
    mode = "interactive"
    if "--all" in sys.argv:
        mode = "all"
    elif "--minimal" in sys.argv:
        mode = "minimal"

    # Step 1: Databases
    db_ok = download_databases(sacred_dir, manifest)

    # Step 2: Ollama
    ollama_ok = install_ollama()

    # Step 3: Models
    if mode == "interactive":
        model_names = select_models("interactive")
    else:
        model_names = select_models(mode)
    models_ok = pull_models(model_names)

    # Step 4: Python runtime
    runtime_ok = setup_python_runtime(sacred_dir)

    # Step 5: Permissions
    fix_permissions(repo_dir)

    # Summary
    print("\n" + "=" * 50)
    status = []
    if db_ok:
        status.append("Databases: ✓")
    else:
        status.append("Databases: Some failed — re-run to retry")
    if ollama_ok:
        status.append("Ollama: ✓")
    else:
        status.append("Ollama: Not installed — install from ollama.com")
    if models_ok:
        status.append(f"Models: ✓ ({len(model_names)} selected)")
    else:
        status.append("Models: Some failed — re-run to retry")
    status.append(f"Python: ✓ ({platform.python_version()})")

    for s in status:
        print(f"  {s}")

    # Launch instructions
    print("\n" + "=" * 50)
    if os_name == "darwin":
        launch = "./Launch_NATNA_all_OS/NATNA_Launch_Mac.command"
    elif os_name == "linux":
        launch = "./Launch_NATNA_all_OS/NATNA_Launch_Linux.sh"
    elif os_name == "windows":
        launch = r"Launch_NATNA_all_OS\NATNA_Launch_Windows.bat"
    else:
        launch = "the appropriate launch script in Launch_NATNA_all_OS/"

    if db_ok and ollama_ok:
        print(f"\n  Setup complete! Run:\n    {launch}\n")
    else:
        print(f"\n  Setup partially complete. Fix issues above, then run:\n    {launch}\n")
        print("  Re-run this installer to retry failed steps (it skips what's done).\n")


if __name__ == "__main__":
    main()
