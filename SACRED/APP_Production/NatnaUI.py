#!/usr/bin/env python3
"""
Terminal UI - Simple, fast, no fluff
"""

from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import requests
import os
import signal
import atexit
import socket
import sys
import threading
import time
from pathlib import Path
try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import gc
import tempfile
from intelligent_translator import IntelligentTigrinyaTranslator

# === Security Constants ===
MAX_BODY_SIZE = 1_048_576  # 1 MB max POST body
_admin_attempts = {}  # Rate limiting: {ip: [timestamps]}

# === NATNA Ti BETA - Local English→Tigrinya Model ===
natna_ti_model = None
natna_ti_tokenizer = None

def load_natna_ti_model():
    """Load the locally trained English→Tigrinya translation model"""
    global natna_ti_model, natna_ti_tokenizer
    if natna_ti_model is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        model_path = str(Path(__file__).parent.parent / "All_corpus_training_data" / "academic_resources" / "reverse_helsinki_en_ti_20260102_180423" / "en_to_ti_model")
        print(f"[NATNA-TI] Loading Natna Ti BETA model from {model_path}...")

        natna_ti_tokenizer = AutoTokenizer.from_pretrained(model_path)
        natna_ti_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Use MPS if available (Apple Silicon)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        natna_ti_model = natna_ti_model.to(device)

        print(f"[NATNA-TI] Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"[NATNA-TI] Error loading model: {e}")
        return False

def translate_with_natna_ti(text):
    """Translate English text to Tigrinya using local model"""
    global natna_ti_model, natna_ti_tokenizer
    import torch

    if natna_ti_model is None:
        if not load_natna_ti_model():
            return None

    try:
        device = next(natna_ti_model.parameters()).device
        inputs = natna_ti_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = natna_ti_model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )

        translation = natna_ti_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        print(f"[NATNA-TI] Translation error: {e}")
        return None

# === SINGLETON PROCESS LOCKING ===
# Prevent multiple NATNA instances using fcntl.flock() - research-backed approach
lock_file = None

def ensure_singleton():
    """Ensure only one NATNA instance runs. Kill script compatible."""
    global lock_file
    lock_path = '/tmp/natna.lock' if os.name != 'nt' else os.path.join(os.environ.get('TEMP', '.'), 'natna.lock')
    if FCNTL_AVAILABLE:
        try:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            print("[LOCK] NATNA singleton lock acquired")
            return True
        except (OSError, BlockingIOError):
            print("[ERROR] NATNA is already running!")
            print("   Use your kill script to stop the existing instance first.")
            sys.exit(1)
    else:
        # Fallback: PID file check (Windows or missing fcntl)
        try:
            if os.path.exists(lock_path):
                with open(lock_path, 'r') as f:
                    old_pid = int(f.read().strip())
                # Check if old process is still running
                try:
                    os.kill(old_pid, 0)
                    print("[ERROR] NATNA is already running (PID {})!".format(old_pid))
                    print("   Use your kill script to stop the existing instance first.")
                    sys.exit(1)
                except (OSError, ProcessLookupError):
                    pass  # Old process is dead, safe to proceed
            lock_file = open(lock_path, 'w')
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            print("[LOCK] NATNA singleton lock acquired (PID file)")
            return True
        except Exception:
            print("[WARN] Could not acquire singleton lock, continuing anyway")
            return True

def cleanup_singleton():
    """Clean up singleton lock on exit"""
    global lock_file
    lock_path = '/tmp/natna.lock' if os.name != 'nt' else os.path.join(os.environ.get('TEMP', '.'), 'natna.lock')
    if lock_file:
        try:
            lock_file.close()
            os.unlink(lock_path)
        except OSError:
            pass

# === MEMORY MONITORING FOR FIELD LAPTOPS ===
def get_memory_usage():
    """Get total AI model memory usage in MB including all Ollama processes"""
    if not PSUTIL_AVAILABLE:
        return {'rss': 0, 'vms': 0, 'percent': 0, 'model': 'monitoring-unavailable'}
    try:
        # First try to get model RAM from Ollama API
        import requests
        try:
            response = requests.get('http://localhost:11434/api/ps', timeout=1)
            if response.status_code == 200:
                data = response.json()
                if data.get('models'):
                    model = data['models'][0]  # Get first loaded model
                    model_ram_bytes = model.get('size_vram', 0)
                    model_ram_mb = round(model_ram_bytes / 1024 / 1024, 1)

                    # Also get web server RAM for total
                    web_process = psutil.Process()
                    web_ram_mb = round(web_process.memory_info().rss / 1024 / 1024, 1)

                    total_ram_mb = model_ram_mb + web_ram_mb
                    total_memory = psutil.virtual_memory()
                    percent = round((total_ram_mb / (total_memory.total / 1024 / 1024)) * 100, 1)

                    return {
                        'rss': total_ram_mb,  # Total AI system RAM
                        'vms': model_ram_mb,  # Model RAM specifically
                        'percent': percent,   # % of system memory
                        'model': model.get('name', 'unknown')
                    }
        except (OSError, ValueError, KeyError):
            pass

        # Fallback: sum all ollama processes
        total_ram = 0
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    total_ram += proc.info['memory_info'].rss
            except (KeyError, TypeError, AttributeError):
                continue

        if total_ram > 0:
            total_ram_mb = round(total_ram / 1024 / 1024, 1)
            total_memory = psutil.virtual_memory()
            percent = round((total_ram_mb / (total_memory.total / 1024 / 1024)) * 100, 1)
            return {
                'rss': total_ram_mb,
                'vms': total_ram_mb,
                'percent': percent,
                'model': 'ollama-processes'
            }

        # Final fallback: web server only
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': round(memory_info.rss / 1024 / 1024, 1),
            'vms': round(memory_info.vms / 1024 / 1024, 1),
            'percent': round(process.memory_percent(), 1),
            'model': 'web-server-only'
        }
    except (OSError, ValueError, AttributeError):
        return {'rss': 0, 'vms': 0, 'percent': 0, 'model': 'error'}

def scan_system_resources():
    """Scan and return all available system resources"""
    try:
        if PSUTIL_AVAILABLE:
            # Get total system memory
            virtual_memory = psutil.virtual_memory()
            total_ram_gb = round(virtual_memory.total / (1024**3), 1)
            available_ram_gb = round(virtual_memory.available / (1024**3), 1)

            # Get CPU information
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            max_freq_ghz = round(cpu_freq.max / 1000, 2) if cpu_freq else "Unknown"
        else:
            # Conservative defaults when psutil is unavailable
            total_ram_gb = 4.0
            available_ram_gb = 2.0
            cpu_count = os.cpu_count() or 2
            max_freq_ghz = "Unknown"
            print("[WARN] psutil unavailable - using conservative resource defaults")

        # Calculate optimal memory allocations
        db_memory_mb = int(total_ram_gb * 1024 * 0.25)  # 25% of total RAM for database

        resources = {
            'total_ram_gb': total_ram_gb,
            'available_ram_gb': available_ram_gb,
            'cpu_count': cpu_count,
            'max_freq_ghz': max_freq_ghz,
            'db_memory_mb': db_memory_mb
        }

        print(f"[LAUNCH] SYSTEM RESOURCES DETECTED:")
        print(f"   [MEM] Total RAM: {total_ram_gb}GB")
        print(f"   [MEM] Available RAM: {available_ram_gb}GB")
        print(f"   [FIRE] CPU Cores: {cpu_count}")
        print(f"   [FIRE] Max CPU Freq: {max_freq_ghz}GHz")
        print(f"   [STATS] Database Memory: {db_memory_mb}MB")

        return resources
    except Exception as e:
        print(f"[WARN] Resource scanning error: {e}")
        return None

def maximize_resource_utilization():
    """Configure NATNA to use ALL available system resources"""
    try:
        resources = scan_system_resources()
        if not resources:
            print("[ERROR] Could not detect system resources - using defaults")
            return False

        # REMOVE ALL MEMORY LIMITATIONS
        # Set garbage collection to Python defaults (maximum performance)
        gc.set_threshold(700, 10, 10)  # Python default - less frequent GC

        # Set process to high priority to maximize performance
        if PSUTIL_AVAILABLE:
            current_process = psutil.Process()
            if hasattr(current_process, 'nice'):
                try:
                    current_process.nice(-5)  # Higher priority (negative = higher priority on Unix)
                except OSError:
                    pass

        # Set environment variables for Ollama maximum performance
        os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Keep ONLY ONE model loaded at a time
        os.environ['OLLAMA_NUM_PARALLEL'] = str(resources['cpu_count'])  # Use all CPU cores

        print("[FIRE] MAXIMUM RESOURCE UTILIZATION ACTIVATED")
        print("   [OK] Memory limitations REMOVED")
        print("   [OK] High process priority SET")
        print("   [OK] Ollama configured for MAXIMUM performance")
        return resources
    except Exception as e:
        print(f"[WARN] Resource maximization warning: {e}")
        return False

def log_memory_status():
    """Log memory usage for field deployment monitoring"""
    memory = get_memory_usage()
    print(f"[STATS] Memory: {memory['rss']}MB physical, {memory['percent']}% system")

    # Alert if memory usage is high for field laptops
    if memory['rss'] > 500:  # 500MB threshold for field laptops
        print(f"[WARN] High memory usage: {memory['rss']}MB - consider restart if issues occur")
    elif memory['rss'] > 1000:  # 1GB critical threshold
        print(f"[CRIT] Critical memory usage: {memory['rss']}MB - field laptops may struggle")

    return memory

# Register cleanup for singleton lock
atexit.register(cleanup_singleton)

# Signal handler for graceful shutdown (kill script compatibility)
def signal_handler(signum, frame):
    print(f"\n[SIGNAL] Received signal {signum}, shutting down gracefully...")
    cleanup_singleton()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Initialize AI
print("[LOADING] Starting AI...")
translator = IntelligentTigrinyaTranslator()

# Preload the default lightweight model into memory
def warmup_model():
    """Preload the default Qwen 2.5 0.5B model so first query is fast"""
    try:
        print("[PRELOAD] Loading Qwen 2.5 0.5B AI model (this may take a moment)...")
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'qwen2.5:0.5b',
                'prompt': 'Hello',
                'stream': False
            },
            timeout=120
        )
        if response.status_code == 200:
            print("[OK] Qwen 2.5 0.5B Model preloaded and ready!")
        else:
            print(f"[WARN] Model preload returned status {response.status_code}")
    except Exception as e:
        print(f"[WARN] Model preload skipped: {e}")

# Warmup the default model
if translator.local_ai_available:
    warmup_model()

print("[OK] Ready")

class TerminalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # API endpoint to fetch full Wikipedia article by ID
        if self.path.startswith('/api/article/'):
            try:
                article_id = int(self.path.split('/')[-1])
                article = translator.get_wikipedia_article(article_id)

                if article:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(article).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Article not found"}).encode('utf-8'))
            except ValueError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid article ID"}).encode('utf-8'))
            except Exception as e:
                print(f"Error fetching article: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))

        elif self.path == '/api/status/ollama':
            # Check actual Ollama server status with auto-restart
            # Uses lightweight GET (returns "Ollama is running") instead of inference call
            global _ollama_last_restart_attempt
            ollama_connected = False
            try:
                resp = requests.get('http://localhost:11434', timeout=2)
                if resp.status_code == 200:
                    ollama_connected = True
            except Exception:
                pass

            if ollama_connected:
                status_result = {"status": "connected"}
            else:
                # Auto-restart with 30-second cooldown
                now = time.time()
                if now - _ollama_last_restart_attempt >= 30:
                    _ollama_last_restart_attempt = now
                    print("🔄 Ollama disconnected — attempting auto-restart...")
                    try:
                        success, error = _start_ollama_server()
                        if success:
                            print("[OK] Ollama auto-restart succeeded")
                            status_result = {"status": "connected"}
                        else:
                            print(f"[ERROR] Ollama auto-restart failed: {error}")
                            status_result = {"status": "disconnected"}
                    except Exception as e:
                        print(f"[ERROR] Ollama auto-restart error: {e}")
                        status_result = {"status": "disconnected"}
                else:
                    status_result = {"status": "disconnected"}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(json.dumps(status_result).encode('utf-8'))

        elif self.path == '/api/status/wikipedia':
            # Check Wikipedia search status
            if translator and hasattr(translator, 'wikipedia_available') and translator.wikipedia_available:
                articles = translator.wikipedia_search.stats.get('quality_articles', 0) if translator.wikipedia_search else 0
                # Handle both string estimates like '400k+' and integers
                if isinstance(articles, str):
                    articles_display = articles
                else:
                    try:
                        articles_display = f"{int(articles):,}"
                    except (ValueError, TypeError):
                        articles_display = "0"
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "available",
                    "articles": articles_display
                }).encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "unavailable"}).encode('utf-8'))

        elif self.path == '/api/warming_status':
            try:
                warming_status = {"phase": 0, "phase_name": "Not available", "progress": 0, "complete": False, "cache_ready": False}

                if hasattr(translator, 'wikipedia_search') and translator.wikipedia_search:
                    warming_status = translator.wikipedia_search.get_warming_status()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps(warming_status).encode('utf-8'))

            except Exception as e:
                print(f"Error getting warming status: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))

        elif self.path == '/api/status/context':
            # Get context usage stats
            try:
                stats = translator.get_context_stats() if translator else {'available': False}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode('utf-8'))
            except Exception as e:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps({'available': False, 'error': str(e)}).encode('utf-8'))

        elif self.path == '/api/status/memory':
            # Get live RAM usage for benchmarking
            try:
                memory_stats = get_memory_usage()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps(memory_stats).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        elif self.path == '/api/status/all':
            # Batched status endpoint — returns all statuses in one response
            # Cuts 4 round-trips to 1 for mobile polling
            result = {}  # _ollama_last_restart_attempt declared global above

            # Ollama status (lightweight GET check)
            try:
                resp = requests.get('http://localhost:11434', timeout=2)
                if resp.status_code == 200:
                    result['ollama'] = {"status": "connected"}
                else:
                    result['ollama'] = {"status": "disconnected"}
            except Exception:
                result['ollama'] = {"status": "disconnected"}

            # Auto-restart Ollama if disconnected
            if result['ollama']['status'] == 'disconnected':
                now = time.time()
                if now - _ollama_last_restart_attempt >= 30:
                    _ollama_last_restart_attempt = now
                    print("🔄 Ollama disconnected — attempting auto-restart...")
                    try:
                        success, error = _start_ollama_server()
                        if success:
                            print("[OK] Ollama auto-restart succeeded")
                            result['ollama'] = {"status": "connected"}
                        else:
                            print(f"[ERROR] Ollama auto-restart failed: {error}")
                    except Exception as e:
                        print(f"[ERROR] Ollama auto-restart error: {e}")

            # Wikipedia status
            if translator and hasattr(translator, 'wikipedia_available') and translator.wikipedia_available:
                articles = translator.wikipedia_search.stats.get('quality_articles', 0) if translator.wikipedia_search else 0
                if isinstance(articles, str):
                    articles_display = articles
                else:
                    try:
                        articles_display = f"{int(articles):,}"
                    except (ValueError, TypeError):
                        articles_display = "0"
                result['wikipedia'] = {"status": "available", "articles": articles_display}
            else:
                result['wikipedia'] = {"status": "unavailable"}

            # Context status
            try:
                stats = translator.get_context_stats() if translator else {'available': False}
                result['context'] = stats
            except Exception:
                result['context'] = {'available': False}

            # Memory status
            try:
                result['memory'] = get_memory_usage()
            except Exception:
                result['memory'] = {}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

        elif self.path == '/api/clear_history':
            # Clear conversation history
            try:
                if translator:
                    translator.clear_conversation_history()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode('utf-8'))

        elif self.path == '/api/status/helsinki':
            # Check Helsinki model status (legacy)
            status = "loaded" if (translator and hasattr(translator, 'helsinki_model') and
                               translator.helsinki_model is not None) else "not_loaded"
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode('utf-8'))

        elif self.path == '/api/status/database':
            # Check database status
            conn = None
            try:
                conn = translator.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM translations LIMIT 1")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "connected"}).encode('utf-8'))
            except Exception:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "disconnected"}).encode('utf-8'))
            finally:
                if conn:
                    conn.close()

        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()

            # Generate QR code SVG for LAN connect widget
            qr_svg_markup = ''
            qr_url = ''
            try:
                if lan_ip and server_port:
                    from qrcodegen import QrCode
                    qr_url = f'http://{lan_ip}:{server_port}'
                    qr = QrCode.encode_text(qr_url, QrCode.Ecc.MEDIUM)
                    size = qr.get_size()
                    # Build SVG from QR matrix
                    parts = []
                    for y in range(size):
                        for x in range(size):
                            if qr.get_module(x, y):
                                parts.append(f'M{x},{y}h1v1h-1z')
                    path_d = ' '.join(parts)
                    border = 2
                    viewbox = f'{-border} {-border} {size + border * 2} {size + border * 2}'
                    qr_svg_markup = (
                        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" '
                        f'width="140" height="140" style="background:#fff;border-radius:6px;">'
                        f'<path d="{path_d}" fill="#000"/></svg>'
                    )
            except Exception:
                qr_svg_markup = ''
                qr_url = ''

            html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NATNA</title>
    <style>
        /* Modern NATNA UI - Clean Professional Design */
        /* Noto Sans Ethiopic: use local system font (no CDN needed for offline use) */

        /* === CSS Custom Properties (Design Tokens) === */
        :root {
            /* Colors - Dark Mode */
            --bg-primary: #0f0f0f;
            --bg-surface: #1a1a1e;
            --bg-elevated: #242428;
            --border-color: #2a2a2e;
            --border-hover: #3a3a3e;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --accent: #22c55e;
            --accent-hover: #16a34a;
            --error: #ef4444;
            --warning: #f59e0b;
            --info: #3b82f6;

            /* Shadows */
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.3);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.4);

            /* Typography */
            --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            --font-ethiopic: 'Noto Sans Ethiopic', 'Nyala', 'Kefa', 'Geez Pro', sans-serif;
            --font-mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;

            /* Transitions */
            --transition-fast: 150ms ease;
            --transition-normal: 200ms ease;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        /* Loading Screen Styles - Clean & Minimal */
        .loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-primary);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: var(--font-ethiopic);
        }

        .natna-loading {
            font-size: 5rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.15em;
            position: relative;
            display: inline-block;
            opacity: 0;
            animation: fadeInUp 0.6s ease forwards;
            font-family: var(--font-ethiopic);
            z-index: 20;
        }

        .letter {
            display: inline-block;
            position: relative;
        }

        .loading-text {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 1.5rem;
            text-align: center;
            font-family: var(--font-sans);
            opacity: 0;
            animation: fadeIn 0.4s ease 0.3s forwards;
        }

        .skip-button {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 10px 24px;
            font-size: 0.9rem;
            border-radius: 6px;
            margin-top: 1.5rem;
            cursor: pointer;
            transition: all var(--transition-normal);
            font-family: var(--font-sans);
        }

        .skip-button:hover {
            background: var(--border-color);
            border-color: var(--border-hover);
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Terminal Styles */
        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-sans);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        /* Main container for terminal + research panel */
        .main-container {
            width: 95%;
            max-width: 1400px;
            height: 85%;
            display: flex;
            gap: 16px;
        }
        .terminal {
            flex: 1;
            min-width: 0;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-lg);
            overflow: hidden; /* Contain scroll to .output area */
        }
        /* Research Panel Styles */
        .research-panel {
            width: 350px;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-lg);
        }
        .research-header {
            background: var(--bg-elevated);
            color: var(--text-primary);
            padding: 14px 16px;
            font-size: 14px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 8px;
            border-radius: 12px 12px 0 0;
        }
        .research-content {
            flex: 1;
            padding: 12px;
            overflow-y: auto;
            font-size: 13px;
        }
        .more-articles-btn {
            background: var(--accent);
            border: none;
            color: #fff;
            padding: 6px 12px;
            margin: 8px auto 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            width: 50%;
            display: block;
            transition: all var(--transition-fast);
        }
        .search-wikipedia-btn {
            background: var(--info);
            border: none;
            color: #fff;
            padding: 6px 12px;
            margin: 0 6px 0 0;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            width: auto;
            display: inline-block;
            transition: all var(--transition-fast);
        }
        .more-articles-btn:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }
        .more-articles-btn:disabled {
            background: var(--bg-elevated);
            color: var(--text-muted);
            cursor: not-allowed;
        }
        .source-card {
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all var(--transition-normal);
        }
        .source-card:hover {
            background: var(--border-color);
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .source-title {
            color: var(--accent);
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 14px;
        }
        .source-summary {
            color: var(--text-secondary);
            font-size: 12px;
            line-height: 1.5;
        }
        .source-meta {
            color: var(--text-muted);
            font-size: 11px;
            margin-top: 8px;
        }
        .no-sources {
            color: var(--text-muted);
            text-align: center;
            padding: 20px;
            font-style: italic;
        }
        /* Article Modal */
        .article-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 2000;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(4px);
        }
        .article-modal.active {
            display: flex;
        }
        .article-container {
            width: 80%;
            max-width: 900px;
            height: 85%;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: var(--shadow-lg);
        }
        .article-header {
            background: var(--bg-elevated);
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .article-title {
            color: var(--text-primary);
            font-size: 18px;
            font-weight: 600;
        }
        .article-buttons {
            display: flex;
            gap: 10px;
        }
        .article-learn-more {
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all var(--transition-fast);
        }
        .article-learn-more:hover {
            background: var(--accent-hover);
        }
        .article-learn-more:disabled {
            background: var(--bg-elevated);
            color: var(--text-muted);
            cursor: not-allowed;
        }
        .article-close {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all var(--transition-fast);
        }
        .article-close:hover {
            background: var(--error);
            border-color: var(--error);
        }
        .article-body {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            color: var(--text-primary);
            font-size: 15px;
            line-height: 1.7;
            white-space: pre-wrap;
        }
        .article-meta {
            background: var(--bg-elevated);
            padding: 12px 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 12px;
        }
        /* Admin Password Modal */
        .admin-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 3000;
            justify-content: center;
            align-items: center;
            backdrop-filter: blur(4px);
        }
        .admin-modal.active {
            display: flex;
        }
        .admin-modal-content {
            width: 90%;
            max-width: 340px;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-lg);
        }
        .admin-modal-header {
            background: var(--bg-elevated);
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
            font-size: 16px;
            font-weight: 600;
            text-align: center;
        }
        .admin-modal-body {
            padding: 20px;
        }
        .admin-modal-body p {
            color: var(--text-secondary);
            font-size: 14px;
            margin-bottom: 12px;
        }
        .admin-password-input {
            width: 100%;
            padding: 12px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 16px;
            box-sizing: border-box;
        }
        .admin-password-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .admin-modal-error {
            color: var(--error);
            font-size: 13px;
            margin-top: 8px;
            min-height: 18px;
        }
        .admin-modal-buttons {
            display: flex;
            border-top: 1px solid var(--border-color);
        }
        .admin-btn {
            flex: 1;
            padding: 14px;
            border: none;
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        .admin-btn.cancel {
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border-right: 1px solid var(--border-color);
        }
        .admin-btn.cancel:hover {
            background: var(--bg-surface);
        }
        .admin-btn.confirm {
            background: var(--accent);
            color: white;
        }
        .admin-btn.confirm:hover {
            background: var(--accent-hover);
        }
        .header {
            background: var(--bg-elevated);
            color: var(--text-primary);
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            position: relative;
            text-align: center;
            border-radius: 12px 12px 0 0;
        }
        .output {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .input-line {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            gap: 8px;
            background: var(--bg-primary);
            border-top: 1px solid var(--border-color);
            border-radius: 0 0 12px 12px;
        }

        /* Sources button inside input - hidden on desktop */
        .input-sources-btn {
            display: none;
        }
        /* Send button - hidden on desktop */
        .send-btn {
            display: none;
        }
        /* Sources toggle in status bar - hidden on desktop (shown on mobile) */
        .sources-toggle {
            display: none;
        }
        .prompt { margin-right: 8px; color: var(--accent); font-weight: 500; }
        .input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 14px;
            outline: none;
        }
        .message { margin-bottom: 12px; }
        .user { color: var(--accent); }
        .ai-en { color: var(--text-primary); }
        .ai-ti { color: #93c5fd; }
        .error { color: var(--error); }

        /* Visual indicator for Enhanced Answer and Learn More content */
        .content-indicator {
            display: inline-block;
            background: #16a34a;
            color: #fff;
            padding: 4px 12px;
            border-radius: 16px;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 8px;
        }
        .content-indicator.enhanced {
            background: #16a34a;
        }
        .content-indicator.learn-more {
            background: #1d4ed8;
        }
        .enhanced-content {
            border-left: 3px solid var(--accent);
            padding-left: 12px;
            margin-left: 4px;
        }
        .cursor { opacity: 1; }

        .main-container {
            opacity: 0;
        }

        /* Domain Selector Styles */
        .domain-selector {
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 16px;
        }

        .selectors-container {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }

        .domain-selector-dropdown {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .domain-selector-dropdown label {
            font-size: 13px;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .domain-select {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            min-width: 180px;
            transition: all var(--transition-fast);
        }

        .domain-select:hover {
            background: var(--bg-elevated);
            border-color: var(--border-hover);
        }

        .domain-select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
        }
        /* Model Selector Styles */
        .model-selector {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .model-selector label {
            color: var(--text-muted);
            font-size: 12px;
        }
        .model-select {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        .model-select:hover {
            border-color: var(--border-hover);
        }
        .model-select:focus {
            outline: none;
            border-color: var(--accent);
        }

        /* Theme Toggle Button */
        .theme-toggle {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all var(--transition-fast);
        }
        .theme-toggle:hover {
            background: var(--bg-elevated);
            border-color: var(--border-hover);
            color: var(--text-primary);
        }

        /* Language Toggle Button - Ethiopian flag colors (yellow/red) */
        .lang-toggle {
            background: #fbbf24;
            border: none;
            color: #dc2626;
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all var(--transition-fast);
        }
        .lang-toggle:hover {
            background: #f59e0b;
        }
        .lang-toggle.english-only {
            background: #3b82f6;
            color: #fff;
        }

        /* Toggle buttons container - horizontal on desktop */
        .toggle-buttons {
            display: flex;
            gap: 8px;
        }

        /* User Message Styles */
        .user-message {
            background: rgba(34, 197, 94, 0.1);
            border-left: 3px solid var(--accent);
            padding: 10px 14px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            color: var(--accent);
            font-weight: 500;
        }

        /* AI Status Message Styles (Teaching, Expanding, Processing) */
        .ai-status {
            background: rgba(147, 197, 253, 0.15);
            border-left: 3px solid #93c5fd;
            padding: 10px 14px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            color: #93c5fd;
            font-weight: 500;
        }

        /* Fix text formatting and spacing */
        .message {
            white-space: pre-wrap;
            word-spacing: normal;
            line-height: 1.6;
        }

        /* Ensure italic elements preserve spacing */
        em, i, .italic {
            font-style: italic;
        }

        /* Fix for text that may have HTML entities or spacing issues */
        .message em,
        .message i {
            /* padding: 0 1px; - REMOVED to fix italic word spacing */
        }

        /* Math display and error handling styles */
        .math-error {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 3px solid var(--error);
            padding: 8px;
            margin: 4px 0;
            font-family: var(--font-mono);
            color: #fca5a5;
        }

        code {
            background-color: var(--bg-elevated);
            color: var(--warning);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 0.9em;
        }

        /* KaTeX rendering improvements */
        .katex {
            color: inherit !important;
            font-size: 1em !important;
        }

        .katex-display {
            margin: 8px 0 !important;
            text-align: center !important;
        }

        /* Light Theme Styles */
        body.light-theme {
            --bg-primary: #fafafa;
            --bg-surface: #ffffff;
            --bg-elevated: #f4f4f5;
            --border-color: #e4e4e7;
            --border-hover: #d4d4d8;
            --text-primary: #18181b;
            --text-secondary: #52525b;
            --text-muted: #a1a1aa;
            --accent: #16a34a;
            --accent-hover: #15803d;
            background: var(--bg-primary);
        }
        .light-theme .terminal {
            background: var(--bg-surface);
            border-color: var(--border-color);
            box-shadow: var(--shadow-lg);
        }
        .light-theme .header {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border-bottom-color: var(--border-color);
        }
        .light-theme .output {
            color: var(--text-primary);
        }
        .light-theme .input-line {
            background: var(--bg-elevated);
            border-top-color: var(--border-color);
        }
        .light-theme .prompt {
            color: var(--accent);
        }
        .light-theme .input {
            color: var(--text-primary);
        }
        .light-theme .user {
            color: var(--accent);
        }
        .light-theme .ai-en {
            color: var(--text-primary);
        }
        .light-theme .ai-ti {
            color: #1d4ed8;
        }
        .light-theme .error {
            color: #dc2626;
        }
        .light-theme .enhanced-content {
            border-left-color: #16a34a;
        }
        .light-theme .domain-selector {
            background: var(--bg-elevated);
            border-bottom-color: var(--border-color);
        }
        .light-theme .domain-selector-dropdown label {
            color: var(--text-secondary);
        }
        .light-theme .domain-select {
            background: var(--bg-surface);
            border-color: var(--border-color);
            color: var(--text-primary);
        }
        .light-theme .domain-select:hover {
            background: var(--bg-elevated);
            border-color: var(--border-hover);
        }
        .light-theme .domain-select:focus {
            border-color: var(--accent);
        }
        .light-theme .model-selector label {
            color: var(--text-muted);
        }
        .light-theme .model-select {
            background: var(--bg-surface);
            border-color: var(--border-color);
            color: var(--text-primary);
        }
        .light-theme .research-panel {
            background: var(--bg-surface);
            border-color: var(--border-color);
            box-shadow: var(--shadow-lg);
        }
        .light-theme .research-header {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border-bottom-color: var(--border-color);
        }
        .light-theme .source-card {
            background: var(--bg-elevated);
            border-color: var(--border-color);
        }
        .light-theme .source-card:hover {
            background: #f0fdf4;
            border-color: var(--accent);
        }
        .light-theme .source-title {
            color: var(--accent);
        }
        .light-theme .source-summary {
            color: var(--text-secondary);
        }
        .light-theme .source-meta {
            color: var(--text-muted);
        }
        .light-theme .no-sources {
            color: var(--text-muted);
        }
        .light-theme .article-modal {
            background: rgba(0, 0, 0, 0.5);
        }
        .light-theme .article-container {
            background: var(--bg-surface);
            border-color: var(--border-color);
        }
        .light-theme .article-header {
            background: var(--bg-elevated);
            border-bottom-color: var(--border-color);
        }
        .light-theme .article-title {
            color: var(--text-primary);
        }
        .light-theme .article-body {
            color: var(--text-primary);
        }
        .light-theme .article-meta {
            background: var(--bg-elevated);
            border-top-color: var(--border-color);
            color: var(--text-secondary);
        }
        .light-theme .theme-toggle {
            background: var(--bg-surface);
            border-color: var(--border-color);
            color: var(--text-secondary);
        }
        .light-theme .theme-toggle:hover {
            background: var(--bg-elevated);
        }
        .light-theme .lang-toggle {
            background: #fbbf24;
            color: #dc2626;
        }
        .light-theme .lang-toggle:hover {
            background: #f59e0b;
        }
        .light-theme .lang-toggle.english-only {
            background: #dc2626;
            color: #fbbf24;
        }
        .light-theme .user-message {
            background: #f0fdf4;
            border-left-color: var(--accent);
            color: var(--accent);
        }
        .light-theme .ai-status {
            background: #eff6ff;
            border-left-color: #3b82f6;
            color: #2563eb;
        }
        /* Status bar light theme */
        .light-theme div[style*="background: #222"] {
            background: var(--bg-elevated) !important;
            border-top-color: var(--border-color) !important;
        }
        .light-theme #clearHistoryBtn {
            background: var(--bg-surface) !important;
            border-color: var(--border-color) !important;
            color: var(--text-secondary) !important;
        }

        /* Fix light mode research panel text colors */
        .light-theme .research-panel {
            color: var(--text-primary) !important;
        }
        .light-theme .research-panel * {
            color: inherit !important;
        }
        .light-theme .source-card {
            color: var(--text-primary) !important;
        }
        .light-theme .source-title {
            color: var(--accent) !important;
        }
        .light-theme .source-summary {
            color: var(--text-secondary) !important;
        }
        /* Override any bright green text in light mode */
        .light-theme [style*="color: #0f0"],
        .light-theme [style*="color: #00ff00"],
        .light-theme [style*="color: #32cd32"],
        .light-theme [style*="color: #90EE90"] {
            color: var(--accent) !important;
        }

        /* Status bar light mode colors */
        .light-theme #ollama-status {
            color: var(--accent) !important;
        }
        .light-theme #wiki-status {
            color: #1d4ed8 !important;
        }
        .light-theme [style*="color: #ff6b6b"] {
            color: #dc2626 !important;
        }
        .light-theme [style*="color: #aaa"] {
            color: var(--text-muted) !important;
        }
        .light-theme [style*="color: #666"] {
            color: var(--text-secondary) !important;
        }

        /* === MOBILE RESPONSIVE === */

        /* Base rules - hidden on desktop */
        .research-drawer-toggle {
            display: none;
        }
        .drawer-backdrop {
            display: none;
        }

        /* Mobile breakpoint */
        @media (max-width: 768px) {
            /* Lock body - prevent document scroll, only .output scrolls */
            body {
                height: 100vh;
                height: 100dvh; /* Dynamic viewport height - works better on mobile */
                overflow: hidden;
                position: fixed;
                width: 100%;
                overscroll-behavior: none;
                touch-action: manipulation;
            }

            /* Layout: full width, stack vertically, contain scroll */
            .main-container {
                width: 100% !important;
                height: 100vh !important;
                height: 100dvh !important;
                max-width: 100% !important;
                flex-direction: column !important;
                gap: 0 !important;
                border-radius: 0;
                overflow: hidden;
            }

            .terminal {
                border-radius: 0;
                border-left: none;
                border-right: none;
                height: 100%;
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .header {
                border-radius: 0;
                flex-shrink: 0;
            }

            .mobile-picker-bar {
                flex-shrink: 0;
            }

            .input-line {
                border-radius: 0;
                flex-shrink: 0;
                padding: 8px 12px;
                gap: 8px;
                background: var(--bg-elevated);
                border-top: 1px solid var(--border-color);
            }

            /* Input field - flex to fill space, with text wrapping */
            .input-line .input {
                flex: 1;
                min-width: 0;
                padding-right: 8px;
            }

            /* Sources button - HIDDEN on mobile input, moved to status bar */
            .input-sources-btn {
                display: none;
            }

            /* Send button inside input line - mobile only */
            .send-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                background: var(--accent);
                color: #fff;
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                font-size: 18px;
                cursor: pointer;
                margin-left: 8px;
            }

            .send-btn:active {
                background: var(--accent-hover);
                transform: scale(0.95);
            }

            /* Sources toggle in status bar - button next to theme */
            .sources-toggle {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: var(--accent);
                border: none;
                color: #fff;
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 600;
            }

            .sources-toggle:active {
                opacity: 0.8;
            }

            /* Blinking animation to attract attention after query */
            @keyframes sourceBlink {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(1.1); }
            }

            .sources-toggle.blinking {
                animation: sourceBlink 0.7s ease-in-out infinite;
                box-shadow: 0 0 10px var(--accent);
            }

            /* Output area - allow shrink, be the flex filler */
            .output {
                flex: 1 1 0;
                min-height: 0;
                overflow-y: auto;
            }

            /* Status bar - prevent flex shrink, keep visible */
            .status-bar {
                flex-shrink: 0;
            }

            /* Language and theme buttons in a row */
            .toggle-buttons {
                display: flex;
                flex-direction: row;
                gap: 6px;
            }

            /* Research panel → slide-out drawer */
            .research-panel {
                position: fixed;
                top: 0;
                right: -100%;
                width: 85%;
                max-width: 350px;
                height: 100%;
                z-index: 900;
                transition: right 0.3s ease;
                border-radius: 0;
                border: none;
                border-left: 1px solid var(--border-color);
                box-shadow: var(--shadow-lg);
            }

            .research-panel.drawer-open {
                right: 0;
            }

            .research-header {
                border-radius: 0;
            }

            /* Backdrop overlay */
            .drawer-backdrop.active {
                display: block;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 899;
            }

            /* Old floating toggle button - now hidden, replaced by in-line button */
            .research-drawer-toggle {
                display: none;
            }

            /* Loading screen */
            .natna-loading {
                font-size: 3.5rem !important;
            }

            /* Hide desktop dropdowns on mobile, show picker buttons */
            .domain-selector {
                display: none !important;
            }

            .mobile-picker-bar {
                display: flex !important;
                position: sticky !important;
                top: 0 !important;
                z-index: 100 !important;
            }

            /* Mobile picker drawer */
            .mobile-picker-drawer {
                position: fixed;
                top: 0;
                left: -100%;
                width: 75%;
                max-width: 280px;
                height: 100%;
                z-index: 910;
                background: var(--bg-surface);
                border-right: 1px solid var(--border-color);
                transition: left 0.3s ease;
                box-shadow: var(--shadow-lg);
                display: flex;
                flex-direction: column;
                overflow-y: auto;
            }
            .mobile-picker-drawer.drawer-open {
                left: 0;
            }
            .mobile-picker-drawer .picker-header {
                padding: 16px;
                font-size: 14px;
                font-weight: 600;
                color: var(--text-primary);
                border-bottom: 1px solid var(--border-color);
                font-family: var(--font-sans);
            }
            .mobile-picker-drawer .picker-option {
                padding: 14px 16px;
                color: var(--text-secondary);
                font-size: 15px;
                border-bottom: 1px solid var(--border-color);
                cursor: pointer;
                transition: background var(--transition-fast);
            }
            .mobile-picker-drawer .picker-option:active {
                background: var(--bg-elevated);
            }
            .mobile-picker-drawer .picker-option.selected {
                color: var(--accent);
                background: rgba(34, 197, 94, 0.1);
            }
            .mobile-picker-drawer .picker-group-label {
                padding: 12px 16px 6px;
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 500;
            }

            /* Light theme overrides for picker drawers */
            .light-theme .mobile-picker-drawer {
                background: var(--bg-surface);
                border-right-color: var(--border-color);
                box-shadow: var(--shadow-lg);
            }
            .light-theme .mobile-picker-drawer .picker-header {
                color: var(--text-primary);
                border-bottom-color: var(--border-color);
            }
            .light-theme .mobile-picker-drawer .picker-option {
                color: var(--text-secondary);
                border-bottom-color: var(--border-color);
            }
            .light-theme .mobile-picker-drawer .picker-option:active {
                background: var(--bg-elevated);
            }
            .light-theme .mobile-picker-drawer .picker-option.selected {
                color: var(--accent);
                background: rgba(34, 197, 94, 0.1);
            }
            .light-theme .mobile-picker-drawer .picker-group-label {
                color: var(--text-muted);
            }

            /* Inputs - touch friendly, no iOS zoom */
            .input {
                font-size: 16px !important;
                min-height: 44px !important;
            }

            /* Article modal - full screen */
            .article-container {
                width: 100% !important;
                max-width: 100% !important;
                height: 100% !important;
                border-radius: 0 !important;
            }

            /* Article header - stack title above buttons on mobile */
            .article-header {
                flex-direction: column;
                align-items: flex-start !important;
                gap: 10px;
                padding: 14px 16px !important;
            }
            .article-title {
                font-size: 16px !important;
            }
            .article-buttons {
                width: 100%;
                gap: 8px !important;
            }
            .article-learn-more,
            .article-close {
                font-size: 13px !important;
                padding: 8px 14px !important;
            }

            /* Status bar - stack vertically */
            .terminal > div:last-child {
                flex-direction: column !important;
                align-items: flex-start !important;
                gap: 6px !important;
                padding: 8px 12px !important;
            }

            /* Hide pipe separators on mobile */
            .terminal > div:last-child span[style*="border-hover"] {
                display: none !important;
            }

            /* Status items wrap */
            .terminal > div:last-child > div:first-child {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            /* Action buttons row */
            .terminal > div:last-child > div:last-child {
                flex-wrap: wrap !important;
                gap: 8px !important;
            }

            /* Hide non-essential status items on mobile */
            #context-status,
            #last-query-time {
                display: none !important;
            }

            /* Compact status bar buttons on mobile */
            .terminal > div:last-child button {
                padding: 4px 8px !important;
                font-size: 11px !important;
            }

            /* Footer adjust - keep right side, move below buttons */
            footer {
                bottom: 4px !important;
                right: 12px !important;
            }

            /* Header text */
            .header {
                font-size: 14px;
                padding: 10px 14px !important;
            }
        }

        /* Light theme + mobile combo */
        @media (max-width: 768px) {
            .light-theme .research-panel {
                border-left-color: var(--border-color);
                box-shadow: var(--shadow-lg);
            }

            .light-theme .drawer-backdrop.active {
                background: rgba(0, 0, 0, 0.3);
            }

            .light-theme .input-sources-btn {
                background: var(--accent);
            }

            .light-theme .send-btn {
                background: var(--accent);
            }

            .light-theme .sources-toggle {
                background: var(--accent);
            }

            .light-theme .input-line {
                background: var(--bg-surface);
                border-top-color: var(--border-color);
            }

            .light-theme .terminal {
                border-left: none;
                border-right: none;
            }
        }

        /* Mobile picker bar - two small buttons, hidden on desktop */
        .mobile-picker-bar {
            display: none;
            gap: 8px;
            padding: 8px 12px;
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border-color);
        }
        .mobile-picker-bar button {
            flex: 1;
            background: var(--bg-surface);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 13px;
            cursor: pointer;
            text-align: left;
            font-family: var(--font-sans);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all var(--transition-fast);
        }
        .mobile-picker-bar button:active {
            background: var(--bg-elevated);
        }
        .mobile-picker-bar .picker-label {
            color: var(--text-muted);
            font-size: 10px;
            display: block;
            margin-bottom: 2px;
            font-weight: 500;
        }
        .light-theme .mobile-picker-bar {
            background: var(--bg-elevated);
            border-bottom-color: var(--border-color);
        }
        .light-theme .mobile-picker-bar button {
            background: var(--bg-surface);
            color: var(--text-primary);
            border-color: var(--border-color);
        }
        .light-theme .mobile-picker-bar button:active {
            background: var(--bg-elevated);
        }
        .light-theme .mobile-picker-bar .picker-label {
            color: var(--text-muted);
        }

        /* Mobile picker drawers - hidden by default (shown via mobile media query) */
        .mobile-picker-drawer {
            display: none;
        }

        /* QR Connect Widget */
        .qr-connect-btn {
            position: fixed;
            bottom: 8px;
            left: 12px;
            z-index: 100;
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            font-size: 12px;
            font-family: var(--font-sans);
            border-radius: 6px;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        .qr-connect-btn:hover {
            background: var(--border-color);
            color: var(--text-primary);
        }
        .qr-connect-popup {
            position: fixed;
            bottom: 42px;
            left: 12px;
            z-index: 101;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            display: none;
            box-shadow: var(--shadow-lg);
        }
        .qr-connect-popup.visible { display: block; }
        .qr-connect-popup .qr-label {
            color: var(--text-secondary);
            font-size: 12px;
            font-family: var(--font-sans);
            margin-top: 10px;
        }
        .qr-connect-popup .qr-hint {
            color: var(--text-muted);
            font-size: 11px;
            margin-top: 4px;
        }

        .light-theme .qr-connect-btn {
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border-color: var(--border-color);
        }
        .light-theme .qr-connect-btn:hover {
            background: var(--border-color);
            color: var(--text-primary);
        }
        .light-theme .qr-connect-popup {
            background: var(--bg-surface);
            border-color: var(--border-color);
            box-shadow: var(--shadow-lg);
        }
        .light-theme .qr-connect-popup .qr-label { color: var(--text-secondary); }
        .light-theme .qr-connect-popup .qr-hint { color: var(--text-muted); }

        @media (max-width: 768px) {
            .qr-connect-btn, .qr-connect-popup { display: none !important; }
        }

        /* Status indicators - subtle dots */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-dot.connected { background: var(--accent); }
        .status-dot.disconnected { background: var(--error); }
        .status-dot.warning { background: var(--warning); }

        /* === Knowledge Cards - Quick Reference === */
        .knowledge-card {
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
            font-size: 13px;
        }
        .knowledge-card-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }
        .knowledge-card-title {
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
        }
        .knowledge-card-badge {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 500;
        }
        .knowledge-card-section {
            margin-bottom: 10px;
        }
        .knowledge-card-section:last-child {
            margin-bottom: 0;
        }
        .knowledge-card-section-title {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .knowledge-card-section ul {
            margin: 0;
            padding-left: 18px;
            color: var(--text-secondary);
        }
        .knowledge-card-section li {
            margin-bottom: 3px;
            line-height: 1.4;
        }
        .knowledge-card-section p {
            margin: 0;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Medical card - red accent for warnings */
        .medical-card {
            border-left: 4px solid var(--error);
        }
        .medical-card .knowledge-card-badge {
            background: rgba(239, 68, 68, 0.15);
            color: var(--error);
        }
        .medical-card .card-section-warning .knowledge-card-section-title {
            color: var(--error);
        }
        .medical-card .card-section-care .knowledge-card-section-title {
            color: var(--accent);
        }
        .medical-card .card-section-causes .knowledge-card-section-title {
            color: var(--info);
        }

        /* Agricultural card - green accent */
        .agricultural-card {
            border-left: 4px solid var(--accent);
        }
        .agricultural-card .knowledge-card-badge {
            background: rgba(34, 197, 94, 0.15);
            color: var(--accent);
        }
        .agricultural-card .tigrinya-name {
            font-family: var(--font-ethiopic);
            color: var(--text-muted);
            font-size: 13px;
        }
        .agricultural-card .knowledge-card-section-title {
            color: var(--accent);
        }

        /* Mental Health card - purple accent (calming, supportive) */
        .mental-health-card {
            border-left: 4px solid #8b5cf6;
        }
        .mental-health-card .knowledge-card-badge {
            background: rgba(139, 92, 246, 0.15);
            color: #8b5cf6;
        }
        .mental-health-card .tigrinya-name {
            font-family: var(--font-ethiopic);
            color: var(--text-muted);
            font-size: 13px;
        }
        .mental-health-card .knowledge-card-section-title {
            color: #8b5cf6;
        }
        .mental-health-card .card-section-danger .knowledge-card-section-title {
            color: var(--error);
        }
        .mental-health-card .card-section-support .knowledge-card-section-title {
            color: #8b5cf6;
        }
        .mental-health-card .card-section-coping .knowledge-card-section-title {
            color: var(--accent);
        }
        /* Crisis card - more prominent warning styling */
        .mental-health-card.crisis-card {
            border-left: 4px solid var(--error);
            background: rgba(239, 68, 68, 0.05);
        }
        .mental-health-card.crisis-card .knowledge-card-badge {
            background: rgba(239, 68, 68, 0.15);
            color: var(--error);
        }
        .mental-health-card.crisis-card .card-section-action .knowledge-card-section-title {
            color: var(--error);
        }
        .mental-health-card.crisis-card .card-section-donot .knowledge-card-section-title {
            color: var(--warning);
        }

        /* Light theme overrides for knowledge cards */
        .light-theme .knowledge-card {
            background: var(--bg-elevated);
            border-color: var(--border-color);
        }
        .light-theme .knowledge-card-header {
            border-bottom-color: var(--border-color);
        }
        .light-theme .knowledge-card-title {
            color: var(--text-primary);
        }
        .light-theme .medical-card {
            border-left-color: #dc2626;
        }
        .light-theme .medical-card .knowledge-card-badge {
            background: rgba(220, 38, 38, 0.1);
            color: #dc2626;
        }
        .light-theme .agricultural-card {
            border-left-color: #16a34a;
        }
        .light-theme .agricultural-card .knowledge-card-badge {
            background: rgba(22, 163, 74, 0.1);
            color: #16a34a;
        }
        .light-theme .mental-health-card {
            border-left-color: #7c3aed;
        }
        .light-theme .mental-health-card .knowledge-card-badge {
            background: rgba(124, 58, 237, 0.1);
            color: #7c3aed;
        }
        .light-theme .mental-health-card.crisis-card {
            border-left-color: #dc2626;
            background: rgba(220, 38, 38, 0.03);
        }
        .light-theme .mental-health-card.crisis-card .knowledge-card-badge {
            background: rgba(220, 38, 38, 0.1);
            color: #dc2626;
        }

        /* Mobile responsiveness for knowledge cards */
        @media (max-width: 768px) {
            .knowledge-card {
                padding: 12px;
                font-size: 12px;
            }
            .knowledge-card-title {
                font-size: 13px;
            }
            .knowledge-card-section ul {
                padding-left: 16px;
            }
        }

        /* FAQ Message Styles */
        .faq-message {
            background: rgba(59, 130, 246, 0.08);
            border-left: 3px solid var(--info);
            padding: 16px 18px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            color: var(--text-primary);
            font-weight: 400;
            line-height: 1.7;
            white-space: normal;
        }
        .faq-message h2 {
            color: var(--info);
            font-size: 1.15em;
            margin: 0 0 10px 0;
            font-weight: 600;
        }
        .faq-message h3 {
            color: var(--accent);
            font-size: 0.95em;
            margin: 14px 0 6px 0;
            padding-top: 10px;
            border-top: 1px solid var(--border-color);
            font-weight: 600;
        }
        .faq-message h3:first-of-type {
            border-top: none;
            padding-top: 0;
            margin-top: 8px;
        }
        .faq-message p {
            margin: 4px 0;
            color: var(--text-secondary);
        }
        .faq-message ul {
            margin: 4px 0 4px 18px;
            padding: 0;
            color: var(--text-secondary);
        }
        .faq-message li {
            margin: 2px 0;
        }
        .faq-message code {
            background: var(--bg-elevated);
            color: var(--warning);
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .faq-message .faq-footer {
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.9em;
        }
        .light-theme .faq-message {
            background: rgba(59, 130, 246, 0.06);
            border-left-color: #2563eb;
            color: var(--text-primary);
        }
        .light-theme .faq-message h2 {
            color: #2563eb;
        }
        .light-theme .faq-message h3 {
            color: var(--accent);
            border-top-color: var(--border-color);
        }
        .light-theme .faq-message p,
        .light-theme .faq-message ul {
            color: var(--text-secondary);
        }

    </style>
    <!-- KaTeX for Mathematical Notation Rendering (Offline) -->
    <link rel="stylesheet" href="assets/katex/katex.min.css">
    <script src="assets/katex/katex.min.js"></script>
    <script src="assets/katex/auto-render.min.js"></script>
    <script>
// Embedded KaTeX processor - no external file needed
    </script>
    <script>
        // Simple KaTeX integration - external processor handles all math rendering

        // AI-Aware KaTeX Math Processor for Dynamic Content (2025 Best Practices)
        function normalizeAILatex(content) {
            if (!content) return content;

            console.log('KaTeX: Normalizing AI-generated LaTeX');

            // Handle inconsistent AI LaTeX output (1, 2, or 4 backslashes)
            let normalized = content;

            // Standardize display math delimiters
            // Python \\\\ -> JS \\  in regex = 1 literal backslash
            // We normalize varying backslash counts down to single-backslash delimiters
            normalized = normalized.replace(/\\\\\\\\\\\\\\\\\\[/g, '\\\\[');  // 4 backslashes+[ to \[
            normalized = normalized.replace(/\\\\\\\\\\[/g, '\\\\[');          // 2 backslashes+[ to \[
            normalized = normalized.replace(/\\\\\\\\\\\\\\\\\\]/g, '\\\\]');  // 4 backslashes+] to \]
            normalized = normalized.replace(/\\\\\\\\\\]/g, '\\\\]');          // 2 backslashes+] to \]

            // Standardize inline math delimiters
            normalized = normalized.replace(/\\\\\\\\\\\\\\\\\\(/g, '\\\\(');  // 4 backslashes+( to \(
            normalized = normalized.replace(/\\\\\\\\\\(/g, '\\\\(');          // 2 backslashes+( to \(
            normalized = normalized.replace(/\\\\\\\\\\\\\\\\\\)/g, '\\\\)');  // 4 backslashes+) to \)
            normalized = normalized.replace(/\\\\\\\\\\)/g, '\\\\)');          // 2 backslashes+) to \)

            // Fix common AI LaTeX issues
            normalized = normalized.replace(/\\\\\\\\text\\{/g, '\\\\text{');  // Fix \\text{ to \text{
            normalized = normalized.replace(/\\\\\\\\frac\\{/g, '\\\\frac{');  // Fix \\frac{ to \frac{

            console.log('KaTeX: LaTeX normalization complete');
            return normalized;
        }

        function preprocessAIContent(content) {
            if (!content) return content;

            // Only process content that already contains math delimiters
            if (!content.includes('$$') && !content.includes('\\\\[') && !content.includes('\\\\(')) {
                return content;
            }

            console.log('KaTeX: Preprocessing mathematical content');
            return content; // No aggressive processing that breaks regular text
        }

        function safeTypesetMath(element) {
            if (!window.renderMathInElement || typeof window.renderMathInElement !== 'function') {
                console.warn('KaTeX: renderMathInElement not available, skipping typeset');
                return Promise.resolve();
            }

            return new Promise((resolve) => {
                try {
                    console.log('KaTeX: Starting AI-aware math rendering');

                    // Process AI content through comprehensive pipeline
                    const originalHTML = element.innerHTML;

                    // Step 1: Preprocess AI content for mathematical expressions
                    let processedHTML = preprocessAIContent(originalHTML);

                    // Step 2: Normalize LaTeX delimiters and syntax
                    processedHTML = normalizeAILatex(processedHTML);

                    if (originalHTML !== processedHTML) {
                        console.log('KaTeX: Applied AI content processing and LaTeX normalization');
                        element.innerHTML = processedHTML;
                    }

                    // Enhanced KaTeX configuration for AI content
                    renderMathInElement(element, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '\\\\[', right: '\\\\]', display: true},
                            {left: '\\\\(', right: '\\\\)', display: false},
                            {left: '$', right: '$', display: false},
                            // Additional delimiters for AI content
                            {left: '\\\\begin{equation}', right: '\\\\end{equation}', display: true},
                            {left: '\\\\begin{align}', right: '\\\\end{align}', display: true}
                        ],
                        throwOnError: false,
                        errorColor: '#cc0000',
                        strict: false,
                        trust: false,
                        // Enhanced options for dynamic content
                        macros: {
                            '\\\\RR': '\\\\mathbb{R}',
                            '\\\\NN': '\\\\mathbb{N}',
                            '\\\\ZZ': '\\\\mathbb{Z}',
                            '\\\\QQ': '\\\\mathbb{Q}',
                            '\\\\CC': '\\\\mathbb{C}'
                        }
                    });

                    console.log('KaTeX: Math rendering successful');
                    resolve();

                } catch (error) {
                    console.error('KaTeX: Rendering error:', error);
                    // Graceful fallback - content remains readable
                    resolve();
                }
            });
        }
    </script>
</head>
<body>
    <!-- Loading Screen -->
    <div class="loading-screen" id="loadingScreen">
        <div style="text-align: center;">
            <div class="natna-loading">
                <span class="letter">ና</span><span class="letter">ት</span><span class="letter">ና</span>
            </div>
            <div class="loading-text">Loading Tigray AI : Offline Internet System</div>
            <button class="skip-button" onclick="skipLoading()">Enter</button>
        </div>
    </div>

    <!-- Admin Password Modal (for mobile model changes) -->
    <div class="admin-modal" id="adminModal">
        <div class="admin-modal-content">
            <div class="admin-modal-header">Admin Access Required</div>
            <div class="admin-modal-body">
                <p>Enter admin password to change the AI model:</p>
                <input type="password" id="adminPassword" class="admin-password-input" placeholder="Password" autocomplete="off">
                <div class="admin-modal-error" id="adminError"></div>
            </div>
            <div class="admin-modal-buttons">
                <button class="admin-btn cancel" onclick="closeAdminModal()">Cancel</button>
                <button class="admin-btn confirm" onclick="verifyAdminPassword()">Confirm</button>
            </div>
        </div>
    </div>

    <!-- Article Modal -->
    <div class="article-modal" id="articleModal">
        <div class="article-container">
            <div class="article-header">
                <span class="article-title" id="articleTitle">Article Title</span>
                <div class="article-buttons">
                    <button class="article-learn-more" id="learnMoreBtn" onclick="learnMore()">Learn More</button>
                    <button class="article-close" onclick="closeArticle()">Close</button>
                </div>
            </div>
            <div class="article-body" id="articleBody">
                Loading article...
            </div>
            <div class="article-meta" id="articleMeta">
                Word count: 0
            </div>
        </div>
    </div>

    <!-- Main Container with Terminal + Research Panel -->
    <div class="main-container" id="mainContainer">
        <button class="research-drawer-toggle" id="researchDrawerToggle" onclick="toggleResearchDrawer()">Sources</button>
        <!-- Terminal Interface -->
        <div class="terminal" id="terminal">
            <div class="header">
                <span>NATNA AI Terminal</span>
            </div>

            <!-- Mobile Picker Buttons (visible only on mobile) -->
            <div class="mobile-picker-bar" id="mobilePickerBar">
                <button onclick="openPickerDrawer('domain')">
                    <span class="picker-label">Domain</span>
                    <span id="mobileDomainLabel">Education</span>
                </button>
                <button onclick="openPickerDrawer('model')">
                    <span class="picker-label">Model</span>
                    <span id="mobileModelLabel">Qwen 3 0.6B</span>
                </button>
            </div>

            <!-- Domain Selector Bar -->
            <div class="domain-selector">
                <!-- Domain and Model Selectors -->
                <div class="selectors-container">
                    <div class="domain-selector-dropdown">
                        <label>Domain:</label>
                        <select class="domain-select" id="domainSelect">
                            <option value="general">General / ሓፈሻዊ</option>
                            <option value="medical">Medical / ሕክምና</option>
                            <option value="education" selected>Education / ትምህርቲ</option>
                            <option value="college">College / ኮለጅ</option>
                            <option value="technical">Technical / ቴክኒክ</option>
                            <option value="agriculture">Agriculture / ሕርሻ</option>
                            <option value="programming">Programming / ፕሮግራሚንግ</option>
                        </select>
                    </div>
                    <div class="model-selector">
                        <label>AI Model:</label>
                        <select class="model-select" id="modelSelect">
                        <optgroup label="--- General (Uncensored) ---">
                            <option value="deepseek-r1-abliterated-4gb">4GB Uncensored</option>
                            <option value="deepseek-r1-uncensored-16gb">16GB Uncensored (advanced)</option>
                        </optgroup>
                        <optgroup label="--- Small (4GB RAM) ---">
                            <option value="smollm2:360m">SmolLM2 360M (726MB)</option>
                            <option value="qwen2.5:0.5b" selected>Qwen 2.5 0.5B (default)</option>
                            <option value="alibayram/smollm3">SmolLM3 3B (1.8GB)</option>
                            <option value="phi4-mini">Phi-4 Mini (2.5GB)</option>
                        </optgroup>
                        <optgroup label="--- Coding (8GB RAM) ---">
                            <option value="qwen2.5-coder:7b">Qwen Coder 7B (4.7GB)</option>
                            <option value="deepseek-coder:6.7b">DeepSeek Coder 6.7B (3.8GB)</option>
                            <option value="codegemma:7b">CodeGemma 7B (5GB)</option>
                        </optgroup>
                        <optgroup label="--- Coding (16GB RAM) ---">
                            <option value="qwen2.5-coder:14b">Qwen Coder 14B (9GB)</option>
                            <option value="deepseek-coder-v2:16b">DeepSeek Coder V2 (8.9GB)</option>
                            <option value="codellama:13b">CodeLlama 13B (7.4GB)</option>
                        </optgroup>
                        </select>
                    </div>
                </div>
            </div>

            <div class="output" id="output">NATNA — Offline AI, English + Tigrinya

Ask any question below. Answers draw from local AI models and 475,000+ Wikipedia articles.

Type /FAQ for a full guide on domains, models, bilingual mode, and more.

</div>
            <div class="input-line">
                <input type="text" class="input" id="input" placeholder="Ask anything..." autofocus>
                <button class="send-btn" id="sendBtn" onclick="submitQuery()" title="Send">&#10148;</button>
            </div>

            <!-- System Status Bar -->
            <div class="status-bar" style="background: var(--bg-elevated); border-top: 1px solid var(--border-color); padding: 10px 16px; font-size: 12px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <span id="ollama-status" style="color: var(--warning);"><span class="status-dot warning"></span>Server: Checking...</span>
                    <button id="start-server-btn" onclick="startOllamaServer()" style="display: none; background: var(--error); border: none; color: white; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 500;">Start Server</button>
                    <div id="model-progress" style="display: none; width: 100px;">
                        <div style="background: var(--border-color); border-radius: 4px; height: 4px;">
                            <div id="progress-bar" style="background: var(--info); height: 4px; border-radius: 4px; width: 0%; transition: width 0.3s;"></div>
                        </div>
                        <div id="progress-text" style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Loading...</div>
                    </div>
                    <span style="color: var(--border-hover);">|</span>
                    <span id="wiki-status" style="color: var(--text-secondary);"><span class="status-dot warning"></span>Wikipedia: Checking...</span>
                    <span style="color: var(--border-hover);">|</span>
                    <span id="context-status" style="color: var(--info);">Context: 0%</span>
                    <span style="color: var(--border-hover);">|</span>
                    <span id="memory-status" style="color: var(--warning);">RAM: Loading...</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span id="last-query-time" style="color: var(--text-muted);">Ready</span>
                    <button id="clearHistoryBtn" onclick="clearHistory()" style="background: var(--bg-surface); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; transition: all 150ms ease;">Clear</button>
                    <button id="expandAnswerBtn" onclick="expandAnswer()" style="background: var(--accent); border: none; color: #fff; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; display: none; font-weight: 500;" title="Enhance answer with Wikipedia context">Expand</button>
                    <div class="toggle-buttons">
                        <button class="lang-toggle" id="langToggle" onclick="toggleLanguage()" title="Toggle Tigrinya responses">Ti/En</button>
                        <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()">Dark</button>
                        <button class="sources-toggle" id="sourcesToggle" onclick="toggleResearchDrawer()" title="View Sources">Src</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mobile Drawer Backdrop -->
        <div class="drawer-backdrop" id="drawerBackdrop" onclick="closeResearchDrawer(); closePickerDrawers();"></div>

        <!-- Mobile Picker Drawers (slide from left) -->
        <div class="mobile-picker-drawer" id="domainPickerDrawer">
            <div class="picker-header">Select Domain</div>
            <div class="picker-option" data-value="general" onclick="pickDomain(this)">General / &#4627;&#4936;&#4667;&#4810;</div>
            <div class="picker-option" data-value="medical" onclick="pickDomain(this)">Medical / &#4629;&#4781;&#4637;&#4755;</div>
            <div class="picker-option selected" data-value="education" onclick="pickDomain(this)">Education / &#4725;&#4637;&#4613;&#4653;&#4722;</div>
            <div class="picker-option" data-value="college" onclick="pickDomain(this)">College / &#4782;&#4616;&#4869;</div>
            <div class="picker-option" data-value="technical" onclick="pickDomain(this)">Technical / &#4724;&#4781;&#4754;&#4781;</div>
            <div class="picker-option" data-value="agriculture" onclick="pickDomain(this)">Agriculture / &#4629;&#4653;&#4667;</div>
            <div class="picker-option" data-value="programming" onclick="pickDomain(this)">Programming / &#4947;&#4686;&#4877;&#4651;&#4636;&#4757;&#4877;</div>
        </div>
        <div class="mobile-picker-drawer" id="modelPickerDrawer">
            <div class="picker-header">Select AI Model</div>
            <div class="picker-group-label">General (Uncensored)</div>
            <div class="picker-option" data-value="deepseek-r1-abliterated-4gb" onclick="pickModel(this)">4GB Uncensored</div>
            <div class="picker-option" data-value="deepseek-r1-uncensored-16gb" onclick="pickModel(this)">16GB Uncensored (advanced)</div>
            <div class="picker-group-label">Small (4GB RAM)</div>
            <div class="picker-option" data-value="smollm2:360m" onclick="pickModel(this)">SmolLM2 360M (726MB)</div>
            <div class="picker-option selected" data-value="qwen2.5:0.5b" onclick="pickModel(this)">Qwen 2.5 0.5B (default)</div>
            <div class="picker-option" data-value="alibayram/smollm3" onclick="pickModel(this)">SmolLM3 3B (1.8GB)</div>
            <div class="picker-option" data-value="phi4-mini" onclick="pickModel(this)">Phi-4 Mini (2.5GB)</div>
            <div class="picker-group-label">Coding (8GB RAM)</div>
            <div class="picker-option" data-value="qwen2.5-coder:7b" onclick="pickModel(this)">Qwen Coder 7B (4.7GB)</div>
            <div class="picker-option" data-value="deepseek-coder:6.7b" onclick="pickModel(this)">DeepSeek Coder 6.7B (3.8GB)</div>
            <div class="picker-option" data-value="codegemma:7b" onclick="pickModel(this)">CodeGemma 7B (5GB)</div>
            <div class="picker-group-label">Coding (16GB RAM)</div>
            <div class="picker-option" data-value="qwen2.5-coder:14b" onclick="pickModel(this)">Qwen Coder 14B (9GB)</div>
            <div class="picker-option" data-value="deepseek-coder-v2:16b" onclick="pickModel(this)">DeepSeek Coder V2 (8.9GB)</div>
            <div class="picker-option" data-value="codellama:13b" onclick="pickModel(this)">CodeLlama 13B (7.4GB)</div>
        </div>

        <!-- Research Panel -->
        <div class="research-panel" id="researchPanel">
            <div class="research-header">
                Research Sources
            </div>
            <div class="research-content" id="researchContent">
                <div class="no-sources">
                    Ask a question to see relevant Wikipedia sources here.
                    <br><br>
                    Click any source to read the full article.
                </div>
            </div>
            <button class="more-articles-btn" id="moreArticlesBtn" onclick="loadMoreArticles()" style="display: none;">
                More Articles
            </button>
        </div>
    </div>

    <footer style="position: fixed; bottom: 8px; right: 12px; font-size: 10px; color: var(--text-muted);">
        powered by parabl
    </footer>

''' + (f'''
    <button class="qr-connect-btn" id="qrConnectBtn" onclick="toggleQrConnect()">&#x1F4F1; Connect</button>
    <div class="qr-connect-popup" id="qrConnectPopup">
        <div class="qr-image">{qr_svg_markup}</div>
        <div class="qr-label">{qr_url}</div>
        <div class="qr-hint">Scan with phone camera</div>
    </div>
''' if qr_svg_markup else '') + '''

    <script>
        const output = document.getElementById('output');
        const input = document.getElementById('input');
        const researchContent = document.getElementById('researchContent');
        const modelSelect = document.getElementById('modelSelect');
        const domainSelect = document.getElementById('domainSelect');
        let processing = false;
        let currentDomain = 'education';
        let currentModel = 'qwen2.5:0.5b';

        // Fetch with AbortController timeout — prevents hung requests from stacking
        function fetchWithTimeout(url, timeoutMs = 5000, options = {}) {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), timeoutMs);
            return fetch(url, { ...options, signal: controller.signal })
                .finally(() => clearTimeout(timeout));
        }

        // Model context limits (tokens)
        const MODEL_LIMITS = {
            'qwen2.5:0.5b': 8192,
            'smollm2:360m': 8192,
            'qwen3:0.6b': 8192,
            'alibayram/smollm3': 8192,
            'phi4-mini': 8192,
            'deepseek-r1-abliterated-4gb': 4096,
            'deepseek-r1-uncensored-16gb': 8192,
            'qwen2.5-coder:7b': 8192,
            'deepseek-coder:6.7b': 8192,
            'codegemma:7b': 8192,
            'qwen2.5-coder:14b': 16384,
            'deepseek-coder-v2:16b': 16384,
            'codellama:13b': 16384
        };

        // Wikipedia warming status tracking
        let warmingStatusInterval = null;
        let _warmingFailCount = 0;

        function checkWarmingStatus() {
            fetchWithTimeout('/api/warming_status', 5000)
                .then(response => response.json())
                .then(status => {
                    _warmingFailCount = 0;  // Reset on success
                    updateWarmingDisplay(status);

                    // Stop checking once complete
                    if (status.complete && warmingStatusInterval) {
                        clearInterval(warmingStatusInterval);
                        warmingStatusInterval = null;
                    }
                })
                .catch(err => {
                    _warmingFailCount++;
                    console.log('Warming status check failed (' + _warmingFailCount + '/3):', err);
                    // Stop zombie polling after 3 consecutive failures
                    if (_warmingFailCount >= 3 && warmingStatusInterval) {
                        clearInterval(warmingStatusInterval);
                        warmingStatusInterval = null;
                        console.log('Warming poll stopped after 3 failures');
                    }
                });
        }

        function updateWarmingDisplay(status) {
            const searchBtn = document.getElementById('searchWikipediaBtn');

            if (status.complete) {
                // Database is loaded and ready
                if (searchBtn) {
                    searchBtn.style.background = 'var(--accent)';
                    searchBtn.style.color = '#fff';
                    searchBtn.title = 'Wikipedia database loaded - search ready';
                    searchBtn.innerHTML = 'Wiki Ready';
                    // After 4 seconds, switch to normal search button
                    setTimeout(() => {
                        searchBtn.style.background = 'var(--info)';
                        searchBtn.style.color = '#fff';
                        searchBtn.title = 'Search Wikipedia for current query';
                        searchBtn.innerHTML = 'Search Wiki';
                    }, 4000);
                }

                // Show loaded message in research panel if still showing warming
                if (researchContent.innerHTML.includes('Wikipedia Database')) {
                    researchContent.innerHTML = `
                        <div class="no-sources">
                            <strong>Wikipedia Database Loaded</strong><br><br>
                            Wikipedia context will be included automatically with your queries.
                        </div>
                    `;
                }
                return;
            }

            // Show warming progress
            if (searchBtn) {
                searchBtn.style.background = 'var(--warning)';
                searchBtn.style.color = '#fff';
                searchBtn.title = `Loading Wikipedia: ${status.phase_name} (${status.progress}%)`;
                searchBtn.innerHTML = `Loading ${status.progress}%`;
            }

            // Show progress in research panel if empty or already showing warming
            if (researchContent.innerHTML.includes('Ask a question') || researchContent.innerHTML.includes('Wikipedia Database')) {
                researchContent.innerHTML = `
                    <div class="no-sources">
                        <strong>Wikipedia Database Loading</strong><br>
                        ${status.phase_name}<br>
                        <div style="background: var(--border-color); height: 8px; border-radius: 4px; margin: 8px 0;">
                            <div style="background: var(--warning); height: 100%; width: ${status.progress}%; border-radius: 4px; transition: width 0.3s;"></div>
                        </div>
                        ${status.progress}% complete
                    </div>
                `;
            }
        }

        // Start checking warming status immediately and every 2 seconds
        checkWarmingStatus();
        warmingStatusInterval = setInterval(checkWarmingStatus, 2000);

        // Model selector functionality
        modelSelect.addEventListener('change', function() {
            currentModel = this.value;
            const modelName = this.options[this.selectedIndex].text;
            const limit = MODEL_LIMITS[currentModel] || 2048;
            appendToOutput(`Switched to ${modelName} (${limit} token context)`, 'ai-en');
            // Refresh context status
            checkSystemStatus();
        });

        // Domain selector functionality
        domainSelect.addEventListener('change', function() {
            // Update current domain
            currentDomain = this.value;
            // Update placeholder text
            const placeholders = {
                'general': 'Ask your question...',
                'medical': 'Ask your health question...',
                'education': 'Ask anything...',
                'technical': 'Ask your technical question...',
                'agriculture': 'Ask your farming question...',
                'programming': 'Ask your coding question...'
            };
            input.placeholder = placeholders[currentDomain];

            // Auto-switch models based on domain
            if (currentDomain === 'programming') {
                modelSelect.value = 'deepseek-coder:6.7b';
                currentModel = 'deepseek-coder:6.7b';
                appendToOutput(`Switched to PROGRAMMING domain`, 'ai-en');
                appendToOutput(`Auto-loaded DeepSeek Coder 6.7B for coding`, 'ai-en');
            } else {
                // Switch back to default Qwen model for non-programming domains
                modelSelect.value = 'qwen2.5:0.5b';
                currentModel = 'qwen2.5:0.5b';
                appendToOutput(`Switched to ${currentDomain.toUpperCase()} domain`, 'ai-en');
                appendToOutput(`Auto-loaded Qwen 2.5 0.5B model`, 'ai-en');
            }
        });

        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !processing) {
                sendMessage();
            }
        });

        // Submit query - called by send button on mobile
        function submitQuery() {
            if (!processing) {
                sendMessage();
            }
        }

        // Math preprocessing function - now handled by external module
        // This function will be defined by math_processor.js

        function appendToOutput(text, className = '', isHTML = false) {
            const div = document.createElement('div');
            div.className = 'message ' + className;

            if (isHTML) {
                console.log('appendToOutput: Processing AI content for math rendering');
                console.log('appendToOutput: Content preview:', text.substring(0, 100) + '...');

                // Set content first
                div.innerHTML = text;

                // Only typeset if content actually has math delimiters
                if (text.includes('$$') || text.includes('\\\\(') || text.includes('\\\\[')) {
                    safeTypesetMath(div);
                }
            } else {
                div.textContent = text;
            }

            output.appendChild(div);
            // Scroll output container directly - avoids scrollIntoView affecting parent elements
            output.scrollTop = output.scrollHeight;
        }

        // Update research panel with Wikipedia sources and knowledge cards
        function updateResearchPanel(sources, knowledgeCards) {
            let html = '';

            // Render knowledge cards first (priority display at top)
            if (knowledgeCards && knowledgeCards.length > 0) {
                knowledgeCards.forEach(card => {
                    html += renderKnowledgeCard(card);
                });
            }

            // Render Wikipedia sources below cards
            if (sources && sources.length > 0) {
                sources.forEach(source => {
                    html += `
                        <div class="source-card" data-article-id="${source.id}" data-article-title="${source.title}" onclick="openArticle(this.dataset.articleId, this.dataset.articleTitle)">
                            <div class="source-title">${source.title}</div>
                            <div class="source-summary">${source.summary}</div>
                            <div class="source-meta">
                                ${source.word_count} words • Relevance: ${source.score}
                            </div>
                        </div>
                    `;
                });
            }

            // Show placeholder if nothing to display
            if (!html) {
                html = `
                    <div class="no-sources">
                        Ask a question and Wikipedia sources will appear here automatically.
                        <br><br>
                        Works great for history, science, people, places, and events.
                    </div>
                `;
            }

            researchContent.innerHTML = html;
        }

        // Render a knowledge card based on type
        function renderKnowledgeCard(card) {
            if (card.type === 'medical') {
                return renderMedicalCard(card);
            } else if (card.type === 'agricultural') {
                return renderAgriculturalCard(card);
            } else if (card.type === 'mental_health') {
                return renderMentalHealthCard(card);
            }
            return '';
        }

        // Render medical Quick Reference card
        function renderMedicalCard(card) {
            const data = card.data || {};
            let html = `
                <div class="knowledge-card medical-card">
                    <div class="knowledge-card-header">
                        <span class="knowledge-card-title">Quick Reference: ${card.keyword}</span>
                        <span class="knowledge-card-badge">Medical</span>
                    </div>
            `;

            // Danger signs section (warning)
            if (data.danger_signs && data.danger_signs.length > 0) {
                html += `
                    <div class="knowledge-card-section card-section-warning">
                        <div class="knowledge-card-section-title">Danger Signs</div>
                        <ul>
                            ${data.danger_signs.slice(0, 4).map(sign => `<li>${sign.replace(/_/g, ' ')}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Immediate care section
            if (data.immediate_care && data.immediate_care.length > 0) {
                html += `
                    <div class="knowledge-card-section card-section-care">
                        <div class="knowledge-card-section-title">Immediate Care</div>
                        <ul>
                            ${data.immediate_care.slice(0, 4).map(care => `<li>${care}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Possible causes section
            if (data.possible_causes && data.possible_causes.length > 0) {
                html += `
                    <div class="knowledge-card-section card-section-causes">
                        <div class="knowledge-card-section-title">Common Causes</div>
                        <ul>
                            ${data.possible_causes.slice(0, 4).map(cause => `<li>${cause.replace(/_/g, ' ')}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            html += '</div>';
            return html;
        }

        // Render agricultural Quick Reference card
        function renderAgriculturalCard(card) {
            const data = card.data || {};
            let html = `
                <div class="knowledge-card agricultural-card">
                    <div class="knowledge-card-header">
                        <span class="knowledge-card-title">Quick Reference: ${card.keyword}</span>
                        ${card.tigrinya_name ? `<span class="tigrinya-name">${card.tigrinya_name}</span>` : ''}
                        <span class="knowledge-card-badge">Agriculture</span>
                    </div>
            `;

            // Planting/Growing info for crops
            if (data.planting_time || data.growing_season) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Growing Season</div>
                        <p>
                            ${data.growing_season ? `Season: ${data.growing_season}` : ''}
                            ${data.planting_time ? `<br>Plant: ${data.planting_time}` : ''}
                            ${data.harvest_time ? `<br>Harvest: ${data.harvest_time}` : ''}
                        </p>
                    </div>
                `;
            }

            // Water requirements
            if (data.water_requirements) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Water Needs</div>
                        <p>${data.water_requirements}</p>
                    </div>
                `;
            }

            // Management practices
            if (data.management && data.management.length > 0) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Management</div>
                        <ul>
                            ${data.management.slice(0, 4).map(m => `<li>${m}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Varieties
            if (data.varieties && data.varieties.length > 0) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Recommended Varieties</div>
                        <p>${data.varieties.join(', ')}</p>
                    </div>
                `;
            }

            // Livestock-specific sections
            if (data.feeding && data.feeding.length > 0) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Feeding</div>
                        <ul>
                            ${data.feeding.slice(0, 3).map(f => `<li>${f}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            if (data.health && data.health.length > 0) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Health Care</div>
                        <ul>
                            ${data.health.slice(0, 3).map(h => `<li>${h}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Drought strategies
            if (data.strategies && data.strategies.length > 0) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Drought Strategies</div>
                        <ul>
                            ${data.strategies.slice(0, 4).map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }

            // Planting calendar entries
            if (data.june || data.july || data.february) {
                html += `
                    <div class="knowledge-card-section">
                        <div class="knowledge-card-section-title">Planting Calendar</div>
                        <ul>
                            ${data.february ? `<li><strong>Feb:</strong> ${data.february}</li>` : ''}
                            ${data.june ? `<li><strong>Jun:</strong> ${data.june}</li>` : ''}
                            ${data.july ? `<li><strong>Jul:</strong> ${data.july}</li>` : ''}
                            ${data.september ? `<li><strong>Sep:</strong> ${data.september}</li>` : ''}
                        </ul>
                    </div>
                `;
            }

            html += '</div>';
            return html;
        }

        // Render mental health Quick Reference card
        function renderMentalHealthCard(card) {
            const data = card.data || {};
            const isCrisis = card.is_crisis || false;
            const cardClass = isCrisis ? 'mental-health-card crisis-card' : 'mental-health-card';
            const badgeText = isCrisis ? 'Crisis Support' : 'Mental Health';

            let html = `
                <div class="knowledge-card ${cardClass}">
                    <div class="knowledge-card-header">
                        <span class="knowledge-card-title">Quick Reference: ${card.keyword}</span>
                        ${card.tigrinya_name ? `<span class="tigrinya-name">${card.tigrinya_name}</span>` : ''}
                        <span class="knowledge-card-badge">${badgeText}</span>
                    </div>
            `;

            // For crisis cards (suicide risk, etc.)
            if (isCrisis) {
                // Immediate action section (most important)
                if (data.immediate_action && data.immediate_action.length > 0) {
                    html += `
                        <div class="knowledge-card-section card-section-action">
                            <div class="knowledge-card-section-title">Immediate Action</div>
                            <ul>
                                ${data.immediate_action.slice(0, 5).map(action => `<li>${action}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                // Recognition signs
                if (data.recognition && data.recognition.length > 0) {
                    html += `
                        <div class="knowledge-card-section card-section-danger">
                            <div class="knowledge-card-section-title">Warning Signs</div>
                            <ul>
                                ${data.recognition.slice(0, 4).map(sign => `<li>${sign}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                // Do NOT section
                if (data.do_not && data.do_not.length > 0) {
                    html += `
                        <div class="knowledge-card-section card-section-donot">
                            <div class="knowledge-card-section-title">Do NOT</div>
                            <ul>
                                ${data.do_not.slice(0, 4).map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
            } else {
                // Standard mental health cards (depression, PTSD, anxiety, etc.)

                // Danger signs section (high priority)
                if (data.danger_signs && data.danger_signs.length > 0) {
                    html += `
                        <div class="knowledge-card-section card-section-danger">
                            <div class="knowledge-card-section-title">Seek Help If</div>
                            <ul>
                                ${data.danger_signs.slice(0, 4).map(sign => `<li>${sign}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                // Immediate support section
                if (data.immediate_support && data.immediate_support.length > 0) {
                    html += `
                        <div class="knowledge-card-section card-section-support">
                            <div class="knowledge-card-section-title">How to Help</div>
                            <ul>
                                ${data.immediate_support.slice(0, 4).map(support => `<li>${support}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                // Coping strategies section
                if (data.coping_strategies && data.coping_strategies.length > 0) {
                    html += `
                        <div class="knowledge-card-section card-section-coping">
                            <div class="knowledge-card-section-title">Coping Strategies</div>
                            <ul>
                                ${data.coping_strategies.slice(0, 4).map(strategy => `<li>${strategy}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }

                // Symptoms section (for recognition)
                if (data.symptoms && data.symptoms.length > 0) {
                    html += `
                        <div class="knowledge-card-section">
                            <div class="knowledge-card-section-title">Common Symptoms</div>
                            <ul>
                                ${data.symptoms.slice(0, 4).map(symptom => `<li>${symptom}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
            }

            html += '</div>';
            return html;
        }

        // Track article offset for pagination
        let articleOffset = 0;
        let lastQuery = '';
        let lastEnglishAnswer = '';
        let lastWikipediaSources = [];

        // Load more Wikipedia articles
        function loadMoreArticles() {
            const btn = document.getElementById('moreArticlesBtn');
            btn.disabled = true;
            btn.textContent = 'Loading...';

            // Use last query or default topic based on domain
            const query = lastQuery || currentDomain;
            // First click: offset 0->3 (initial shows 3), subsequent: 3->8, 8->13, etc.
            articleOffset = (articleOffset === 0) ? 3 : articleOffset + 5;

            fetch('/api/more_articles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, offset: articleOffset, limit: 5 })
            })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.textContent = 'More Articles';

                if (data.articles && data.articles.length > 0) {
                    let html = researchContent.innerHTML;
                    data.articles.forEach(article => {
                        html += `
                            <div class="source-card" data-article-id="${article.id}" data-article-title="${article.title}" onclick="openArticle(this.dataset.articleId, this.dataset.articleTitle)">
                                <div class="source-title">${article.title}</div>
                                <div class="source-snippet">${article.snippet || article.content.substring(0, 150)}...</div>
                                <div class="source-meta">Click to read full article</div>
                            </div>
                        `;
                    });
                    researchContent.innerHTML = html;
                } else {
                    appendToOutput('No more articles found for this topic.', 'ai-en');
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.textContent = 'More Articles';
                console.error('Error loading more articles:', err);
            });
        }

        // Search Wikipedia on demand
        function searchWikipedia() {
            const btn = document.getElementById('searchWikipediaBtn');
            const moreBtn = document.getElementById('moreArticlesBtn');

            if (!lastQuery) {
                appendToOutput('No recent query to search Wikipedia with.', 'ai-en');
                return;
            }

            btn.disabled = true;
            btn.textContent = 'Searching...';

            fetch('/api/search_wikipedia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: lastQuery,
                    max_results: 3
                })
            })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.textContent = 'Search Wiki';

                if (data.articles && data.articles.length > 0) {
                    // Reset article offset for new search
                    articleOffset = 0;

                    // Update research panel
                    updateResearchPanel(data.articles, []);

                    // Show More Articles button for pagination
                    moreBtn.style.display = 'inline-block';

                    appendToOutput(`Found ${data.articles.length} Wikipedia articles`, 'ai-en');
                } else {
                    updateResearchPanel([], []);
                    appendToOutput('No Wikipedia articles found for this query.', 'ai-en');
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.textContent = 'Search Wiki';
                console.error('Error searching Wikipedia:', err);
                appendToOutput('Error searching Wikipedia: ' + err.message, 'ai-en');
            });
        }

        // Store current article for Learn More feature
        let currentArticle = { title: '', content: '' };

        // Open full article in modal
        function openArticle(articleId, title) {
            // Quick validation
            if (!articleId) return;

            const modal = document.getElementById('articleModal');
            const articleTitle = document.getElementById('articleTitle');
            const articleBody = document.getElementById('articleBody');
            const articleMeta = document.getElementById('articleMeta');
            const learnMoreBtn = document.getElementById('learnMoreBtn');

            // Show modal with loading state
            modal.classList.add('active');
            articleTitle.textContent = title;
            articleBody.textContent = 'Loading article...';
            articleMeta.textContent = '';
            learnMoreBtn.disabled = true;

            // Fetch full article
            fetch(`/api/article/${articleId}`)
                .then(response => response.json())
                .then(article => {
                    if (article.error) {
                        articleBody.textContent = 'Error loading article: ' + article.error;
                    } else {
                        // Use textContent — content is plain text from wiki cleanup, not HTML.
                        // innerHTML would interpret < > in math notation as broken HTML tags,
                        // hiding chunks of content (e.g. "e < 3" becomes invisible tag "< 3").
                        articleBody.textContent = article.content;
                        articleMeta.textContent = `Word count: ${article.word_count}`;
                        // Store article for Learn More feature
                        currentArticle = { title: title, content: article.content };
                        learnMoreBtn.disabled = false;
                    }
                })
                .catch(error => {
                    articleBody.textContent = 'Error loading article: ' + error.message;
                });
        }

        // Learn More - Have AI teach about the article topic
        function learnMore(retryCount = 0) {
            if (!currentArticle.content) return;

            const learnMoreBtn = document.getElementById('learnMoreBtn');
            learnMoreBtn.disabled = true;
            learnMoreBtn.textContent = 'Loading...';

            // Close the modal only on first attempt
            if (retryCount === 0) {
                closeArticle();
                // Show that we're processing
                appendToOutput(`Teaching: ${currentArticle.title}...`, 'ai-status');
            } else {
                // Show retry message
                appendToOutput(`Retrying after context reset (attempt ${retryCount + 1})...`, 'ai-en');
            }

            // Use the same efficient approach as regular chat - let backend search Wikipedia automatically
            const teachingPrompt = `Teach me about ${currentArticle.title}`;

            // Send to AI
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: teachingPrompt,
                    domain: currentDomain,
                    model: currentModel,
                    english_only: isEnglishOnly
                })
            })
            .then(response => response.json())
            .then(data => {
                // Remove the "preparing" message
                output.removeChild(output.lastChild);

                if (data.response) {
                    const response = typeof data.response === 'string' ? JSON.parse(data.response) : data.response;

                    // Simplified validation - just show the response without complex checking

                    // Show the response with Learn More visual indicator
                    if (response.english) {
                        // Add Learn More indicator (escape title to prevent XSS)
                        const indicatorDiv = document.createElement('div');
                        const safeTitle = (currentArticle.title || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
                        indicatorDiv.innerHTML = `<span class="content-indicator learn-more">Learn More: ${safeTitle}</span>`;
                        output.appendChild(indicatorDiv);

                        // Check if content contains mathematical formulas or HTML formatting
                        const hasFormulas = response.english.includes('$$') || response.english.includes('\\\\(') || response.english.includes('\\\\[') || response.english.includes('$');

                        // Add content with enhanced styling
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'message ai-en enhanced-content';
                        if (hasFormulas) {
                            contentDiv.innerHTML = 'EN: ' + response.english;
                            safeTypesetMath(contentDiv);
                        } else {
                            contentDiv.textContent = 'EN: ' + response.english;
                        }
                        output.appendChild(contentDiv);
                        output.scrollTop = output.scrollHeight;
                    }
                    if (response.tigrinya && !isEnglishOnly) {
                        // Check if content contains mathematical formulas
                        const hasFormulas = response.tigrinya.includes('$$') || response.tigrinya.includes('\\\\(') || response.tigrinya.includes('\\\\[') || response.tigrinya.includes('$');
                        appendToOutput('TI: ' + response.tigrinya, 'ai-ti', hasFormulas);
                    }
                } else if (data.error) {
                    appendToOutput('Error: ' + data.error, 'ai-en');
                }

                // Reset button
                learnMoreBtn.disabled = false;
                learnMoreBtn.textContent = 'Learn More';
            })
            .catch(error => {
                output.removeChild(output.lastChild);
                appendToOutput('Error: ' + error.message, 'ai-en');
                learnMoreBtn.disabled = false;
                learnMoreBtn.textContent = 'Learn More';
            });
        }

        // Close article modal
        function closeArticle() {
            document.getElementById('articleModal').classList.remove('active');
        }

        // Close modal when clicking outside
        document.getElementById('articleModal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeArticle();
            }
        });

        // Close modal/drawer with Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeArticle();
                closeResearchDrawer();
            }
        });

        function handleSlashCommand(msg) {
            const cmd = msg.trim().toLowerCase().replace(/^\//, '').split(/\s+/)[0];
            if (cmd === 'faq' || cmd === 'help') {
                showFAQ();
            } else {
                appendToOutput('Unknown command: /' + cmd + '  —  Try /FAQ for a quick overview of NATNA.', 'ai-status');
            }
        }

        function showFAQ() {
            const faqHTML = `
<h2>Welcome to NATNA — Frequently Asked Questions</h2>

<h3>What is NATNA?</h3>
<p>NATNA is an offline AI assistant that runs entirely from a USB drive — no internet needed. It answers questions in both <strong>English</strong> and <strong>Tigrinya</strong>, making knowledge accessible where connectivity is limited.</p>

<h3>How do I ask a question?</h3>
<p>Type your question in the input bar at the bottom and press <strong>Enter</strong> (or tap the send button on mobile). NATNA will respond with an English answer and, if enabled, a Tigrinya translation.</p>

<h3>What are Domains?</h3>
<p>Domains help NATNA focus its answers. Pick the one closest to your topic:</p>
<ul>
  <li><strong>General</strong> — everyday questions, no filter</li>
  <li><strong>Medical</strong> — health, symptoms, first aid</li>
  <li><strong>Education</strong> — school subjects, learning</li>
  <li><strong>Agriculture</strong> — farming, soil, crops, livestock</li>
  <li><strong>Legal</strong> — basic legal concepts</li>
  <li><strong>Technical</strong> — computers, engineering, science</li>
</ul>
<p>You can switch domains anytime from the selector at the top — it takes effect on your next question.</p>

<h3>AI Models — Small vs. Large</h3>
<p>NATNA ships with several AI models of different sizes:</p>
<ul>
  <li><strong>Smaller models</strong> — faster responses, great for simple questions</li>
  <li><strong>Larger models</strong> — deeper, more detailed answers but take longer</li>
</ul>
<p>Switch models from the dropdown at the top. Start small; move up if you need more depth.</p>

<h3>Bilingual Responses (English + Tigrinya)</h3>
<p>By default, every answer comes in both English and Tigrinya. You can toggle to <strong>English-only</strong> mode using the language button in the header — handy when you only need one language and want faster output.</p>

<h3>Using NATNA from a Phone or Tablet</h3>
<p>NATNA runs a local web server at <code>http://localhost:8080</code>. Any device on the <strong>same WiFi network</strong> can connect:</p>
<ul>
  <li>On this computer, open <code>http://localhost:8080</code> in any browser</li>
  <li>From a phone or tablet, use the LAN IP shown when NATNA launches (e.g. <code>http://192.168.x.x:8080</code>)</li>
  <li>You'll get the same full interface — no app install needed</li>
</ul>

<h3>Wikipedia Integration</h3>
<p>NATNA includes an offline Wikipedia snapshot with <strong>475,000+ articles</strong>:</p>
<ul>
  <li>The <strong>Research Panel</strong> on the right shows relevant Wikipedia sources for each question</li>
  <li>Click any source to read the full article</li>
  <li>Use <strong>Expand Answer</strong> to ask the AI to go deeper using Wikipedia context</li>
</ul>

<h3>Tips &amp; Shortcuts</h3>
<ul>
  <li>Type <code>/FAQ</code> — show this help page</li>
  <li><strong>Clear History</strong> — use the clear button in the status bar to reset the chat</li>
  <li><strong>Switch models</strong> mid-conversation — each new question uses whichever model is selected</li>
  <li><strong>Theme toggle</strong> — switch between dark and light mode from the header</li>
</ul>

<div class="faq-footer">NATNA — Knowledge without borders. Built for offline, bilingual access to AI and encyclopedic content.</div>
`;
            appendToOutput(faqHTML, 'faq-message', true);
        }

        function sendMessage() {
            const message = input.value.trim();
            if (!message) return;

            // Intercept slash commands (client-side only, no server call)
            if (message.startsWith('/')) {
                appendToOutput('You: ' + message, 'user-message');
                input.value = '';
                handleSlashCommand(message);
                return;
            }

            // Save query for "load more" functionality
            lastQuery = message;
            articleOffset = 0;

            processing = true;

            // Add user's message to chat history
            appendToOutput('You: ' + message, 'user-message');

            input.value = '';
            input.blur();  // Dismiss mobile keyboard
            input.placeholder = 'Processing...';

            appendToOutput('Processing...', 'ai-status');

            // Clear research panel - no automatic search
            researchContent.innerHTML = '<div class="no-sources">Ask a question to see relevant Wikipedia sources here.<br><br>Click any source to read the full article.</div>';

            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    domain: currentDomain,
                    model: currentModel,
                    english_only: isEnglishOnly
                })
            })
            .then(response => response.json())
            .then(data => {
                // Remove "Processing..." message
                output.removeChild(output.lastChild);

                if (data.response) {
                    const response = typeof data.response === 'string' ? JSON.parse(data.response) : data.response;

                    // Show context warning if any
                    if (response.context_warning) {
                        appendToOutput(response.context_warning, 'ai-en');
                    }

                    if (response.english) {
                        // Check if content contains mathematical formulas or HTML formatting
                        const hasFormulas = response.english.includes('$$') || response.english.includes('\\\\(') || response.english.includes('\\\\[') || response.english.includes('$');
                        appendToOutput('EN: ' + response.english, 'ai-en', hasFormulas);

                        // Save for expand functionality
                        lastEnglishAnswer = response.english;
                    }
                    if (response.tigrinya && !isEnglishOnly) {
                        // Check if content contains mathematical formulas
                        const hasFormulas = response.tigrinya.includes('$$') || response.tigrinya.includes('\\\\(') || response.tigrinya.includes('\\\\[') || response.tigrinya.includes('$');
                        appendToOutput('TI: ' + response.tigrinya, 'ai-ti', hasFormulas);
                    }

                    // Update research panel with Wikipedia sources AND knowledge cards
                    const moreBtn = document.getElementById('moreArticlesBtn');
                    const hasWikiSources = data.wikipedia_sources && data.wikipedia_sources.length > 0;
                    const hasKnowledgeCards = data.knowledge_cards && data.knowledge_cards.length > 0;

                    if (hasWikiSources || hasKnowledgeCards) {
                        // Pass both sources and knowledge cards to research panel
                        updateResearchPanel(data.wikipedia_sources || [], data.knowledge_cards || []);

                        // Start blinking Sources button to attract attention (mobile)
                        const sourcesBtn = document.getElementById('sourcesToggle');
                        if (sourcesBtn && !sourcesBtn.classList.contains('blinking')) {
                            sourcesBtn.classList.add('blinking');
                        }

                        // Show appropriate message
                        if (hasKnowledgeCards && hasWikiSources) {
                            appendToOutput('See Research Panel for Quick Reference + sources', 'ai-en');
                        } else if (hasKnowledgeCards) {
                            appendToOutput('See Research Panel for Quick Reference', 'ai-en');
                        } else {
                            appendToOutput('See Research Panel for sources', 'ai-en');
                        }

                        // Save Wikipedia sources and show expand + more articles buttons
                        lastWikipediaSources = data.wikipedia_sources || [];
                        articleOffset = 0;
                        const expandBtn = document.getElementById('expandAnswerBtn');
                        if (expandBtn && lastEnglishAnswer && hasWikiSources) {
                            expandBtn.style.display = 'inline';
                        }
                        if (moreBtn && hasWikiSources) {
                            moreBtn.style.display = 'inline-block';
                        }
                    } else {
                        updateResearchPanel([], []);

                        // Hide expand and more articles buttons if no sources
                        const expandBtn = document.getElementById('expandAnswerBtn');
                        if (expandBtn) {
                            expandBtn.style.display = 'none';
                        }
                        if (moreBtn) {
                            moreBtn.style.display = 'none';
                        }
                        lastWikipediaSources = [];
                    }

                    // Update context status after response
                    setTimeout(checkSystemStatus, 500);
                } else {
                    appendToOutput('Error: No response', 'error');
                    updateResearchPanel([], []);
                }

                appendToOutput('');
                processing = false;
                input.placeholder = 'Ask your question...';
            })
            .catch(error => {
                output.removeChild(output.lastChild);
                appendToOutput('Error: ' + error.message, 'error');
                appendToOutput('');
                processing = false;
                input.placeholder = 'Ask your question...';
                updateResearchPanel([], []);
            });
        }

        // Function to transition to terminal
        function showTerminal() {
            const loadingScreen = document.getElementById('loadingScreen');
            const mainContainer = document.getElementById('mainContainer');

            // Immediate hide loading screen and show main container
            loadingScreen.style.display = 'none';
            mainContainer.style.opacity = '1';
        }

        // Function for skip button
        function skipLoading() {
            showTerminal();
        }

        // Ollama connection state
        let _ollamaStarting = false;
        let _ollamaWasConnected = false;
        let _ollamaReconnectCycle = false;
        // Update Ollama status display from pre-fetched data (used by batched endpoint)
        function updateOllamaDisplay(data) {
            if (_ollamaStarting) return;
            if (window._ollamaStartGrace && (Date.now() - window._ollamaStartGrace < 10000)) return;

            const status = document.getElementById('ollama-status');
            const startBtn = document.getElementById('start-server-btn');

            if (data && data.status === 'connected') {
                status.innerHTML = '<span class="status-dot connected"></span>Server: Connected';
                status.style.color = 'var(--accent)';
                startBtn.style.display = 'none';
                _ollamaWasConnected = true;
                _ollamaReconnectCycle = false;
                window._ollamaStartGrace = 0;
            } else {
                if (_ollamaWasConnected && !_ollamaReconnectCycle) {
                    status.innerHTML = '<span class="status-dot warning"></span>Server: Reconnecting...';
                    status.style.color = 'var(--warning)';
                    startBtn.style.display = 'none';
                    _ollamaReconnectCycle = true;
                } else {
                    status.innerHTML = '<span class="status-dot disconnected"></span>Server: Disconnected';
                    status.style.color = 'var(--error)';
                    startBtn.style.display = 'inline-block';
                    _ollamaWasConnected = false;
                    _ollamaReconnectCycle = false;
                }
            }
        }
        // Legacy standalone check (used by manual start flow)
        function checkOllamaStatus() {
            if (window._ollamaStartGrace && (Date.now() - window._ollamaStartGrace < 10000)) return;
            if (_ollamaStarting) return;
            fetchWithTimeout('/api/status/ollama', 5000)
                .then(response => response.json())
                .then(data => updateOllamaDisplay(data))
                .catch(() => updateOllamaDisplay(null));
        }

        // Manual Server Start Function
        function startOllamaServer() {
            if (_ollamaStarting) return;
            _ollamaStarting = true;

            const status = document.getElementById('ollama-status');
            const startBtn = document.getElementById('start-server-btn');
            const progressDiv = document.getElementById('model-progress');
            const progressBar = document.getElementById('progress-bar');
            const progressText = document.getElementById('progress-text');

            status.innerHTML = '<span class="status-dot warning"></span>Server: Starting...';
            status.style.color = 'var(--warning)';
            startBtn.style.display = 'none';
            progressDiv.style.display = 'block';
            progressBar.style.width = '100%';
            progressBar.style.animation = 'pulse 1.5s ease-in-out infinite';
            progressText.innerHTML = 'Starting server...';

            fetch('/api/start-ollama', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                progressBar.style.animation = '';
                progressDiv.style.display = 'none';
                _ollamaStarting = false;

                if (data.status === 'started') {
                    status.innerHTML = '<span class="status-dot connected"></span>Server: Connected';
                    status.style.color = 'var(--accent)';
                    startBtn.style.display = 'none';
                    appendToOutput('AI server started successfully', 'system-message');
                    // Suppress immediate status poll flicker
                    window._ollamaStartGrace = Date.now();
                } else {
                    const errMsg = data.error || 'Unknown error';
                    status.innerHTML = '<span class="status-dot disconnected"></span>Server: Start Failed';
                    status.style.color = 'var(--error)';
                    startBtn.style.display = 'inline-block';
                    appendToOutput('Failed to start server: ' + errMsg, 'error-message');
                }
            })
            .catch(error => {
                progressBar.style.animation = '';
                progressDiv.style.display = 'none';
                _ollamaStarting = false;
                status.innerHTML = '<span class="status-dot disconnected"></span>Server: Start Failed';
                status.style.color = 'var(--error)';
                startBtn.style.display = 'inline-block';
                appendToOutput('Failed to start server: ' + error.message, 'error-message');
            });
        }


        // System Status Checking — single batched request replaces 4 separate fetches
        function checkSystemStatus() {
            fetchWithTimeout('/api/status/all', 5000)
                .then(response => response.json())
                .then(data => {
                    // Ollama status
                    updateOllamaDisplay(data.ollama || null);

                    // Wikipedia status
                    const wikiEl = document.getElementById('wiki-status');
                    if (data.wikipedia && data.wikipedia.status === 'available') {
                        wikiEl.innerHTML = '<span class="status-dot connected"></span>Wikipedia: ' + data.wikipedia.articles + ' articles';
                        wikiEl.style.color = 'var(--info)';
                    } else {
                        wikiEl.innerHTML = '<span class="status-dot disconnected"></span>Wikipedia: Unavailable';
                        wikiEl.style.color = 'var(--error)';
                    }

                    // Context status
                    const ctxEl = document.getElementById('context-status');
                    const ctxData = data.context || {};
                    if (ctxData.limit) {
                        const percent = ctxData.usage_percent || 0;
                        const msgs = ctxData.message_count || 0;
                        let color = 'var(--info)';
                        if (percent > 80) color = 'var(--warning)';
                        if (percent > 95) color = 'var(--error)';

                        ctxEl.innerHTML = 'Context: ' + percent + '% (' + msgs + ' msgs)';
                        ctxEl.style.color = color;

                        if (percent >= 100 && !ctxEl.dataset.autoCompacted) {
                            appendToOutput('Context at ' + percent + '% - Auto-compacting history...', 'ai-en');
                            ctxEl.dataset.autoCompacted = 'true';
                            clearHistory();
                        } else if (percent > 80 && !ctxEl.dataset.warned && percent < 100) {
                            appendToOutput('Context is ' + percent + '% full. Will auto-compact at 100%.', 'ai-en');
                            ctxEl.dataset.warned = 'true';
                        }
                        if (percent < 50) {
                            ctxEl.dataset.warned = '';
                            ctxEl.dataset.autoCompacted = '';
                        }
                    } else {
                        ctxEl.innerHTML = 'Context: N/A';
                        ctxEl.style.color = 'var(--text-muted)';
                    }

                    // Memory status
                    const memEl = document.getElementById('memory-status');
                    const memData = data.memory || {};
                    if (memData.rss !== undefined && memData.percent !== undefined && memData.model !== 'monitoring-unavailable') {
                        let color = 'var(--info)';
                        if (memData.rss > 500) color = 'var(--warning)';
                        if (memData.rss > 1000) color = 'var(--error)';
                        memEl.innerHTML = 'RAM: ' + (parseFloat(memData.rss) / 1024).toFixed(1) + 'GB';
                        memEl.style.color = color;
                    } else {
                        memEl.innerHTML = 'RAM: N/A';
                        memEl.style.color = 'var(--text-muted)';
                    }
                })
                .catch(() => {
                    // Batch fetch failed — show error state on all indicators
                    updateOllamaDisplay(null);
                    document.getElementById('wiki-status').innerHTML = '<span class="status-dot disconnected"></span>Wikipedia: Error';
                    document.getElementById('wiki-status').style.color = 'var(--error)';
                    document.getElementById('context-status').innerHTML = 'Context: N/A';
                    document.getElementById('context-status').style.color = 'var(--text-muted)';
                    document.getElementById('memory-status').innerHTML = 'RAM: Error';
                    document.getElementById('memory-status').style.color = 'var(--error)';
                });
        }

        // Clear conversation history
        function clearHistory() {
            fetch('/api/clear_history')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        appendToOutput('Conversation history cleared. Starting fresh.', 'ai-en');
                        // Reset context status display
                        const status = document.getElementById('context-status');
                        status.innerHTML = 'Context: 0%';
                        status.style.color = 'var(--info)';
                        status.dataset.warned = '';

                        // Clear expand functionality data and hide button
                        lastQuery = '';
                        lastEnglishAnswer = '';
                        lastWikipediaSources = [];
                        const expandBtn = document.getElementById('expandAnswerBtn');
                        if (expandBtn) {
                            expandBtn.style.display = 'none';
                        }
                    }
                })
                .catch(err => {
                    appendToOutput('Error clearing history: ' + err.message, 'ai-en');
                });
        }

        function expandAnswer() {
            const expandBtn = document.getElementById('expandAnswerBtn');

            if (!lastEnglishAnswer || !lastWikipediaSources.length) {
                appendToOutput('No answer available to expand.', 'ai-en');
                return;
            }

            // Disable button and show loading
            expandBtn.disabled = true;
            expandBtn.textContent = 'Expanding...';

            const requestData = {
                original_answer: lastEnglishAnswer,
                query: lastQuery,
                wikipedia_sources: lastWikipediaSources,
                english_only: isEnglishOnly
            };

            fetch('/api/expand', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.enhanced_answer) {
                    // Add enhanced answer to chat with visual indicator
                    const indicatorDiv = document.createElement('div');
                    indicatorDiv.innerHTML = '<span class="content-indicator enhanced">Enhanced Answer</span>';
                    output.appendChild(indicatorDiv);

                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message ai-en enhanced-content';
                    contentDiv.textContent = 'EN: ' + data.enhanced_answer;
                    output.appendChild(contentDiv);

                    // Show Tigrinya translation if available
                    if (data.enhanced_answer_ti && !isEnglishOnly) {
                        const tiDiv = document.createElement('div');
                        tiDiv.className = 'message ai-ti enhanced-content';
                        tiDiv.textContent = 'TI: ' + data.enhanced_answer_ti;
                        output.appendChild(tiDiv);
                    }

                    if (data.source) {
                        appendToOutput(data.source, 'ai-en');
                    }
                    output.scrollTop = output.scrollHeight;

                    // Hide expand button after use
                    expandBtn.style.display = 'none';
                } else {
                    appendToOutput('Failed to expand answer: ' + (data.error || 'Unknown error'), 'ai-en');
                }
            })
            .catch(err => {
                appendToOutput('Error expanding answer: ' + err.message, 'ai-en');
            })
            .finally(() => {
                // Re-enable button
                expandBtn.disabled = false;
                expandBtn.textContent = 'Expand';
            });
        }

        // Theme toggle functionality
        let isDarkMode = true;
        function toggleTheme() {
            const body = document.body;
            const themeBtn = document.getElementById('themeToggle');

            if (isDarkMode) {
                body.classList.add('light-theme');
                themeBtn.textContent = 'Light';
                isDarkMode = false;
                localStorage.setItem('natna-theme', 'light');
            } else {
                body.classList.remove('light-theme');
                themeBtn.textContent = 'Dark';
                isDarkMode = true;
                localStorage.setItem('natna-theme', 'dark');
            }
        }

        // Language toggle functionality
        let isEnglishOnly = false;
        function toggleLanguage() {
            const langBtn = document.getElementById('langToggle');

            if (isEnglishOnly) {
                // Switch back to dual language
                langBtn.textContent = 'Ti/En';
                langBtn.classList.remove('english-only');
                langBtn.title = 'Toggle Tigrinya responses';
                isEnglishOnly = false;
                localStorage.setItem('natna-lang', 'dual');

                // Show all existing Tigrinya responses
                const tigrinya = document.querySelectorAll('.ai-ti');
                tigrinya.forEach(el => el.style.display = 'block');
            } else {
                // Switch to English only
                langBtn.textContent = 'En';
                langBtn.classList.add('english-only');
                langBtn.title = 'English only mode (faster)';
                isEnglishOnly = true;
                localStorage.setItem('natna-lang', 'english');

                // Hide all existing Tigrinya responses
                const tigrinya = document.querySelectorAll('.ai-ti');
                tigrinya.forEach(el => el.style.display = 'none');
            }
        }

        // Load saved theme on startup
        function loadSavedTheme() {
            const savedTheme = localStorage.getItem('natna-theme');
            if (savedTheme === 'light') {
                document.body.classList.add('light-theme');
                document.getElementById('themeToggle').textContent = 'Light';
                isDarkMode = false;
            }
        }

        // Load saved language preference on startup
        function loadSavedLanguage() {
            const savedLang = localStorage.getItem('natna-lang');
            if (savedLang === 'english') {
                const langBtn = document.getElementById('langToggle');
                langBtn.textContent = 'En';
                langBtn.classList.add('english-only');
                langBtn.title = 'English only mode (faster)';
                isEnglishOnly = true;

                // Hide all existing Tigrinya responses
                setTimeout(() => {
                    const tigrinya = document.querySelectorAll('.ai-ti');
                    tigrinya.forEach(el => el.style.display = 'none');
                }, 100);
            }
        }

        // Mobile research drawer toggle
        // Mobile picker drawer functions
        function openPickerDrawer(type) {
            closePickerDrawers();
            var id = type === 'domain' ? 'domainPickerDrawer' : 'modelPickerDrawer';
            var drawer = document.getElementById(id);
            var backdrop = document.getElementById('drawerBackdrop');
            if (drawer) { drawer.style.display = 'flex'; setTimeout(function() { drawer.classList.add('drawer-open'); }, 10); }
            if (backdrop) backdrop.classList.add('active');
        }
        function closePickerDrawers() {
            var drawers = document.querySelectorAll('.mobile-picker-drawer');
            drawers.forEach(function(d) { d.classList.remove('drawer-open'); setTimeout(function() { if (!d.classList.contains('drawer-open')) d.style.display = ''; }, 300); });
        }
        function pickDomain(el) {
            var val = el.getAttribute('data-value');
            var sel = document.getElementById('domainSelect');
            if (sel) { sel.value = val; sel.dispatchEvent(new Event('change')); }
            // Update selected state
            el.parentNode.querySelectorAll('.picker-option').forEach(function(o) { o.classList.remove('selected'); });
            el.classList.add('selected');
            // Update mobile button label
            var label = document.getElementById('mobileDomainLabel');
            if (label) label.textContent = el.textContent.split('/')[0].trim();
            closePickerDrawers();
            var backdrop = document.getElementById('drawerBackdrop');
            if (backdrop) backdrop.classList.remove('active');
        }
        // Pending model change (for admin password verification)
        var pendingModelElement = null;

        function pickModel(el) {
            // Store the element for later if password verification needed
            pendingModelElement = el;
            // Show password modal - mobile users need admin access
            showAdminModal();
        }

        function applyModelChange(el) {
            var val = el.getAttribute('data-value');
            var sel = document.getElementById('modelSelect');
            if (sel) { sel.value = val; sel.dispatchEvent(new Event('change')); }
            // Update selected state
            el.parentNode.querySelectorAll('.picker-option').forEach(function(o) { o.classList.remove('selected'); });
            el.classList.add('selected');
            // Update mobile button label
            var label = document.getElementById('mobileModelLabel');
            if (label) label.textContent = el.textContent;
            closePickerDrawers();
            var backdrop = document.getElementById('drawerBackdrop');
            if (backdrop) backdrop.classList.remove('active');
        }

        function showAdminModal() {
            var modal = document.getElementById('adminModal');
            var input = document.getElementById('adminPassword');
            var error = document.getElementById('adminError');
            if (modal) modal.classList.add('active');
            if (input) { input.value = ''; input.focus(); }
            if (error) error.textContent = '';
        }

        function closeAdminModal() {
            var modal = document.getElementById('adminModal');
            if (modal) modal.classList.remove('active');
            pendingModelElement = null;
            // Close the picker drawer too
            closePickerDrawers();
            var backdrop = document.getElementById('drawerBackdrop');
            if (backdrop) backdrop.classList.remove('active');
        }

        function verifyAdminPassword() {
            var input = document.getElementById('adminPassword');
            var error = document.getElementById('adminError');
            var password = input ? input.value : '';

            fetch('/api/verify_admin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password: password })
            })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.success) {
                    // Save reference before closing modal (which clears it)
                    var elementToApply = pendingModelElement;
                    // Close modal (just hide it, don't clear pending yet)
                    var modal = document.getElementById('adminModal');
                    if (modal) modal.classList.remove('active');
                    // Apply the model change
                    if (elementToApply) {
                        applyModelChange(elementToApply);
                    }
                    pendingModelElement = null;
                } else {
                    if (error) error.textContent = 'Incorrect password';
                    if (input) input.value = '';
                }
            })
            .catch(function() {
                if (error) error.textContent = 'Verification failed';
            });
        }

        // Allow Enter key to submit password
        document.addEventListener('DOMContentLoaded', function() {
            var pwInput = document.getElementById('adminPassword');
            if (pwInput) {
                pwInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') verifyAdminPassword();
                });
            }
        });

        function toggleResearchDrawer() {
            const panel = document.getElementById('researchPanel');
            const backdrop = document.getElementById('drawerBackdrop');
            const sourcesBtn = document.getElementById('sourcesToggle');

            // Stop blinking when user opens sources
            if (sourcesBtn) {
                sourcesBtn.classList.remove('blinking');
            }

            if (panel.classList.contains('drawer-open')) {
                closeResearchDrawer();
            } else {
                panel.classList.add('drawer-open');
                backdrop.classList.add('active');
            }
        }

        function closeResearchDrawer() {
            const panel = document.getElementById('researchPanel');
            const backdrop = document.getElementById('drawerBackdrop');
            panel.classList.remove('drawer-open');
            backdrop.classList.remove('active');
        }

        // QR Connect widget toggle
        function toggleQrConnect() {
            var popup = document.getElementById('qrConnectPopup');
            if (popup) popup.classList.toggle('visible');
        }
        document.addEventListener('click', function(e) {
            var popup = document.getElementById('qrConnectPopup');
            var btn = document.getElementById('qrConnectBtn');
            if (popup && btn && !popup.contains(e.target) && e.target !== btn) {
                popup.classList.remove('visible');
            }
        });

        // Loading screen transition
        window.addEventListener('load', function() {
            // Load saved preferences
            loadSavedTheme();
            loadSavedLanguage();

            // After 3 seconds, automatically show terminal
            setTimeout(() => {
                showTerminal();
                // Start checking system status every 10 seconds (batched endpoint)
                checkSystemStatus();
                setInterval(checkSystemStatus, 10000);
            }, 3000);
        });
    </script>
</body>
</html>'''
            self.wfile.write(html.encode('utf-8'))

        # Static file serving
        elif self.path.endswith('.js'):
            try:
                # Get the file name (e.g., math_processor.js)
                filename = self.path.lstrip('/')
                base_dir = os.path.realpath(os.path.dirname(__file__))
                filepath = os.path.realpath(os.path.join(base_dir, filename))
                if not filepath.startswith(base_dir):
                    self.send_error(403)
                    return

                if os.path.exists(filepath):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/javascript')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    with open(filepath, 'rb') as file:
                        self.wfile.write(file.read())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'File not found')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'Server error: {str(e)}'.encode('utf-8'))

        # CSS file serving
        elif self.path.endswith('.css'):
            try:
                filename = self.path.lstrip('/')
                base_dir = os.path.realpath(os.path.dirname(__file__))
                filepath = os.path.realpath(os.path.join(base_dir, filename))
                if not filepath.startswith(base_dir):
                    self.send_error(403)
                    return

                if os.path.exists(filepath):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/css')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    with open(filepath, 'rb') as file:
                        self.wfile.write(file.read())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'CSS file not found')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'CSS server error: {str(e)}'.encode('utf-8'))

        # Font file serving
        elif self.path.endswith(('.woff', '.woff2', '.ttf')):
            try:
                filename = self.path.lstrip('/')
                base_dir = os.path.realpath(os.path.dirname(__file__))
                filepath = os.path.realpath(os.path.join(base_dir, filename))
                if not filepath.startswith(base_dir):
                    self.send_error(403)
                    return

                if os.path.exists(filepath):
                    self.send_response(200)
                    if self.path.endswith('.woff'):
                        self.send_header('Content-type', 'font/woff')
                    elif self.path.endswith('.woff2'):
                        self.send_header('Content-type', 'font/woff2')
                    elif self.path.endswith('.ttf'):
                        self.send_header('Content-type', 'font/truetype')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    with open(filepath, 'rb') as file:
                        self.wfile.write(file.read())
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'Font file not found')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'Font server error: {str(e)}'.encode('utf-8'))

        elif self.path == '/favicon.ico':
            # Return empty favicon to prevent 404 errors
            self.send_response(204)  # No content
            self.end_headers()

        else:
            # Return 404 for other requests
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not found')

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE or content_length < 0:
                self.send_error(413)
                return
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                domain = data.get('domain', 'general')
                model = data.get('model', 'qwen2.5:0.5b')
                english_only = data.get('english_only', False)

                if not message:
                    self.send_error(400, "Message required")
                    return

                print(f"Query ({domain} domain, {model}, english_only={english_only}): {message}")

                # Process query with selected AI model
                start_time = time.time()
                print(f"[TIME] Starting translator.process_query at {start_time:.3f}")

                result = translator.process_query(message, model=model, domain=domain, english_only=True)  # Always get English first

                end_time = time.time()
                duration = end_time - start_time
                print(f"[TIME] translator.process_query completed in {duration:.3f} seconds")

                # Handle response and translate to Tigrinya if Ti toggle is ON
                if result and 'response' in result:
                    response_data = result['response']

                    # If Tigrinya is enabled (english_only=False), translate with Natna Ti BETA
                    if not english_only:
                        english_text = response_data.get('english', '') if isinstance(response_data, dict) else str(response_data)
                        print(f"[NATNA-TI] Translating response to Tigrinya...")
                        ti_start = time.time()

                        tigrinya_text = translate_with_natna_ti(english_text)

                        if tigrinya_text:
                            print(f"[NATNA-TI] Translation completed in {time.time() - ti_start:.3f}s")
                            response_data = {
                                'english': english_text,
                                'tigrinya': tigrinya_text
                            }
                        else:
                            # Fallback if translation fails
                            response_data = {
                                'english': english_text,
                                'tigrinya': '[Natna Ti BETA: Translation loading...]'
                            }

                    # Build response with Wikipedia sources preserved
                    response = {"response": json.dumps(response_data)}

                    # Extract knowledge cards for Quick Reference display
                    knowledge_cards = translator.extract_knowledge_cards(message, domain=domain)
                    if knowledge_cards:
                        response['knowledge_cards'] = knowledge_cards
                        print(f"[CARDS] Extracted {len(knowledge_cards)} knowledge card(s) for query")

                    # Pass through Wikipedia sources for research panel
                    # Check both result and result['response'] since process_query wraps AI results
                    inner_response = result.get('response', {})
                    if isinstance(inner_response, dict):
                        if 'wikipedia_sources' in inner_response:
                            response['wikipedia_sources'] = inner_response['wikipedia_sources']
                        if 'source' in inner_response:
                            response['source'] = inner_response['source']
                        if 'domain' in inner_response:
                            response['domain'] = inner_response['domain']
                    # Also check top-level for non-AI responses
                    if 'wikipedia_sources' in result:
                        response['wikipedia_sources'] = result['wikipedia_sources']
                    if 'source' in result:
                        response['source'] = result['source']
                    if 'domain' in result:
                        response['domain'] = result['domain']
                else:
                    # Better fallback for general knowledge questions
                    fallback_english = f"I can help with medical, agricultural, and educational questions. For '{message}', please try switching to the appropriate domain or ask about health, farming, or learning topics."

                    if not english_only:
                        fallback_tigrinya = translate_with_natna_ti(fallback_english) or "ንሕክምናዊ፣ ንህርሻዊ፣ ከምኡ ድማ ንትምህርታዊ ሕቶታት ክሕግዝ እኽእል።"
                        fallback_response = {
                            "english": fallback_english,
                            "tigrinya": fallback_tigrinya
                        }
                    else:
                        fallback_response = {"english": fallback_english}

                    response = {"response": json.dumps(fallback_response)}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                print(f"Error processing chat: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))

        elif self.path == '/api/more_articles':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE or content_length < 0:
                self.send_error(413)
                return
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                query = data.get('query', 'education')
                offset = data.get('offset', 0)
                limit = data.get('limit', 5)

                # Search for more Wikipedia articles using the translator's search
                articles = []
                if hasattr(translator, 'wikipedia_search') and translator.wikipedia_search:
                    # Get enough results to support pagination - request more than we need
                    total_needed = offset + limit
                    results = translator.wikipedia_search.search(query, max_results=max(total_needed, 20), min_words=50)
                    # Skip already shown articles
                    results = results[offset:offset + limit] if len(results) > offset else []
                    for r in results:
                        articles.append({
                            'id': r.get('id', 0),
                            'title': r.get('title', 'Unknown'),
                            'snippet': r.get('summary', r.get('content', '')[:200]) if r.get('summary') or r.get('content') else '',
                            'word_count': r.get('word_count', 0)
                        })

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"articles": articles}).encode('utf-8'))

            except Exception as e:
                print(f"Error loading more articles: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))

        elif self.path == '/api/start-ollama':
            # Manual Ollama server start endpoint
            try:
                print("[LAUNCH] Manual Ollama server start requested")
                success, error = _start_ollama_server()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                if success:
                    print("[OK] Ollama server started successfully")
                    self.wfile.write(json.dumps({
                        "status": "started",
                        "message": "Ollama server is running"
                    }).encode('utf-8'))
                else:
                    print(f"[ERROR] Ollama server failed to start: {error}")
                    self.wfile.write(json.dumps({
                        "status": "failed",
                        "error": error or "Unknown error"
                    }).encode('utf-8'))

            except Exception as e:
                print(f"[ERROR] Error in start-ollama handler: {e}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "failed",
                    "error": "Failed to start Ollama server"
                }).encode('utf-8'))

        elif self.path == '/api/expand':
            # Expand Answer endpoint - enhance answer with Wikipedia context
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE or content_length < 0:
                self.send_error(413)
                return
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                original_answer = data.get('original_answer', '')
                query = data.get('query', '')
                wikipedia_sources = data.get('wikipedia_sources', [])
                english_only = data.get('english_only', False)

                if not original_answer or not wikipedia_sources:
                    self.send_error(400, "Original answer and Wikipedia sources required")
                    return

                print(f"[SEARCH] Expanding answer for: {query}")

                # Use the top Wikipedia result to enhance the answer
                top_source = wikipedia_sources[0] if wikipedia_sources else {}
                wiki_content = top_source.get('content', '')
                wiki_title = top_source.get('title', 'Unknown')

                # Create enhancement prompt
                enhancement_prompt = f"""Based on this Wikipedia article about "{wiki_title}":

{wiki_content[:2000]}...

Please enhance this answer with additional context and details:

{original_answer}

Provide a more comprehensive response that integrates the Wikipedia information naturally."""

                # Call Ollama to enhance the answer
                enhanced_response_obj = translator._query_local_model(enhancement_prompt, model='qwen2.5:0.5b')

                # Extract the English text from the response object
                if enhanced_response_obj and isinstance(enhanced_response_obj, dict) and 'english' in enhanced_response_obj:
                    enhanced_answer = enhanced_response_obj['english']
                else:
                    enhanced_answer = str(enhanced_response_obj) if enhanced_response_obj else "Failed to enhance answer"

                # Translate to Tigrinya if not english_only
                enhanced_answer_ti = None
                if not english_only and enhanced_answer:
                    print(f"[NATNA-TI] Translating enhanced answer to Tigrinya...")
                    enhanced_answer_ti = translate_with_natna_ti(enhanced_answer)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response_data = {
                    "enhanced_answer": enhanced_answer,
                    "source": f"Enhanced with Wikipedia: {wiki_title}"
                }
                if enhanced_answer_ti:
                    response_data["enhanced_answer_ti"] = enhanced_answer_ti
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as e:
                print(f"Error expanding answer: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))

        elif self.path == '/api/verify_admin':
            # Admin password verification for mobile model changes
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE or content_length < 0:
                self.send_error(413)
                return
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                password = data.get('password', '')

                # Admin password from environment (fallback for field deployment)
                ADMIN_PASSWORD = os.environ.get('NATNA_ADMIN_PW', 'Mekelle8080')

                # Rate limiting: max 5 attempts per minute per IP
                client_ip = self.client_address[0]
                now = time.time()
                if client_ip not in _admin_attempts:
                    _admin_attempts[client_ip] = []
                # Prune attempts older than 60 seconds
                _admin_attempts[client_ip] = [t for t in _admin_attempts[client_ip] if now - t < 60]
                if len(_admin_attempts[client_ip]) >= 5:
                    self.send_response(429)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Too many attempts. Try again later."}).encode('utf-8'))
                    return
                _admin_attempts[client_ip].append(now)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                if password == ADMIN_PASSWORD:
                    self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
                else:
                    self.wfile.write(json.dumps({"success": False}).encode('utf-8'))

            except Exception as e:
                print(f"Error in admin verification: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))

        elif self.path == '/api/search_wikipedia':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE or content_length < 0:
                self.send_error(413)
                return
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                query = data.get('query', '')
                max_results = data.get('max_results', 3)

                if not query:
                    self.send_error(400, "Query required")
                    return

                print(f"[SEARCH] Wikipedia search request: {query}")

                # Search Wikipedia using the enhanced search
                articles = []
                if hasattr(translator, 'wikipedia_search') and translator.wikipedia_search:
                    try:
                        results = translator.wikipedia_search.search(
                            query,
                            max_results=max_results,
                            min_words=50
                        )

                        for r in results:
                            articles.append({
                                'id': r.get('id', 0),
                                'title': r.get('title', 'Unknown'),
                                'summary': r.get('summary', '')[:150] + '...' if r.get('summary') and len(r.get('summary', '')) > 150 else r.get('summary', ''),
                                'word_count': r.get('word_count', 0),
                                'score': r.get('score', 0)
                            })

                        print(f"[OK] Found {len(articles)} Wikipedia articles")
                    except Exception as e:
                        print(f"[ERROR] Wikipedia search failed: {e}")

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"articles": articles}).encode('utf-8'))

            except Exception as e:
                print(f"Error in Wikipedia search: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error"}).encode('utf-8'))



    def log_message(self, format, *args):
        pass

# === BULLETPROOF PROCESS MANAGEMENT SYSTEM ===

# Global state for graceful shutdown
shutdown_flag = False
httpd_server = None
server_port = None
lan_ip = None
pid_file_path = None
spawned_processes = []  # Track all spawned processes for cleanup
_ollama_last_restart_attempt = 0  # Timestamp of last auto-restart attempt (30s cooldown)

def _find_ollama_binary():
    """Locate the ollama binary on the system."""
    import shutil
    path = shutil.which('ollama')
    if path:
        return path
    # Check common macOS/Linux locations
    for candidate in ['/usr/local/bin/ollama', '/opt/homebrew/bin/ollama',
                      os.path.expanduser('~/bin/ollama'), '/usr/bin/ollama']:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

def _start_ollama_server():
    """Attempt to start the Ollama server. Returns (success: bool, error: str|None)."""
    import subprocess
    global spawned_processes

    # Check if Ollama is already running
    try:
        resp = requests.get('http://localhost:11434', timeout=2)
        if resp.status_code == 200:
            return True, None
    except Exception:
        pass

    ollama_bin = _find_ollama_binary()
    if not ollama_bin:
        return False, "Ollama binary not found. Install from https://ollama.com"

    try:
        # Spawn ollama serve as a background process
        proc = subprocess.Popen(
            [ollama_bin, 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        spawned_processes.append(proc)
        print(f"[NOTE] Tracking spawned Ollama process PID {proc.pid} for cleanup")

        # Poll for readiness (up to 15 seconds)
        for i in range(15):
            time.sleep(1)
            # Check process hasn't crashed
            if proc.poll() is not None:
                return False, f"Ollama process exited with code {proc.returncode}"
            try:
                resp = requests.get('http://localhost:11434', timeout=2)
                if resp.status_code == 200:
                    print(f"[OK] Ollama server ready after {i + 1}s")
                    return True, None
            except Exception:
                continue

        return False, "Ollama started but did not respond within 15 seconds"

    except Exception as e:
        return False, str(e)

def find_free_port(start_port=8080, max_attempts=50):
    """Find a free port starting from start_port, with conflict detection"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def create_pid_file(port):
    """Create PID file for process management"""
    global pid_file_path
    try:
        pid_file_path = os.path.join(tempfile.gettempdir(), f"natna_ui_{port}.pid")
        with open(pid_file_path, 'w') as f:
            f.write(f"{os.getpid()}\n{port}\n")
        print(f"📁 PID file created: {pid_file_path}")
    except Exception as e:
        print(f"[WARN] Warning: Could not create PID file: {e}")

def cleanup_resources():
    """Comprehensive resource cleanup on shutdown"""
    global httpd_server, pid_file_path, spawned_processes

    print("\n🧹 Performing graceful shutdown...")

    try:
        # LAYER 6: Ollama and Child Process Management
        print("  [AI] Stopping spawned processes (including Ollama)...")
        cleanup_child_processes()

        # Shutdown HTTP server
        if httpd_server:
            print("  [SIGNAL] Stopping HTTP server...")
            httpd_server.shutdown()
            httpd_server.server_close()

        # Clean up PID file
        if pid_file_path and os.path.exists(pid_file_path):
            print(f"  📁 Removing PID file: {pid_file_path}")
            os.unlink(pid_file_path)

        # Close any translator resources
        if 'translator' in globals() and hasattr(translator, 'cleanup'):
            print("  [AI] Cleaning up translator resources...")
            translator.cleanup()

        print("[OK] Graceful shutdown completed")
    except Exception as e:
        print(f"[WARN] Error during cleanup: {e}")

def cleanup_child_processes():
    """Clean up all spawned child processes including Ollama"""
    global spawned_processes
    import subprocess

    try:
        # Kill tracked spawned processes
        for proc in spawned_processes[:]:  # Copy list to avoid modification during iteration
            try:
                if proc.poll() is None:  # Process still running
                    print(f"    🔄 Terminating tracked process PID {proc.pid}...")
                    proc.terminate()

                    # Give process time to terminate gracefully
                    try:
                        proc.wait(timeout=3)
                        print(f"    [OK] Process {proc.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"    [WARN] Force killing process {proc.pid}...")
                        proc.kill()
                        proc.wait()
                        print(f"    [OK] Process {proc.pid} force killed")

                spawned_processes.remove(proc)
            except Exception as e:
                print(f"    [WARN] Error cleaning process {proc.pid}: {e}")
                spawned_processes.remove(proc)

        # Additional cleanup for any remaining Ollama processes
        print("    [SEARCH] Scanning for any remaining Ollama processes...")
        try:
            # Find Ollama processes
            result = subprocess.run(["pgrep", "-f", "ollama"],
                                  capture_output=True, text=True, check=False)

            if result.stdout.strip():
                ollama_pids = result.stdout.strip().split('\n')
                for pid in ollama_pids:
                    if pid:
                        print(f"    [AI] Terminating Ollama process {pid}...")
                        subprocess.run(["kill", "-TERM", pid], check=False)

                # Wait a moment for graceful shutdown
                time.sleep(2)

                # Force kill any remaining
                result = subprocess.run(["pgrep", "-f", "ollama"],
                                      capture_output=True, text=True, check=False)
                if result.stdout.strip():
                    remaining_pids = result.stdout.strip().split('\n')
                    for pid in remaining_pids:
                        if pid:
                            print(f"    ⚡ Force killing Ollama process {pid}...")
                            subprocess.run(["kill", "-9", pid], check=False)
            else:
                print("    [OK] No additional Ollama processes found")

        except Exception as e:
            print(f"    [WARN] Error during Ollama cleanup: {e}")

        # Clear port 11434 (Ollama default port)
        try:
            result = subprocess.run(["lsof", "-ti:11434"], capture_output=True, text=True, check=False)
            if result.stdout.strip():
                port_pids = result.stdout.strip().split('\n')
                for pid in port_pids:
                    if pid:
                        print(f"    [PLUG] Clearing port 11434, killing PID {pid}...")
                        subprocess.run(["kill", "-9", pid], check=False)
        except Exception as e:
            print(f"    [WARN] Error clearing port 11434: {e}")

    except Exception as e:
        print(f"  [WARN] Error during child process cleanup: {e}")

def handle_shutdown_signal(signum, frame):
    """Handle SIGTERM and SIGINT gracefully"""
    global shutdown_flag
    print(f"\n[STOP] Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True

def setup_signal_handlers():
    """Register signal handlers for graceful shutdown"""
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    atexit.register(cleanup_resources)
    print("[SHIELD] Signal handlers registered for graceful shutdown")

def track_spawned_process(process):
    """Track a spawned process for cleanup on shutdown"""
    global spawned_processes
    spawned_processes.append(process)
    print(f"[NOTE] Tracking spawned process PID {process.pid} for cleanup")

def run_server_loop(httpd):
    """Controlled server loop that can be gracefully interrupted"""
    global shutdown_flag
    print("[LAUNCH] Starting controlled server loop (monitoring for shutdown signals)...")

    # Memory monitoring for field laptops (log every 10 minutes)
    request_count = 0
    last_memory_check = time.time()
    memory_check_interval = 600  # 10 minutes

    try:
        while not shutdown_flag:
            # Use timeout to allow checking shutdown_flag regularly
            httpd.timeout = 1.0
            httpd.handle_request()
            request_count += 1

            # Periodic memory monitoring for field laptop optimization
            current_time = time.time()
            if current_time - last_memory_check >= memory_check_interval:
                print(f"[STATS] Field laptop check after {request_count} requests:")
                log_memory_status()

                # Aggressive garbage collection for field laptops every 10 minutes
                gc.collect()

                last_memory_check = current_time
                request_count = 0

    except KeyboardInterrupt:
        print("\n[KEY] Keyboard interrupt received")
        shutdown_flag = True
    except Exception as e:
        print(f"[ERROR] Server loop error: {e}")
        shutdown_flag = True
    finally:
        print("🔄 Exiting server loop")
        print("[STATS] Final memory status:")
        log_memory_status()

def main():
    global httpd_server, server_port, lan_ip

    # CRITICAL: Ensure singleton first to prevent process pile-up
    ensure_singleton()

    print("[SHIELD] Initializing BULLETPROOF NATNA Terminal UI with 8-Layer Defense System")
    print("[LOCK] Layer 0: Singleton Process Lock (NEW - prevents pile-up)")
    print("[STATS] Layer 1: Dynamic Port Management")
    print("[STOP] Layer 2: Signal-Based Graceful Shutdown")
    print("🧹 Layer 3: Resource Cleanup System")
    print("🔄 Layer 4: Controlled Server Loop")
    print("💓 Layer 5: Process Health Monitoring")
    print("[AI] Layer 6: Ollama Process Management")
    print("[LAUNCH] Layer 7: Maximum Resource Utilization (UPDATED)")

    # Apply maximum resource utilization
    system_resources = maximize_resource_utilization()
    log_memory_status()

    # Layer 1: Dynamic Port Management
    preferred_port = 8080
    port = find_free_port(preferred_port)

    if not port:
        print(f"[ERROR] No free ports found starting from {preferred_port}")
        return 1

    if port != preferred_port:
        print(f"[WARN] Port {preferred_port} in use, using port {port} instead")

    server_port = port

    # Layer 2: Signal-Based Graceful Shutdown
    setup_signal_handlers()

    # Layer 5: Process Health Monitoring
    create_pid_file(port)

    try:
        # Layer 4: Controlled Server Loop (Threading for Concurrent Requests)
        print(f"[SIGNAL] Starting threaded HTTP server on port {port}...")
        httpd_server = ThreadingHTTPServer(('', port), TerminalHandler)

        # Allow address reuse to prevent "Address already in use" errors
        httpd_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Detect LAN IP for network access (prefer WiFi/Ethernet over VPN tunnels)
        lan_ip = None
        try:
            import netifaces
            # Prefer physical interfaces: en0 (WiFi/macOS), en1, eth0, wlan0
            preferred = ['en0', 'en1', 'eth0', 'wlan0', 'en2', 'en3']
            for iface in preferred:
                try:
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        ip = addrs[netifaces.AF_INET][0].get('addr')
                        if ip and not ip.startswith('127.'):
                            lan_ip = ip
                            break
                except (ValueError, KeyError):
                    continue
        except ImportError:
            pass
        if not lan_ip:
            # Fallback: parse ifconfig for en0 (works without netifaces on macOS)
            try:
                import subprocess
                result = subprocess.run(['ifconfig', 'en0'], capture_output=True, text=True, timeout=3)
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('inet ') and not line.startswith('inet6'):
                        lan_ip = line.split()[1]
                        break
            except Exception:
                pass
        if not lan_ip:
            # Last resort: socket trick (may return VPN IP)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
                s.close()
            except Exception:
                pass

        print(f"\n[SUCCESS] NATNA Terminal UI ready!")
        print(f"[GLOBE] Local access:   http://localhost:{port}")
        if lan_ip:
            print(f"[MOBILE] Network access: http://{lan_ip}:{port}  (phone/tablet on same Wi-Fi)")
        print(f"[STOP] Press Ctrl+C for graceful shutdown\n")

        # Use controlled loop instead of serve_forever()
        run_server_loop(httpd_server)

    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[ERROR] Port {port} is already in use. Another instance may be running.")
            print("[TIP] Try using KILL_NATNA_PROCESSES.command to clean up processes, then launch NATNA again.")
            return 1
        else:
            print(f"[ERROR] Network error: {e}")
            return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1
    finally:
        print("\n🔄 NATNA Terminal UI shutdown complete.")

    return 0

if __name__ == "__main__":
    main()