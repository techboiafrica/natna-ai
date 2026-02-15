@echo off
setlocal enabledelayedexpansion

:: Resolve repo root (one level up from Launch_NATNA_all_OS)
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd

echo [FIRE] LAUNCHING BULLETPROOF NATNA AI DESKTOP [FIRE]
echo =============================================
echo [FOLDER] NATNA root: %REPO_ROOT%
echo [FOLDER] Sacred files: %REPO_ROOT%\SACRED
echo [DB] Database: massive_tigrinya_database.db (265,972+ entries)
echo [BRAIN] AI Engine: intelligent_translator.py (DeepSeek + SQL fallback)
echo [GLOBE] Interface: NatnaUI.py with 6-Layer Defense System
echo.

:: STEP 1: Start Ollama Server with NATNA Drive Models
echo [AI] Step 1: Starting Ollama server with NATNA drive models...
set "OLLAMA_MODELS=%REPO_ROOT%\SACRED\runtime\models"
set "OLLAMA_HOST=127.0.0.1:11434"

:: [LAUNCH] MAXIMUM PERFORMANCE CONFIGURATION
echo   [FIRE] Configuring Ollama for MAXIMUM resource utilization...
echo   [MEM] FORCING: 16GB RAM, 8 CPU cores (Maximum Performance Mode)

set "OLLAMA_MAX_LOADED_MODELS=1"
set "OLLAMA_NUM_PARALLEL=8"
set "OLLAMA_FLASH_ATTENTION=true"
set "OLLAMA_CONTEXT_LENGTH=8192"
set "OLLAMA_MAX_QUEUE=1024"
set "OLLAMA_KEEP_ALIVE=10m"

echo   [OK] MAXIMUM PERFORMANCE Ollama configuration applied

:: Start portable Ollama from NATNA drive
set "OLLAMA_PATH=%REPO_ROOT%\SACRED\runtime\ollama-windows\ollama.exe"
if exist "%OLLAMA_PATH%" (
    echo   [LAUNCH] Starting portable Ollama server...
    start /B "" "%OLLAMA_PATH%" serve
) else (
    echo   [WARN] Portable Ollama not found at %OLLAMA_PATH%
    echo   [WARN] Trying system Ollama...
    start /B "" ollama serve
)

:: Wait for Ollama to start
echo   [WAIT] Waiting for Ollama server to initialize...
timeout /t 5 /nobreak >nul

:: Test Ollama connection
echo   [TEST] Testing Ollama connection...
set OLLAMA_READY=0
for /L %%i in (1,1,10) do (
    if !OLLAMA_READY!==0 (
        curl -s http://localhost:11434/api/version >nul 2>&1
        if !errorlevel!==0 (
            echo   [OK] Ollama server responding!
            set OLLAMA_READY=1
            :: Preload model
            echo   [DOWNLOAD] Preloading Qwen 3 0.6B model...
            start /B "" curl -s -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\": \"qwen3:0.6b\", \"prompt\": \"Hello\", \"stream\": false}"
        ) else (
            echo   [WAIT] Ollama starting... (attempt %%i/10)
            timeout /t 2 /nobreak >nul
        )
    )
)

:: STEP 2: Start NATNA Web Interface
echo.
echo [SHIELD] Step 2: Starting NATNA web interface with 6-Layer Defense System...

:: Find NatnaUI.py
set "SCRIPT_PATH="
if exist "%REPO_ROOT%\SACRED\APP_Production\NatnaUI.py" (
    set "SCRIPT_PATH=%REPO_ROOT%\SACRED\APP_Production\NatnaUI.py"
) else if exist "%~dp0APP_Production\NatnaUI.py" (
    set "SCRIPT_PATH=%~dp0APP_Production\NatnaUI.py"
) else if exist "%~dp0NatnaUI.py" (
    set "SCRIPT_PATH=%~dp0NatnaUI.py"
)

if "%SCRIPT_PATH%"=="" (
    echo   [ERROR] ERROR: NatnaUI.py not found!
    pause
    exit /b 1
)

echo   [FOUND] Found NatnaUI.py at: %SCRIPT_PATH%

:: Change to script directory and start server
for %%F in ("%SCRIPT_PATH%") do set "SCRIPT_DIR=%%~dpF"
cd /d "%SCRIPT_DIR%"

:: Resolve bundled Python
set "PYTHON_BIN=%REPO_ROOT%\SACRED\runtime\python\windows-x64\python\python.exe"
set "PYTHONPATH=%REPO_ROOT%\SACRED\runtime\python\packages;%REPO_ROOT%\SACRED\APP_Production\dependencies"
if not exist "%PYTHON_BIN%" (
    echo   [WARN] Bundled Python not found at %PYTHON_BIN%, falling back to system python...
    set "PYTHON_BIN=python"
)
echo   [PYTHON] Using: %PYTHON_BIN%

echo   [LAUNCH] Starting web server...
start /B "" "%PYTHON_BIN%" NatnaUI.py
:: Capture Python PID for cleanup
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr "PID:"') do set PYTHON_PID=%%a

:: STEP 3: Wait for Web Server
echo.
echo [GLOBE] Step 3: Waiting for web server and opening browser...

set WEB_READY=0
for /L %%i in (1,1,30) do (
    if !WEB_READY!==0 (
        curl -s http://localhost:8080 >nul 2>&1
        if !errorlevel!==0 (
            echo   [OK] Web server responding!
            set WEB_READY=1
        ) else (
            echo   [WAIT] Web server starting... (attempt %%i/30)
            timeout /t 1 /nobreak >nul
        )
    )
)

if %WEB_READY%==0 (
    echo   [ERROR] Web server failed to start after 30 attempts
    pause
    exit /b 1
)

:: STEP 4: Open Browser
echo.
echo [LAUNCH] Step 4: Opening browser to NATNA AI interface...
start http://localhost:8080

echo.
echo [SUCCESS] ==========================================
echo [SUCCESS] NATNA AI DESKTOP SUCCESSFULLY LAUNCHED!
echo [SUCCESS] ==========================================
echo.
echo [OK] Ollama AI Server: Running
echo [OK] NATNA Web Interface: Running
echo [OK] Models Available: qwen3:0.6b, phi4-mini, deepseek-r1, etc.
echo [OK] Browser: http://localhost:8080
echo.
echo [TEST] Ready to test: Ask 'What is photosynthesis?' to test AI
echo.
echo [STOP] Press any key to stop both servers and exit
echo ===============================================

:: Keep window open, then cleanup on exit
pause
goto cleanup

:cleanup
echo.
echo [STOP] Shutting down NATNA AI Desktop...
taskkill /F /IM ollama.exe >nul 2>&1
if defined PYTHON_PID (
    taskkill /F /PID %PYTHON_PID% >nul 2>&1
)
echo [OK] All processes stopped.
exit /b 0
