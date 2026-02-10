@echo off
:: NATNA Process Cleanup Script - Windows

echo [CLEANUP] Killing all NATNA processes...

:: Step 1: Kill Python server and Ollama processes
echo [KILL] Killing NATNA processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *NatnaUI*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *intelligent_translator*" 2>nul
taskkill /F /IM ollama.exe 2>nul

:: Step 2: Clear ports
echo [PORTS] Clearing ports...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /PID %%a /F 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081') do taskkill /PID %%a /F 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8082') do taskkill /PID %%a /F 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :11434') do taskkill /PID %%a /F 2>nul

:: Step 3: Wait a moment for cleanup
timeout /t 2 /nobreak >nul

:: Step 4: Verify cleanup
echo [CHECK] Verifying cleanup...
echo.
echo [OK] CLEANUP COMPLETE
echo.
echo [RESTART] Now use NATNA_Launch_Windows.bat to restart
echo.

pause
