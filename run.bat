@echo off
REM ===================================================================
REM  VisionAsk — Windows launcher
REM  Double-click this file to start the server.
REM ===================================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   VisionAsk - BLIP-2 + Grounding DINO
echo ============================================================
echo.

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo [!] No virtual environment found - using system Python.
    echo     ^(If you get import errors, run setup.bat first^)
)

echo.
echo [*] Starting server on http://localhost:8000/ ...
echo [*] First run downloads ~5-6 GB of model weights (one-time).
echo [*] Press Ctrl+C to stop.
echo.

uvicorn backend:app --host 0.0.0.0 --port 8000

echo.
echo [*] Server stopped.
pause
