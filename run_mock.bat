@echo off
REM ===================================================================
REM  VisionAsk — UI-only mock server (no ML models, no deps needed)
REM  Use this to test the frontend without running BLIP-2/Grounding DINO.
REM ===================================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   VisionAsk - Mock Server (UI testing only)
echo ============================================================
echo.
echo [*] Starting mock server on http://localhost:8000/ ...
echo [*] Returns fake answers - no real model inference.
echo [*] Press Ctrl+C to stop.
echo.

python mock_server.py

echo.
echo [*] Server stopped.
pause
