@echo off
REM ===================================================================
REM  VisionAsk — One-time setup
REM  Creates a virtual environment and installs all dependencies.
REM  Only needs to be run ONCE (or if you delete the venv folder).
REM ===================================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   VisionAsk - First-time setup
echo ============================================================
echo.

REM Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python is not installed or not on PATH.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     Make sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo [*] Creating virtual environment in .\venv ...
python -m venv venv
if errorlevel 1 (
    echo [!] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [*] Activating virtual environment...
call "venv\Scripts\activate.bat"

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [*] Installing PyTorch with CUDA 12.1 (NVIDIA GPU)...
echo     If you don't have an NVIDIA GPU, press Ctrl+C and edit this
echo     script to use the CPU-only build instead.
echo.
pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [!] PyTorch install failed.
    pause
    exit /b 1
)

echo.
echo [*] Installing remaining dependencies...
pip install fastapi uvicorn python-multipart pillow numpy transformers accelerate groundingdino-py "protobuf>=4.21,<5"
if errorlevel 1 (
    echo [!] Dependency install failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete!
echo   Now double-click run.bat to start the server.
echo ============================================================
echo.
pause
