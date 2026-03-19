@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   BioNexus ML - Dynamic Dashboard
echo ==========================================
echo.

echo [INFO] Python check passed.
taskkill /f /im python.exe /t >nul 2>&1
echo [INFO] Existing Python processes cleaned up.

:: Check app file
if not exist "app_streamlit.py" (
    echo [ERROR] app_streamlit.py not found!
    pause
    exit /b
)

:: Setup virtual environment
if not exist ".venv" (
echo Creating virtual environment...
python -m venv .venv
)

call .venv\Scripts\activate

echo.
echo [1/2] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies!
    echo.
    echo [TIP] If you see "Access is denied", a Python process might be holding onto files.
    echo Try manually deleting the ".venv" folder and then run this script again.
    pause
    exit /b
)

echo.
echo [2/2] Launching Streamlit Dashboard...
start http://localhost:8501
streamlit run app_streamlit.py

if %errorlevel% neq 0 (
echo.
echo [ERROR] Failed to start Streamlit.
)

echo.
pause
