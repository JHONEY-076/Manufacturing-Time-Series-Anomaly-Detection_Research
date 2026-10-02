@echo off
cd /d "%~dp0"
if not exist ".venv-api\Scripts\python.exe" (
    echo Missing Python environment. See fmd_api\README.md.
    pause
    exit /b 1
)
".venv-api\Scripts\python.exe" "tools\research11_service.py" stop
if errorlevel 1 pause
