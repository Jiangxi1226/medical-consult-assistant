@echo off
setlocal
title Medical Assistant - Multi-Agent Triage
cd /d "%~dp0"

if not exist .env (
  echo [ERROR] .env not found.
  echo         Copy .env.example and fill in LLM_API_KEY, then retry.
  pause
  exit /b 1
)

if not exist web\index.html (
  echo [WARN] web\index.html is missing.
)

set PORT=8001

echo.
echo  ==================================================
echo    Medical Assistant is starting...
echo    URL: http://localhost:8001   (port 8001; project-1 uses 8000)
echo    Browser will open in ~3 seconds.
echo    Keep this window open. Press Ctrl+C to stop.
echo  ==================================================
echo.

start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8001"

set PYTHONUTF8=1
python app.py

pause
