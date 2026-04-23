@echo off
cd /d D:\Work\UlisseBrain

if not exist logs mkdir logs

echo [%date% %time%] Starting Ulisse Brain... >> logs\startup.log

start "Ulisse Backend" /min cmd /k "python webapp\backend\app.py >> logs\backend.log 2>&1"
echo [%date% %time%] Backend started. >> logs\startup.log

timeout /t 5 /nobreak > nul

start "Ulisse Sync" /min cmd /k "python scripts\sync_watcher.py >> logs\sync.log 2>&1"
echo [%date% %time%] Sync watcher started. >> logs\startup.log

echo.
echo ==========================================
echo   ULISSE BRAIN is starting up...
echo   Backend log:  logs\backend.log
echo   Watcher log:  logs\sync.log
echo ==========================================
echo.
