@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   User Insight Bot
echo ============================================================
echo.

echo [1/3] FastAPI (:8000)
start "API" cmd /k "cd /d %cd% && uv run python run_api.py"

echo [2/3] Gradio (:7860)
start "Gradio" cmd /k "cd /d %cd% && uv run python app.py"

echo [3/3] Vue (:5173)
if exist "%cd%\frontend\node_modules" (
    start "Vue" cmd /k "cd /d %cd%\frontend && npm run dev"
) else (
    start "Vue" cmd /k "cd /d %cd%\frontend && npm install && npm run dev"
)

echo.
echo Waiting for services to become ready (first run may take a few minutes)...
set /a tries=0

:wait_loop
set /a tries+=1
curl -s -o nul http://localhost:8000/health && curl -s -o nul http://localhost:7860 && curl -s -o nul http://localhost:5173 && goto all_ready
if %tries% GEQ 150 goto wait_timeout
<nul set /p "=."
timeout /t 2 /nobreak >nul
goto wait_loop

:all_ready
echo.
echo All services are up. Opening browser...
start "" http://localhost:5173
echo.
echo   Vue AI shop:     http://localhost:5173
echo   Gradio console:  http://localhost:7860
echo   Swagger docs:    http://localhost:8000/docs
echo.
echo This window will close in 5 seconds. Close the service windows to stop.
timeout /t 5 /nobreak >nul
exit /b 0

:wait_timeout
echo.
echo [!] Timeout waiting for services. Not responding:
curl -s -o nul http://localhost:8000/health || echo     - FastAPI :8000
curl -s -o nul http://localhost:7860 || echo     - Gradio  :7860
curl -s -o nul http://localhost:5173 || echo     - Vue     :5173
echo Check the service windows for errors.
pause
exit /b 1
