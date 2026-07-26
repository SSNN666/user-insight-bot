@echo off
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
echo Wait 10s then open:
echo   http://localhost:8000/docs
echo   http://localhost:7860
echo   http://localhost:5173
pause
