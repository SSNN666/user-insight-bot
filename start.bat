@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   User Insight Bot
echo ============================================================
echo.

rem ── Docker 基础设施(PG 会话库 + Milvus 三件套)────────────────
rem .env 走生产开关(SESSION_STORE=postgres / milvus-remote)时必起;
rem 若切回默认(json/milvus-lite)本步自动幂等跳过(容器已在跑则 no-op)。
set "DCK=E:\Docker\Docker\resources\bin\docker.exe"
set "DD=E:\Docker\Docker\Docker Desktop.exe"

echo [0/3] Docker infra (Postgres + Milvus) ...
call :ensure_docker_engine
call :start_pg
call :start_milvus

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

:ensure_docker_engine
"%DCK%" info >nul 2>&1
if not errorlevel 1 goto :eof
echo   Docker engine not running, starting Docker Desktop...
start "" "%DD%"
set /a t=0
:dd_loop
set /a t+=1
"%DCK%" info >nul 2>&1 && goto :dd_ok
if %t% GEQ 60 goto :dd_timeout
timeout /t 2 /nobreak >nul
goto :dd_loop
:dd_ok
echo   Docker engine ready.
goto :eof
:dd_timeout
echo   [!!] Docker engine not ready after ~2min.
echo        API will fail-fast on PG session store. Start Docker Desktop manually.
goto :eof

:start_pg
"%DCK%" start pg-session >nul 2>&1
if errorlevel 1 (
    echo   [!!] pg-session container missing. Create once with:
    echo        "%DCK%" run -d --name pg-session -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
    goto :eof
)
echo   pg-session started, waiting for readiness...
set /a t=0
:pg_loop
set /a t+=1
"%DCK%" exec pg-session pg_isready -U postgres >nul 2>&1 && goto :pg_ok
if %t% GEQ 15 goto :pg_timeout
timeout /t 2 /nobreak >nul
goto :pg_loop
:pg_ok
echo   PostgreSQL ready.
goto :eof
:pg_timeout
echo   [!!] pg-session not accepting connections yet (API will tell you if needed).
goto :eof

:start_milvus
"%DCK%" compose -f "%~dp0docker-compose.milvus.yml" start >nul 2>&1
if errorlevel 1 (
    echo   [!!] Milvus stack containers missing. Create once with:
    echo        "%DCK%" compose -f "%~dp0docker-compose.milvus.yml" up -d
    goto :eof
)
echo   Milvus stack (etcd/MinIO/Milvus) started.
goto :eof
