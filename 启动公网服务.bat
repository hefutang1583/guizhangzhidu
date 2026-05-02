@echo off
chcp 65001 >nul
title 校园规章制度智能咨询助手 - 公网服务
echo ══════════════════════════════════════════════════
echo   校园规章制度智能咨询助手 - 一键启动公网服务
echo ══════════════════════════════════════════════════
echo.

echo [1/2] 启动 FastAPI 后端服务...
start /B "" "venv\Scripts\python.exe" -X utf8 -m uvicorn main:app --host 0.0.0.0 --port 8000
timeout /t 8 /nobreak >nul

echo [2/2] 启动 Cloudflare 隧道（创建公网地址）...
echo.
echo ⏳ 正在获取公网地址，请稍候...
echo.
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000

echo.
echo 服务已停止。
pause
