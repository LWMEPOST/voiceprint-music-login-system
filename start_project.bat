@echo off
setlocal
cd /d "%~dp0"

powershell -ExecutionPolicy Bypass -File "%~dp0start_project.ps1"

if errorlevel 1 (
    echo.
    echo Startup failed.
    pause
)

endlocal
