@echo off
REM Publishes the server to the MCP Registry.
REM
REM Authentication opens a browser page where a code must be entered. The code
REM expires after fifteen minutes, so it is entered right after it appears.

setlocal
cd /d "%~dp0"

set PUBLISHER=%~dp0mcp-publisher.exe
if not exist "%PUBLISHER%" set PUBLISHER=mcp-publisher.exe

echo.
echo === Validating server.json ===
"%PUBLISHER%" validate
if errorlevel 1 exit /b 1

echo.
echo === Signing in to GitHub ===
echo A code will appear below. Open https://github.com/login/device,
echo enter it, and authorise the application.
echo.
"%PUBLISHER%" login github
if errorlevel 1 (
    echo.
    echo Sign-in failed. The code expires after fifteen minutes; run this again.
    exit /b 1
)

echo.
echo === Publishing ===
"%PUBLISHER%" publish
if errorlevel 1 exit /b 1

echo.
echo Published. Check it with:
echo   curl "https://registry.modelcontextprotocol.io/v0/servers?search=markdown-document-search"
