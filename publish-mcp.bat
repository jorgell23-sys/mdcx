@echo off
REM Publishes the server to the MCP Registry.
REM
REM Authentication opens a browser page where a code must be entered. Two
REM deadlines apply and both are short. The device code expires fifteen minutes
REM after it appears, and the registry token that the sign-in returns expires
REM five minutes after that. Signing in and publishing therefore have to run in
REM one go, which is why this script does both, and the code has to be entered
REM as soon as it is shown rather than later.

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
