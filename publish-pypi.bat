@echo off
REM Uploads mdcx to PyPI. See PUBLISHING.md for the account and token steps.
REM
REM   publish-pypi.bat          uploads to PyPI
REM   publish-pypi.bat --test   uploads to TestPyPI first (recommended)

setlocal
cd /d "%~dp0"

set PYTHON=..\.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

echo.
echo === Rebuilding the distributions ===
if exist dist rmdir /s /q dist
"%PYTHON%" -m build
if errorlevel 1 exit /b 1

echo.
echo === Validating them ===
"%PYTHON%" -m twine check dist/*
if errorlevel 1 (
    echo Validation failed. Nothing was uploaded.
    exit /b 2
)

echo.
if /I "%~1"=="--test" (
    echo === Uploading to TestPyPI ===
    echo Username is __token__ and the password is the token.
    "%PYTHON%" -m twine upload --repository testpypi dist/*
    echo.
    echo Check it at https://test.pypi.org/project/mdcx/
    exit /b %ERRORLEVEL%
)

echo === Uploading to PyPI ===
echo Username is __token__ and the password is the token.
echo A published version cannot be replaced.
"%PYTHON%" -m twine upload dist/*
if errorlevel 1 exit /b 1

echo.
echo Done. The package is at https://pypi.org/project/mdcx/
echo Anyone can now run:  pip install mdcx
