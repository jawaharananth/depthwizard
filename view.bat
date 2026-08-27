@echo off
REM Opens the DepthWizard 3D viewer in your default browser, GPU-accelerated.
REM
REM A local web server is required rather than opening index.html directly:
REM ES module imports and texture loading are blocked under the file://
REM protocol by browser CORS rules.

cd /d "%~dp0viewer"
echo Starting viewer at http://localhost:8800
echo Press Ctrl+C in this window to stop.
start "" http://localhost:8800/index.html
python -m http.server 8800
