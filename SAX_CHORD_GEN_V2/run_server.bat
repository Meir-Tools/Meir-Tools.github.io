@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo מפעיל שרת בתיקייה: %CD%
start "" "http://localhost:8000/"
python -m http.server 8000
pause