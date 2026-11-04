@echo off
rem Launch sync_push.py (logic lives in Python to avoid GBK/UTF-8 issues)
"C:\Users\Admin\.workbuddy\binaries\python\versions\3.13.12\python.exe" "%~dp0sync_push.py"
echo.
pause
