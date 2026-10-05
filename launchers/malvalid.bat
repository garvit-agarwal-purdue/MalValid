@echo off
rem MalValid launcher for Windows: double-click this file.
rem It sets MalValid up on first use (needs internet, a few minutes), starts the MalValid web UI on this
rem computer only (127.0.0.1) and opens your browser on it, already signed in.
rem Close this window to stop MalValid. Ctrl+C also stops it; Windows then asks "Terminate batch job (Y/N)?",
rem and either answer closes MalValid. The logic is in launch.ps1 next to this file.
setlocal EnableExtensions DisableDelayedExpansion
set "MG_PS1=%~dp0launch.ps1"
if not exist "%MG_PS1%" (
  echo malvalid: ERROR: launch.ps1 is missing next to malvalid.bat.
  echo If you opened malvalid.bat from inside the ZIP file, first extract the whole ZIP
  echo ^(right-click it, Extract All^), then double-click malvalid.bat in the extracted launchers folder.
  pause
  exit /b 1
)
title malvalid
rem Full path: a damaged PATH must not hide Windows PowerShell.
set "MG_PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%MG_PS%" set "MG_PS=powershell.exe"
"%MG_PS%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%MG_PS1%" %*
set "MG_RC=%ERRORLEVEL%"
rem Stopped with Ctrl+C (STATUS_CONTROL_C_EXIT or 130): a normal way to quit, no pause (if cmd asked
rem "Terminate batch job (Y/N)?" and you answered N, this line is reached and exits quietly).
if "%MG_RC%"=="-1073741510" goto done
if "%MG_RC%"=="130" goto done
if not "%MG_RC%"=="0" (
  echo.
  echo MalValid stopped with exit code %MG_RC%. The messages above say why.
  pause
)
:done
endlocal & exit /b %MG_RC%
