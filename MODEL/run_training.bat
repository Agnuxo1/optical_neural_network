@echo off
setlocal EnableDelayedExpansion

rem Resolve project root (parent of this script's folder)
for %%I in ("%~dp0..") do set ROOT=%%~fI
set EXE=%ROOT%\build\optical_digit.exe
set DATA=%ROOT%\digit-recognizer

rem Configurables (edit if you want)
set EPOCHS=2000
set BATCH=1024
set LR=0.002

rem Make a timestamped submission filename
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set TS=%%i
set SUB=%DATA%\submission_!TS!.csv

echo [RUN] Project root: %ROOT%
echo [RUN] Executable   : %EXE%
echo [RUN] Data folder  : %DATA%

if not exist "%EXE%" (
  echo [ERROR] No se encuentra el ejecutable: %EXE%
  echo        Compila primero (carpeta build) y vuelve a intentarlo.
  pause
  exit /b 1
)

if not exist "%DATA%\train.csv" (
  echo [ERROR] No se encuentra train.csv en %DATA%
  pause
  exit /b 1
)
if not exist "%DATA%\test.csv" (
  echo [ERROR] No se encuentra test.csv en %DATA%
  pause
  exit /b 1
)

echo [RUN] Entrenando %EPOCHS% epochs (batch=%BATCH%, lr=%LR%)...
"%EXE%" --train "%DATA%\train.csv" --test "%DATA%\test.csv" --epochs %EPOCHS% --batch %BATCH% --lr %LR% --submission "%SUB%"
set ERR=%ERRORLEVEL%

if not %ERR%==0 (
  echo [FAIL] El entrenamiento o la inferencia han fallado (c=%ERR%).
  pause
  exit /b %ERR%
)

echo [DONE] Submission guardada en: %SUB%
pause
exit /b 0

