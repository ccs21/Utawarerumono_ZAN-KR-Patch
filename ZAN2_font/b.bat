@echo off
setlocal enabledelayedexpansion

REM ---- edit if your VS path differs ----
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

call %VCVARS% x64 >nul
if errorlevel 1 (
  echo [ERR] vcvarsall failed
  exit /b 1
)

echo.
echo ===== BUILD: x64 (dinput8 proxy + io log) =====

set OUTDIR=out\x64
if not exist %OUTDIR% mkdir %OUTDIR%

set MH=build\MinHook
set INCS=/I "%MH%\include" /I "%MH%\src"
set SRCS=dinput8_hook.cpp ^
  "%MH%\src\hook.c" ^
  "%MH%\src\buffer.c" ^
  "%MH%\src\trampoline.c" ^
  "%MH%\src\hde\hde64.c"

cl /nologo /O2 /MT /LD /EHsc %INCS% %SRCS% ^
  /link /NOLOGO ^
  /DEF:dinput8.def ^
  /OUT:%OUTDIR%\dinput8.dll ^
  user32.lib kernel32.lib shlwapi.lib

if errorlevel 1 (
  echo [ERR] build failed
  exit /b 1
)

echo [OK] %OUTDIR%\dinput8.dll
echo DONE. Copy out\x64\dinput8.dll next to the game exe (same folder).