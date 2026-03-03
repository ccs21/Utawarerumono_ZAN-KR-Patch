@echo off
setlocal

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

if not exist %VCVARS% (
  echo [ERROR] vcvarsall.bat not found. Edit b_dinput8.bat and set correct path.
  exit /b 1
)

call %VCVARS% x64 >nul
if errorlevel 1 (
  echo [ERROR] Failed to init MSVC environment.
  exit /b 1
)

set PROJ=%~dp0
set MH=%PROJ%build\MinHook

if not exist "%MH%\include\MinHook.h" (
  echo [ERROR] MinHook not found at: %MH%
  exit /b 1
)

if not exist "%PROJ%dinput8_hook.cpp" (
  echo [ERROR] dinput8_hook.cpp not found
  exit /b 1
)

if not exist "%PROJ%dinput8.def" (
  echo [ERROR] dinput8.def not found
  exit /b 1
)

if not exist "%PROJ%out" mkdir "%PROJ%out"
del /q "%PROJ%out\dinput8.dll" 2>nul

echo ===== BUILD: x64 (dinput8) =====

cl /nologo /O2 /MT /LD /EHsc ^
  "%PROJ%dinput8_hook.cpp" ^
  "%MH%\src\hook.c" ^
  "%MH%\src\buffer.c" ^
  "%MH%\src\trampoline.c" ^
  "%MH%\src\hde\hde64.c" ^
  /I "%MH%\include" /I "%MH%\src" ^
  /link /OUT:"%PROJ%out\dinput8.dll" /DEF:"%PROJ%dinput8.def" ^
  kernel32.lib user32.lib

if errorlevel 1 (
  echo [ERROR] build failed
  exit /b 1
)

echo [OK] out\dinput8.dll
echo Copy out\dinput8.dll next to the game exe as dinput8.dll
endlocal