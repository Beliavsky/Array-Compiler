@echo off
setlocal

cd /d "%~dp0"
set "MSGFILE=%TEMP%\array_compiler_commit_msg.txt"

git status
if errorlevel 1 goto :fail

git add .
if errorlevel 1 goto :fail

(
echo Wrap emitted Fortran lines and add column-limit regressions
echo.
echo - add a free-form Fortran line wrapper to the backend and apply it to ordinary module emission and bind^(C^) wrapper emission
echo - remove reliance on -ffree-line-length-none in strict gfortran -std=f2018 compile-backed tests
echo - add regression checks that emitted Fortran stays within 132 columns for representative long-line cases
echo - keep the current Python-to-Fortran test suite passing under strict compilation
) > "%MSGFILE%"

git commit -F "%MSGFILE%"
if errorlevel 1 goto :fail

del "%MSGFILE%" >nul 2>nul

git push
if errorlevel 1 goto :fail

echo.
echo Commit and push completed.
goto :eof

:fail
del "%MSGFILE%" >nul 2>nul
echo.
echo Batch file failed.
exit /b 1
