@echo off
setlocal

cd /d "%~dp0"
set "MSGFILE=%TEMP%\array_compiler_commit_msg.txt"

git status
if errorlevel 1 goto :fail

git add .
if errorlevel 1 goto :fail

(
echo Add semantic Python-vs-Fortran regressions and test cleanup
echo.
echo - add end-to-end semantic regressions for xbs.py, xamerican_options.py, and xbs_monte_carlo.py
echo - compare deterministic outputs directly and Monte Carlo outputs with statistical tolerances
echo - fix explicit-step range lowering so Python exclusive stop semantics are preserved in generated Fortran loops
echo - factor compile/run helpers into tests\test_utils.py
echo - clean up temporary compiled executables automatically with a context-managed test helper
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
