@echo off
setlocal

cd /d "%~dp0"

git status
if errorlevel 1 goto :error

git add .
if errorlevel 1 goto :error

set "msgfile=%TEMP%\array_compiler_commit_mixtures_runtime_%RANDOM%.txt"
(
  echo Improve NumPy-to-Fortran runtime support for mixture models
  echo.
  echo - fix column-broadcast lowering in the Python frontend for patterns such as
  echo   log_prob - amax[:, None], log_prob - log_norm[:, None], and
  echo   diff * resp[:, j][:, None]
  echo - add matching Fortran runtime helpers ac_sub_col and ac_mul_col
  echo - replace the fragile matrix inverse path with a pivoted Gauss-Jordan
  echo   implementation
  echo - replace the old slogdet path with LU-based helpers for sign and
  echo   log-absolute-determinant
  echo - add a compile-and-run regression for xfit_mix_mv.py against Python on a
  echo   synthetic dataset with tolerance-based numeric checks
  echo - keep the full Array-Compiler test suite passing
) > "%msgfile%"

git commit -F "%msgfile%"
set "rc=%errorlevel%"
del "%msgfile%" >nul 2>nul
if not "%rc%"=="0" goto :error

git push
if errorlevel 1 goto :error

echo Commit and push completed.
exit /b 0

:error
echo Batch file failed.
exit /b 1
