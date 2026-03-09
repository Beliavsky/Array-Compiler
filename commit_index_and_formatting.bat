@echo off
setlocal

cd /d "%~dp0"

git status
if errorlevel 1 exit /b %errorlevel%

git add .
if errorlevel 1 exit /b %errorlevel%

set MSGFILE=%TEMP%\array_compiler_commit_msg.txt
> "%MSGFILE%" (
  echo Fix integer index lowering and improve default Fortran output formatting
  echo.
  echo - treat non-integer generated array indices as a transpiler bug and fix
  echo   the xar_sim_acf_pacf.py index path to emit standard-conforming Fortran
  echo - stop printing a timing summary when Fortran compilation fails in x2f.py
  echo - preserve top-level runtime setup such as numpy.set_printoptions in
  echo   generated run_main code
  echo - add cleaner default print settings for generated Fortran output when
  echo   the Python source does not call numpy.set_printoptions
  echo - carry width and precision through formatted float output so printed
  echo   tables have aligned decimal points
  echo - extend regressions for xar_sim_acf_pacf.py, build-failure CLI behavior,
  echo   and emitted integer-index sections
)

git commit -F "%MSGFILE%"
if errorlevel 1 exit /b %errorlevel%

git push
if errorlevel 1 exit /b %errorlevel%

del "%MSGFILE%" >nul 2>nul

endlocal
