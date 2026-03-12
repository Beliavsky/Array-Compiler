@echo off
setlocal EnableExtensions EnableDelayedExpansion

where git >nul 2>nul
if errorlevel 1 (
    echo git not found in PATH
    exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo current directory is not a git repository
    exit /b 1
)

set "MESSAGE=%~1"
set "REMOTE=%~2"
if "%MESSAGE%"=="" (
    set "MESSAGE=Add R-to-Python and R-to-Fortran workflow updates, batch tools, Python and R checkers, improved annotation comments, stricter Fortran code generation, working translated timer support, and README documentation updates"
)
if "%REMOTE%"=="" set "REMOTE=origin"

echo Clearing current staged set...
git reset >nul
if errorlevel 1 exit /b 1

echo Staging core source directories...
call :stage_path ".gitignore"
call :stage_path "README.md"
call :stage_path "pyproject.toml"
call :stage_path "corpus.toml"
call :stage_path "fortran_style.txt"
call :stage_path "runall.bat"
call :stage_path "x2f.py"
call :stage_path "x2f_batch.py"
call :stage_path "x2p.py"
call :stage_path "x2p_batch.py"
call :stage_path "xpyannotate.py"
call :stage_path "xpycheck.py"
call :stage_path "xrcheck.py"
call :stage_path "upload_to_github.bat"
call :stage_path "array_compiler"
call :stage_path "scripts"

echo Staging root source files...
for %%F in (*.py *.r *.R *.f90 *.bat) do (
    if exist "%%~fF" call :maybe_stage_root "%%~nxF"
)

echo Staged files:
git diff --cached --name-only

git diff --cached --quiet
if not errorlevel 1 (
    echo no project source changes staged
    exit /b 0
)

echo Committing...
echo Commit message: %MESSAGE%
git commit -m "%MESSAGE%"
if errorlevel 1 exit /b 1

echo Pushing to %REMOTE%...
git push %REMOTE% HEAD
if errorlevel 1 exit /b 1

echo Upload complete.
exit /b 0

:stage_path
if exist "%~1" (
    git add -- "%~1"
    if errorlevel 1 exit /b 1
)
exit /b 0

:maybe_stage_root
set "FILE=%~1"
set "SKIP="

if /I "!FILE!"=="runall.bat" set "SKIP=1"
if /I "!FILE!"=="upload_to_github.bat" set "SKIP=1"
if /I "!FILE!"=="x2f.py" set "SKIP=1"
if /I "!FILE!"=="x2f_batch.py" set "SKIP=1"
if /I "!FILE!"=="x2p.py" set "SKIP=1"
if /I "!FILE!"=="x2p_batch.py" set "SKIP=1"
if /I "!FILE!"=="xpyannotate.py" set "SKIP=1"
if /I "!FILE!"=="xpycheck.py" set "SKIP=1"
if /I "!FILE!"=="xrcheck.py" set "SKIP=1"

echo(!FILE!| findstr /R /I "_p\.py$ _p\.f90$ _annotated\.py$ \.warnings\.txt$ ^temp ^tmp ^_tmp ^commit.*\.bat$ \.exe$ \.obj$ \.o$ \.mod$" >nul
if not errorlevel 1 set "SKIP=1"

if defined SKIP exit /b 0

git add -- "!FILE!"
if errorlevel 1 exit /b 1
exit /b 0
