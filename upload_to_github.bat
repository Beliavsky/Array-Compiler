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

echo Staging curated project files...
for %%F in (
    ".gitignore"
    "README.md"
    "pyproject.toml"
    "corpus.toml"
    "fortran_style.txt"
    "runall.bat"
    "x2f.py"
    "x2f_batch.py"
    "x2p.py"
    "x2p_batch.py"
    "xpyannotate.py"
    "xpycheck.py"
    "xrcheck.py"
    "upload_to_github.bat"
) do (
    if exist %%~F git add -- %%~F
    if errorlevel 1 exit /b 1
)

if exist "array_compiler" git add -- "array_compiler"
if errorlevel 1 exit /b 1

if exist "scripts" git add -- "scripts"
if errorlevel 1 exit /b 1

if exist "tests" git add -- "tests"
if errorlevel 1 exit /b 1

echo Staging top-level source files with project prefixes...
for %%F in (x*.py x*.r x*.R x*.f90 x*.bat) do (
    if exist "%%~fF" call :maybe_stage "%%~nxF"
)

echo Staged files:
git diff --cached --name-only

git diff --cached --quiet
if not errorlevel 1 (
    echo no curated project changes staged
    exit /b 0
)

echo Committing...
echo Commit message: %MESSAGE%
git commit -m "%MESSAGE%"
if errorlevel 1 exit /b 1

echo Pushing to %REMOTE%...
git push %REMOTE% HEAD
if errorlevel 1 (
    echo Push failed because the remote branch is ahead of this checkout.
    echo Suggested next steps:
    echo   git fetch %REMOTE%
    echo   git rebase %REMOTE%/main
    echo   git push %REMOTE% HEAD
    exit /b 1
)

echo Upload complete.
exit /b 0

:maybe_stage
set "FILE=%~1"
set "SKIP="

echo(!FILE!| findstr /R /I "_p\.py$ _p\.f90$ _annotated\.py$ \.warnings\.txt$ ^temp ^tmp ^_tmp ^commit.*\.bat$ \.exe$ \.obj$ \.o$ \.mod$" >nul
if not errorlevel 1 set "SKIP=1"

if defined SKIP exit /b 0

git add -- "!FILE!"
if errorlevel 1 exit /b 1
exit /b 0
