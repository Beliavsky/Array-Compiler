@echo off
setlocal EnableExtensions

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
    set "MESSAGE=Update core compiler frontends, Fortran backend/runtime, CLI tools, Python and R checkers, and README"
)
if "%REMOTE%"=="" set "REMOTE=origin"

echo Clearing current staged set...
git reset >nul
if errorlevel 1 exit /b 1

echo Staging core project files...
for %%F in (
    ".gitignore"
    "README.md"
    "pyproject.toml"
    "corpus.toml"
    "fortran_style.txt"
    "x2f.py"
    "x2f_batch.py"
    "x2p.py"
    "x2p_batch.py"
    "xpyannotate.py"
    "xpycheck.py"
    "xrcheck.py"
    "upload_core_to_github.bat"
    "array_compiler"
) do (
    if exist %%~F git add -- %%~F
    if errorlevel 1 exit /b 1
)

echo Staged files:
git diff --cached --name-only

git diff --cached --quiet
if not errorlevel 1 (
    echo no core project changes staged
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
