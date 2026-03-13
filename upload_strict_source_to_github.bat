@echo off
setlocal EnableExtensions

rem Upload only the strictly necessary Array-Compiler source files.
rem Excludes tests, example programs, docs, generated files, outputs, and temp files.

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
    set "MESSAGE=Update core compiler/runtime sources, benchmark tooling, R/Python translation paths, and README"
)
if "%REMOTE%"=="" set "REMOTE=origin"

echo Clearing current staged set...
git reset >nul
if errorlevel 1 exit /b 1

echo Staging strict source whitelist...
for %%F in (
    ".gitignore"
    "README.md"
    "pyproject.toml"
    "benchmarks.toml"
    "x2f.py"
    "x2f_batch.py"
    "x2f_bench.py"
    "x2p.py"
    "x2p_batch.py"
    "xpyannotate.py"
    "xpycheck.py"
    "xrcheck.py"
    "upload_strict_source_to_github.bat"
    "array_compiler"
) do (
    if exist %%~F git add -- %%~F
    if errorlevel 1 exit /b 1
)

echo Excluded by design: tests/, example programs, docs/, scripts/, generated files, outputs, and temp files.

echo Staged files:
git diff --cached --name-only

git diff --cached --quiet
if not errorlevel 1 (
    echo no strict source changes staged
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
