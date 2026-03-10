@echo off
setlocal
cd /d c:\python\Array-Compiler

git status
git add .

set MSGFILE=%TEMP%\array_compiler_commit_msg.txt
> "%MSGFILE%" echo Remove temporary scratch files from the repo
>> "%MSGFILE%" echo.
>> "%MSGFILE%" echo - delete tracked _tmp* and temp* scratch artifacts that do not belong in the repository
>> "%MSGFILE%" echo - update .gitignore to keep temporary annotator and scratch files out of future commits

git commit -F "%MSGFILE%"
if errorlevel 1 goto :cleanup

git push

:cleanup
del "%MSGFILE%" >nul 2>nul
endlocal
