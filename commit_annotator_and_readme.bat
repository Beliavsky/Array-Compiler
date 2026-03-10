@echo off
setlocal
cd /d c:\python\Array-Compiler

git status
git add .

set MSGFILE=%TEMP%\array_compiler_commit_msg.txt
> "%MSGFILE%" echo Extend xpyannotate and document the annotator in README
>> "%MSGFILE%" echo.
>> "%MSGFILE%" echo - add conservative parameter-intent analysis to the Python annotator and make it optional with --no-infer-intent
>> "%MSGFILE%" echo - make Final promotion scope-aware so single-assignment locals are handled correctly per function
>> "%MSGFILE%" echo - rewrite multi-line function signatures and add support for inferred T ^| None optional parameter annotations
>> "%MSGFILE%" echo - propagate inferred argument types forward across functions in the same module so later signatures can reuse them
>> "%MSGFILE%" echo - expand annotator regression coverage for intent reporting, multiline signatures, optional defaults, and Final promotion
>> "%MSGFILE%" echo - update README.md to describe the annotator and its place in the Array-Compiler workflow

git commit -F "%MSGFILE%"
if errorlevel 1 goto :cleanup

git push

:cleanup
del "%MSGFILE%" >nul 2>nul
endlocal
