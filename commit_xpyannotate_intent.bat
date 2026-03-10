@echo off
setlocal
cd /d c:\python\Array-Compiler

git status
git add .

set MSGFILE=%TEMP%\array_compiler_commit_msg.txt
> "%MSGFILE%" echo Extend xpyannotate with intent analysis and richer signature inference
>> "%MSGFILE%" echo.
>> "%MSGFILE%" echo - add conservative parameter-intent analysis to the Python annotator and make it optional with --no-infer-intent
>> "%MSGFILE%" echo - make assignment counting scope-aware so single-assignment locals are marked Final correctly inside each function
>> "%MSGFILE%" echo - rewrite multi-line function signatures, not just single-line ones
>> "%MSGFILE%" echo - infer and emit optional parameter annotations of the form T ^| None for = None defaults when the body provides a stable type signal
>> "%MSGFILE%" echo - propagate inferred argument types forward across functions in the same module so later signatures can use them
>> "%MSGFILE%" echo - add annotator regressions for intent reporting, opt-out behavior, multiline signatures, optional defaults, and Final promotion

git commit -F "%MSGFILE%"
if errorlevel 1 goto :cleanup

git push

:cleanup
del "%MSGFILE%" >nul 2>nul
endlocal
