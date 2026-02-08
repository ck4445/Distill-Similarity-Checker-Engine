@echo off
setlocal enabledelayedexpansion

:: =========================
:: Config
:: =========================
set "SRC=C:\work\Main\QuickAcess\test"
set "DST=C:\work\Main\Distill-Similarity-Checker-Engine"
set "REMOTE=https://github.com/ck4445/Distill-Similarity-Checker-Engine.git"

echo.
echo [1/7] Switching to destination repo...
cd /d "%DST%" || (
  echo ERROR: Destination folder not found: %DST%
  goto :fail
)

echo.
echo [2/7] Ensuring engine package folder exists...
if not exist "distill_similarity_checker_engine" mkdir "distill_similarity_checker_engine"

echo.
echo [3/7] Copying required files from source...
copy /Y "%SRC%\README.md" "%DST%\README.md" >nul
copy /Y "%SRC%\requirements.txt" "%DST%\requirements.txt" >nul
copy /Y "%SRC%\preprocess.py" "%DST%\preprocess.py" >nul
copy /Y "%SRC%\features_style.py" "%DST%\features_style.py" >nul
copy /Y "%SRC%\scoring.py" "%DST%\scoring.py" >nul
copy /Y "%SRC%\engine_cli.py" "%DST%\engine_cli.py" >nul
copy /Y "%SRC%\distill_similarity_checker_engine\__init__.py" "%DST%\distill_similarity_checker_engine\__init__.py" >nul
copy /Y "%SRC%\distill_similarity_checker_engine\engine.py" "%DST%\distill_similarity_checker_engine\engine.py" >nul

echo.
echo [4/7] Verifying required files...
set "MISSING=0"
for %%F in (
  "README.md"
  "requirements.txt"
  "preprocess.py"
  "features_style.py"
  "scoring.py"
  "engine_cli.py"
  "distill_similarity_checker_engine\__init__.py"
  "distill_similarity_checker_engine\engine.py"
) do (
  if not exist "%%~F" (
    echo MISSING: %%~F
    set "MISSING=1"
  )
)

if "!MISSING!"=="1" (
  echo ERROR: Missing required files. Aborting push.
  goto :fail
)

echo.
echo [5/7] Optional import sanity check...
python -c "import distill_similarity_checker_engine; print('engine import ok')" || (
  echo WARNING: Python import check failed. Continuing to git steps.
)

echo.
echo [6/7] Setting git remote (clear old origin first)...
git remote remove origin 2>nul
git remote add origin "%REMOTE%" || (
  echo ERROR: Failed to add remote.
  goto :fail
)

echo.
echo [7/7] Commit and push...
git add .
git commit -m "Sync engine files and publish Distill Similarity Checker Engine" 2>nul
git branch -M main
git push -u origin main || (
  echo ERROR: Push failed. Check repo exists and you have access.
  goto :fail
)

echo.
echo SUCCESS: Engine repo pushed to %REMOTE%
goto :end

:fail
echo.
echo FAILED.
exit /b 1

:end
endlocal
pause
