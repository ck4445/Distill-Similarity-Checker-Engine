@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo Distill Similarity Checker Engine - Upload Script
echo Target repo: https://github.com/ck4445/Distill-Similarity-Checker-Engine.git
echo ======================================================

cd /d C:\work\Main\Distill-Similarity-Checker-Engine || (
  echo ERROR: Could not open C:\work\Main\Distill-Similarity-Checker-Engine
  goto :fail
)

echo.
echo [1/5] Checking required files...
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
  echo ERROR: One or more required files are missing.
  goto :fail
)

echo.
echo [2/5] Setting remote (removing old origin first)...
git remote remove origin 2>nul
git remote add origin https://github.com/ck4445/Distill-Similarity-Checker-Engine.git || (
  echo ERROR: Failed to set remote origin.
  goto :fail
)

echo.
echo [3/5] Git add + commit...
git add .
git commit -m "Update standalone engine package and files" 2>nul

echo.
echo [4/5] Ensuring main branch...
git branch -M main

echo.
echo [5/5] Pushing to GitHub...
git push -u origin main || (
  echo ERROR: Push failed. Confirm repo exists and your GitHub auth is valid.
  goto :fail
)

echo.
echo SUCCESS: Engine repo uploaded.
goto :end

:fail
echo.
echo FAILED.
exit /b 1

:end
endlocal
pause
