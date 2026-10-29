@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Please follow README.md.
  exit /b 1
)
if not exist "artifacts\policy_selection.json" (
  echo Models and policy are not ready. Please complete the README workflow.
  exit /b 2
)
if "%~1"=="--check" (
  ".venv\Scripts\python.exe" -c "from spam_analysis.experiments.runtime import ExperimentSession; from spam_analysis.inference import record_from_text; session=ExperimentSession(); session.scores(record_from_text('runtime check', 'local content check')); print('Current published model and local inference check passed')"
  exit /b
)
echo Open http://127.0.0.1:8601 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8601 --server.headless true
endlocal
