@echo off
echo Starting Adaptive-HTFL Project...

cd /d %~dp0

echo Step 1: Running federated learning experiment...
python run_experiment.py
if %errorlevel% neq 0 (
    echo Experiment failed!
    pause
    exit /b 1
)

echo Step 2: Generating dashboard figures...
python dashboard.py
if %errorlevel% neq 0 (
    echo Dashboard generation failed!
    pause
    exit /b 1
)

echo Step 3: Starting web server...
cd results\figures
echo Dashboard ready! Open http://localhost:8000/index.html in your browser
python -m http.server 8000

pause