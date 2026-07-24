#!/usr/bin/env python3
"""
Auto-update dashboard watcher — monitors code changes and re-runs experiment/dashboard.
Usage: python watch_dashboard.py
"""

import time
import subprocess
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_run = 0
        self.cooldown = 5  # seconds between auto-runs

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith('.py'):
            return

        # Skip dashboard.py itself to avoid loops
        if 'dashboard.py' in event.src_path:
            return

        now = time.time()
        if now - self.last_run < self.cooldown:
            return

        print(f"\n[{time.strftime('%H:%M:%S')}] Code change detected: {os.path.basename(event.src_path)}")
        self.last_run = now
        self.run_update()

    def run_update(self):
        try:
            print("Running experiment...")
            result = subprocess.run([
                sys.executable, 'run_experiment.py'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

            if result.returncode != 0:
                print(f"Experiment failed: {result.stderr}")
                return

            print("Generating dashboard...")
            result = subprocess.run([
                sys.executable, 'dashboard.py'
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

            if result.returncode != 0:
                print(f"Dashboard failed: {result.stderr}")
                return

            print("Dashboard updated! Refresh http://localhost:8000/index.html")

        except Exception as e:
            print(f"Error during update: {e}")

if __name__ == "__main__":
    print("Starting auto-update watcher...")
    print("Monitoring .py files in core/, attacks/, evaluation/, utils/, data/, blockchain/")
    print("Press Ctrl+C to stop")

    event_handler = CodeChangeHandler()
    observer = Observer()

    # Watch key directories
    watch_dirs = ['core', 'attacks', 'evaluation', 'utils', 'data', 'blockchain']
    for d in watch_dirs:
        if os.path.exists(d):
            observer.schedule(event_handler, d, recursive=True)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")

    observer.join()