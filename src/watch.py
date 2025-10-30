import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json

# === CONFIGURATION ===
with open("path.json", "r") as f:
    config_data = json.load(f)


WATCH_DIR = config_data["WATCH_DIR"]  # directory to watch
TRANSCRIPT_SCRIPT = config_data["TRANSCRIPT_SCRIPT"]
PYTHON_EXEC = config_data["PYTHON_EXEC"]  # path to your venv python
# =====================


class WatchHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        print(f"\nNew file detected: {filepath}")

        # Wait a bit to make sure file writing is complete
        time.sleep(2)

        try:
            print(f"Starting transcription for {os.path.basename(filepath)}...")
            subprocess.run([PYTHON_EXEC, TRANSCRIPT_SCRIPT, filepath], check=True)
            print(f"Done: {os.path.basename(filepath)}\n")
        except subprocess.CalledProcessError as e:
            print(f"Transcription failed for {filepath}: {e}")


def main():
    os.makedirs(WATCH_DIR, exist_ok=True)
    print(f"Watching directory: {WATCH_DIR}")

    event_handler = WatchHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
