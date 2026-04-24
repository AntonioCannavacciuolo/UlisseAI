import os
import sys
import time
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

load_dotenv()

def get_path_from_env(env_var, default_folder):
    env_path = os.getenv(env_var)
    if env_path:
        return Path(env_path)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / default_folder

project_root = Path(__file__).resolve().parent.parent
corpus_dir = get_path_from_env("CORPUS_PATH", "corpus")
logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / "sync_watcher.log"

formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(
    filename=str(log_file),
    mode='a',
    maxBytes=5*1024*1024,
    backupCount=2,
    delay=True
)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler]
)

class CorpusHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_run_chatgpt = 0
        self.last_run_ulisse = 0
        self.debounce = 2 

    def run_command(self, cmd_list, description):
        logging.info(f"Running: {description}")
        try:
            # sys.executable ensures we use the exact same Python interpreter running this script
            full_cmd = [sys.executable] + cmd_list
            result = subprocess.run(full_cmd, capture_output=True, text=True, cwd=str(project_root))
            if result.returncode == 0:
                logging.info(f"Success: {description}")
            else:
                logging.error(f"Error in {description}:\n{result.stderr}")
        except Exception as e:
            logging.error(f"Exception running {description}: {e}")

    def on_modified(self, event):
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        filename = path.name
        now = time.time()
        
        if filename == "conversations.json":
            if now - self.last_run_chatgpt > self.debounce:
                self.last_run_chatgpt = now
                logging.info(f"Detected changes in {filename}")
                self.run_command(["scripts/parse_chatgpt.py"], "Parse ChatGPT")
                self.run_command(["scripts/embed_corpus.py"], "Embed Corpus")
                
        elif filename == "new_conversations.jsonl":
            if now - self.last_run_ulisse > self.debounce:
                self.last_run_ulisse = now
                logging.info(f"Detected changes in {filename}")
                self.run_command(["scripts/sync_ulisse_conversations.py"], "Sync Ulisse Conversations")



    def on_created(self, event):
        self.on_modified(event)

def main():
    if not corpus_dir.exists():
        logging.error(f"Corpus directory not found: {corpus_dir}")
        corpus_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created corpus directory: {corpus_dir}")
        
    event_handler = CorpusHandler()
    observer = Observer()
    observer.schedule(event_handler, str(corpus_dir), recursive=False)
    
    logging.info(f"Starting watcher on {corpus_dir}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
        observer.stop()
        
    observer.join()

if __name__ == "__main__":
    main()
