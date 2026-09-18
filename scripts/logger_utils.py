# =========================================================
# logger_utils.py — Reusable logging system
# =========================================================
import sys
import os

class LoggerWriter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logger(log_file_path):
    """
    Redirects sys.stdout to both the terminal and a log file.
    Creates the target directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = LoggerWriter(log_file_path)
