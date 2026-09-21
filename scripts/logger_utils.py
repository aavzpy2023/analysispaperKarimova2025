# =========================================================
# logger_utils.py — Reusable logging system
# =========================================================
import logging
import os
import sys


class LoggerWriter:
    """Tee stream: writes to the original terminal stream and to a shared log file."""

    def __init__(self, terminal, log_handle):
        self.terminal = terminal
        self.log = log_handle

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        # Delegate anything else (encoding, fileno, ...) to the real terminal stream
        return getattr(self.terminal, name)


def setup_logger(log_file_path, level=logging.INFO):
    """
    Duplicates stdout AND stderr into a log file and routes the standard
    `logging` module through the same stream.

    - print() output           -> terminal + log file (as before)
    - logger.info(...) output  -> terminal + log file (previously LOST)
    - tracebacks / warnings    -> terminal + log file (previously terminal only)

    Safe to call more than once: it never nests tee streams.
    """
    fmt = "%(asctime)s | %(levelname)s | %(message)s"

    # When launched by run_pipeline.py the runner already captures this process's stdout/stderr
    # into the per-step and master logs. Opening the same file here would truncate/overwrite it,
    # so in that mode we only route `logging` to stdout and leave file handling to the runner.
    if os.environ.get("MLGNN_LOG_MANAGED") == "1":
        logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)
        return

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Unwrap a previous tee so repeated calls do not stack writers
    out = sys.stdout.terminal if isinstance(sys.stdout, LoggerWriter) else sys.stdout
    err = sys.stderr.terminal if isinstance(sys.stderr, LoggerWriter) else sys.stderr
    if isinstance(sys.stdout, LoggerWriter):
        try:
            sys.stdout.log.close()
        except Exception:
            pass

    log_handle = open(log_file_path, "w", encoding="utf-8")
    sys.stdout = LoggerWriter(out, log_handle)
    sys.stderr = LoggerWriter(err, log_handle)

    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stdout,   # the tee writer created above
        force=True,          # replace any pre-existing root handlers
    )
