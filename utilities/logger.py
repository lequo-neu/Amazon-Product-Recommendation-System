import logging
import time
from pathlib import Path
from typing import Optional

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR":    logging.ERROR,
    "WARNING":  logging.WARNING,
    "INFO":     logging.INFO,
    "DEBUG":    logging.DEBUG,
}

class Logger:
    def __init__(
        self,
        process_name: str = "",
        log_file: Optional[str | Path] = None,
        level: str | int = "INFO",
        console: bool = True,
        fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
        logger_name: Optional[str] = None,
    ):
        # Resolve level
        lvl = _LEVEL_MAP.get(str(level).upper(), level if isinstance(level, int) else logging.INFO)

        # Resolve log path
        # Accept both "folder/" or "folder/prefix.log" from Configurations
        if log_file is None:
            log_path = Path("logs")
        else:
            log_path = Path(log_file)

        # If a directory is provided, create a file name inside it
        if log_path.suffix == "":  # looks like a directory or prefix without extension
            log_path.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d_%H%M%S")
            base = (process_name or "app")
            log_path = log_path / f"{base}_{ts}.log"
        else:
            # ensure parent exists
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # Pick a stable, specific logger name (avoid __name__ for notebooks)
        lname = logger_name or (process_name or "aprs")
        self.logger = logging.getLogger(lname)
        self.logger.setLevel(lvl)
        self.logger.propagate = False  # do not pass to root

        # Clear existing handlers to avoid duplicates when cells re-run
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter(fmt)

        # File handler
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Optional console handler
        if console:
            ch = logging.StreamHandler()
            ch.setLevel(lvl)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Keep for reference
        self.log_file = str(log_path)

    # Convenience methods
    def log_info(self, msg: str):     self.logger.info(msg)
    def log_warning(self, msg: str):  self.logger.warning(msg)
    def log_error(self, msg: str):    self.logger.error(msg)
    def log_debug(self, msg: str):    self.logger.debug(msg)
    def log_exception(self, msg: str): self.logger.exception(msg)