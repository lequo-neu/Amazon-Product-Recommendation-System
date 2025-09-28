# # logger.py
# import logging
# import time

# class Logger:
#     def __init__(self, process_name='', log_file=None):
#         if process_name:
#             log_file += f"{process_name}_{time.strftime('%Y-%m-%d_%H%M%S')}.log"
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)

#         # Create a file handler for logging
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(logging.INFO)
        
#         # Create a console handler for logging
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
        
#         # Define the log format
#         log_format = '%(asctime)s - %(levelname)s - %(message)s'
#         formatter = logging.Formatter(log_format)
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
        
#         # Add handlers to the logger
#         self.logger.addHandler(file_handler)
#         self.logger.addHandler(console_handler)
        
#     def log_info(self, message):
#         """Log info messages."""
#         self.logger.info(message)
    
#     def log_warning(self, message):
#         """Log warning messages."""
#         self.logger.warning(message)
    
#     def log_error(self, message):
#         """Log error messages."""
#         self.logger.error(message)
    
#     def log_debug(self, message):
#         """Log debug messages."""
#         self.logger.debug(message)
    
#     def log_exception(self, message):
#         """Log exception messages."""
#         self.logger.exception(message)

# logger.py
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
    """
    Notebook-safe Logger:
      - Clears old handlers to avoid duplicate logs when re-running cells.
      - Supports custom level, file path, console on/off.
      - File name auto-suffixed with process_name + timestamp.
    """
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