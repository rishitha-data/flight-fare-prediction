import logging
import os
from logging.handlers import RotatingFileHandler

from src.config import LOG_PATH  # ✅ FIXED


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Creates and returns a configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Set log level from environment (default: INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Ensure log directory exists
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # =========================
    # FORMATTER
    # =========================
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # =========================
    # FILE HANDLER (with rotation)
    # =========================
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    # =========================
    # CONSOLE HANDLER
    # =========================
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # =========================
    # ADD HANDLERS
    # =========================
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Avoid propagation to root logger
    logger.propagate = False

    return logger