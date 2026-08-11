"""Global logging configuration for Skein."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone

_configured = False


def configure_logging(
    level: int = logging.INFO,
    log_dir: str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure global logging for Skein.

    Args:
        level: Logging level (default: INFO)
        log_dir: Directory for log files. If None, only console output.
        format_string: Custom log format. Uses default if None.
    """
    global _configured
    if _configured:
        return

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    formatter = logging.Formatter(format_string)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"skein_{timestamp}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name, typically __name__ of the module.

    Returns:
        Logger instance configured with global settings.
    """
    return logging.getLogger(name)
