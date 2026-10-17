"""Structured logging utilities with both console and file output support."""

import json
import logging
import sys
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom formatter to emit structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_obj.update(record.extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(
    name: str = "recsys",
    log_file: Path | None = None,
    level: int = logging.INFO,
    json_format: bool = False,
) -> logging.Logger:
    """Returns a configured logger instance with console and optional file handlers.

    Args:
        name: Name of the logger.
        log_file: Path to log file (optional).
        level: Logging level.
        json_format: Whether to use structured JSON formatting.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(fmt)

        logger.addHandler(console_handler)

        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter() if json_format else fmt)
            logger.addHandler(file_handler)

    return logger
