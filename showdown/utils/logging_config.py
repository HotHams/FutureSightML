"""Logging configuration."""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure and return the root logger for the application."""
    logger = logging.getLogger("showdown")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    console = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(fh)

    return logger
