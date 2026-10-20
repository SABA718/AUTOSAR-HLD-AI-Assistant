"""Structured logging setup for AUTOSAR HLD AI Assistant.

Provides unified log formatting and level configuration across all
application components.
"""

import logging
import sys
from typing import Optional
from config.settings import get_settings


DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: Optional[str] = None, log_level: Optional[str] = None) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Name of the logger, typically __name__ of the calling module.
        log_level: Optional log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logging.Logger instance.
    """
    logger_name = name or "autosar_assistant"
    logger = logging.getLogger(logger_name)

    # Determine effective log level
    if log_level is None:
        try:
            settings = get_settings()
            effective_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        except Exception:
            effective_level = logging.INFO
    else:
        effective_level = getattr(logging, log_level.upper(), logging.INFO)

    logger.setLevel(effective_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_level)
        formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to the root logger if we handle it directly
    logger.propagate = False

    return logger
