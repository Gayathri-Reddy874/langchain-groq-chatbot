"""Application-wide logging setup."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "langchain_groq_chatbot") -> logging.Logger:
    """Return a configured logger, creating handlers only once per process."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
