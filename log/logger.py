"""JSON structured logging with rotation support.

Two log types:
- ``user_chat``  → logs/user_chat_log/   (user conversation logs)
- ``auto_task``  → logs/auto_task_log/   (autonomous pipeline / agent logs)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from pythonjsonlogger import jsonlogger

from config.settings import get_settings


# Reserved LogRecord attributes — extra keys must not clash with these
_LOG_RECORD_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName",
}


def get_logger(name: str, log_type: str = "auto_task") -> logging.Logger:
    """Return a configured logger that writes JSON lines to a rotated file.

    Args:
        name: The ``__name__`` of the calling module.
        log_type: ``"user_chat"`` or ``"auto_task"`` (default).

    Returns:
        A stdlib Logger with a JSON RotatingFileHandler + console handler.
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured — idempotent

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    log_dir = (
        settings.LOG_DIR_USER_CHAT if log_type == "user_chat"
        else settings.LOG_DIR_AUTO_TASK
    )
    os.makedirs(log_dir, exist_ok=True)

    handler = RotatingFileHandler(
        os.path.join(log_dir, f"{name.replace('.', '_')}.jsonl"),
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Console handler at WARNING+ for development visibility
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter(
        "%(levelname)s [%(name)s] %(message)s"
    ))
    logger.addHandler(console)

    return logger
