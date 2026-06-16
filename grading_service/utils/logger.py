"""
Structured JSON logger for NitHub Grading Service.
Outputs JSON logs to console (captured by Render) and to logs/grading.log.
Integrates with Sentry for error tracking.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from dotenv import load_dotenv

load_dotenv()


# ── Sentry Setup ─────────────────────────────────────────────────────────────

def init_sentry():
    """
    Initialise Sentry error tracking.
    Call this once at app startup in main.py.
    Sentry captures every ERROR log and unhandled exception automatically.
    """
    dsn = os.getenv("SENTRY_DSN")
    environment = os.getenv("ENVIRONMENT", "development")

    if not dsn:
        print("[logger] SENTRY_DSN not set — Sentry disabled")
        return

    sentry_logging = LoggingIntegration(
        level=logging.WARNING,       # Captures WARNING and above as breadcrumbs
        event_level=logging.ERROR    # Sends ERROR and above as Sentry events
    )

    sentry_sdk.init(
        dsn=dsn,
        integrations=[FastApiIntegration(), sentry_logging],
        environment=environment,
        traces_sample_rate=0.2,      # 20% of requests traced for performance
        send_default_pii=False,      # Never send student PII to Sentry
    )


# ── Logger Factory ────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named JSON logger.
    Usage in any file:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Action", extra={"exam_id": "...", "student_id": "..."})
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # Console handler — Render captures this as service logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — rotating, max 10MB per file, keeps last 5 files
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/grading.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
