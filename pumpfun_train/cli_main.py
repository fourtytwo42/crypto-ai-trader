"""Main entry point for Pump.fun Prediction Model.

Provides CLI interface for pump.fun operations.
"""

import sys

import structlog

# Simplified config for logging only
import os

# Configure structlog with log level filtering
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Set standard library logging level BEFORE configuring structlog
# This ensures structlog respects the log level
import logging
if log_level in ("WARNING", "ERROR", "CRITICAL"):
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        handlers=[logging.NullHandler()] if log_level == "CRITICAL" else None,
        force=True,
    )
    logging.getLogger().setLevel(getattr(logging, log_level, logging.WARNING))
    if log_level == "CRITICAL":
        logging.getLogger().handlers = [logging.NullHandler()]
        logging.getLogger().propagate = False
    # Suppress noisy loggers
    for logger_name in ["pumpfun_train.normalizer", "pumpfun_train.progress"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level, logging.WARNING))
        if log_level == "CRITICAL":
            logger.handlers = [logging.NullHandler()]
            logger.propagate = False

# Configure structlog - suppress all output if log level is CRITICAL
if log_level == "CRITICAL":
    # Use a null logger factory that discards all output
    class NullLogger:
        def _log(self, *args, **kwargs):
            pass
        def debug(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass
        def critical(self, *args, **kwargs):
            pass
    
    class NullLoggerFactory:
        def __call__(self, *args, **kwargs):
            return NullLogger()
    
    structlog.configure(
        processors=[],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=NullLoggerFactory(),
        cache_logger_on_first_use=True,
    )
else:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if os.getenv("LOG_FORMAT", "console") == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger(__name__)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    try:
        # Import here to avoid circular imports
        from pumpfun_train.cli.cli import cli

        logger.info("Starting Pump.fun Prediction Model")
        cli()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
