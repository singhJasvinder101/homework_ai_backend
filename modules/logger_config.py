import logging
from logging.handlers import RotatingFileHandler
import structlog

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            RotatingFileHandler('homework_ai.log', maxBytes=10000000, backupCount=5),
            logging.StreamHandler()
        ]
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )