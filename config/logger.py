import logging
import sys
from config.settings import settings

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL if hasattr(settings, "LOG_LEVEL") else logging.INFO)
    
    if not logger.handlers:
        # Create standard stdout stream handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Structured log format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # Prevent logger from propagating to root logger to avoid double logging
    logger.propagate = False
    return logger
