# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='app.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Rotating file handler: 5 MB per file, keeping 5 backups
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Clear existing handlers (if any) and add new one
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    
    # Also add a console handler for easier debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# In main.py, before using logging, simply call:
# from logging_config import setup_logging
# setup_logging()
