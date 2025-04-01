import os
import logging
from logging.handlers import RotatingFileHandler
import time
from config import get_config

def setup_logger():
    """Configure application logging."""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up logger
    logger = logging.getLogger('eduspark_tts')
    
    # Set log level based on configuration
    log_level = getattr(logging, config.LOG_LEVEL)
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler with rotation
    log_file = os.path.join('logs', f'eduspark_tts_{time.strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=10)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 