"""
Setup logging for the application
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from config.config import active_config

def setup_logging(app_name='candidate-matcher'):
    """Setup application logging with rotation"""
    os.makedirs(active_config.LOG_DIR, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    file_handler = RotatingFileHandler(
        f"{active_config.LOG_DIR}/{app_name}.log", 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, active_config.LOG_LEVEL))
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, active_config.LOG_LEVEL))
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, active_config.LOG_LEVEL))
    
    # Remove existing handlers to avoid duplicates if this function is called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


# Create a logger instance to import in other modules
logger = setup_logging()
