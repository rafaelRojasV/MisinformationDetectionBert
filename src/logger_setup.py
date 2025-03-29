import os
import logging
import warnings
from logging.handlers import RotatingFileHandler
import transformers

def setup_logging(log_file='logs/training.log', log_level='INFO'):
    """
    Sets up centralized logging without concurrency locks:
      - RotatingFileHandler for the file
      - StreamHandler for console
      - Captures HF logs, plus Python warnings
    """
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Clear any existing root logger handlers
    logger = logging.getLogger()
    logger.handlers.clear()

    # Set overall log level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 1) File Handler (rotating)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 ** 7,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2) Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 3) Capture Python warnings in logging
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # 4) Let Hugging Face logs propagate into Python logger
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.set_verbosity_info()

    logger.info("🔄 Centralized logging is set up successfully (no concurrency lock).")
    return logger
