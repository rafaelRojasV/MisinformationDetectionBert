# src/logger_setup.py

import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import os
from contextlib import contextmanager

def setup_logging(log_file='logs/pipeline.log', log_level='INFO'):
    """
    Sets up centralized logging with a concurrent rotating file handler and a stream handler.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    try:
        file_handler = ConcurrentRotatingFileHandler(
            log_file, maxBytes=10**7, backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️ Failed to set up file handler for logging: {e}")

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("🔄 Centralized logging is set up successfully.")

def get_model_logger(model_name, run_dir, log_level='INFO'):
    """
    Creates and returns a logger for a specific model.
    """
    logger = logging.getLogger(f"model_{model_name}")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        log_file = os.path.join(run_dir, 'logs', f"training_{model_name}.log")
        try:
            file_handler = ConcurrentRotatingFileHandler(
                log_file, maxBytes=10**7, backupCount=5
            )
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"⚠️ Failed to set up file handler for model '{model_name}': {e}")

    logger.info(f"🔄 Logger for model '{model_name}' is set up.")
    return logger

@contextmanager
def logging_redirector(logger):
    """
    Redirects logging messages within the context to the specified logger.
    """
    original_logger = logging.getLogger()
    try:
        # Remove all handlers from the root logger
        root_handlers = original_logger.handlers[:]
        for handler in root_handlers:
            original_logger.removeHandler(handler)

        # Add the model logger's handlers to the root logger
        for handler in logger.handlers:
            original_logger.addHandler(handler)

        yield
    finally:
        original_logger.handlers = root_handlers
