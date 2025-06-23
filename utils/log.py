import logging
import os

def setup_logger(log_dir: str, name: str) -> logging.Logger:
    """
    Setup logger
    
    Args:
        log_dir: Directory to save logs
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create handlers
    log_file = os.path.join(log_dir, f'{name}.log')
    
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger