"""
Logging configuration for desktop app
"""

import logging
import sys
from pathlib import Path

def setup_logging():
    """Configure application logging"""
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "app.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger("FizyonomiAI")
    logger.info("=" * 50)
    logger.info("Fizyonomi AI Desktop Uygulaması Başlatıldı")
    logger.info(f"Log dosyası: {log_file}")
    logger.info("=" * 50)
    
    return logger

# Global logger
logger = None

def get_logger():
    """Get application logger"""
    global logger
    if logger is None:
        logger = setup_logging()
    return logger
