"""Logging configuration for backend API."""

import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / '.kaelum' / 'kaelum.log'
LOG_FILE.parent.mkdir(exist_ok=True)


class NoLogsEndpointFilter(logging.Filter):
    """Filter out GET requests to /api/logs endpoint to avoid recursive logging."""
    
    def filter(self, record):
        message = record.getMessage()
        # Filter out GET /api/logs requests
        if 'GET /api/logs' in message or '/api/logs?' in message:
            return False
        return True


def setup_backend_logging():
    """Configure basic logging for backend API."""
    # Clear log file
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_FILE.touch()
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(str(LOG_FILE))
    
    # Add filter to prevent /api/logs GET requests from being logged
    logs_filter = NoLogsEndpointFilter()
    console_handler.addFilter(logs_filter)
    file_handler.addFilter(logs_filter)
    
    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[console_handler, file_handler]
    )
    
    # Silence noisy libraries
    for lib in ["httpcore", "httpx", "urllib3", "sentence_transformers",
                "transformers", "huggingface_hub", "filelock"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


