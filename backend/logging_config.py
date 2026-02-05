"""Logging configuration for backend API."""

import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / '.kaelum' / 'kaelum.log'
LOG_FILE.parent.mkdir(exist_ok=True)

class NoLogsEndpointFilter(logging.Filter):
    """Filter out GET requests to /api/logs endpoint to avoid recursive logging."""
    
    def filter(self, record):
        message = record.getMessage()

        if 'GET /api/logs' in message or '/api/logs?' in message:
            return False
        return True

class ComponentPrefixFormatter(logging.Formatter):
    """Add visual prefixes to component logs for better frontend display."""
    
    COMPONENT_PREFIXES = {

        'kaelum.router': '🧭 [ROUTER]',
        'kaelum.orchestrator': '🎯 [ORCHESTRATOR]',
        'kaelum.lats': '🌳 [TREE SEARCH]',
        'kaelum.verification': '✅ [VERIFICATION]',
        'kaelum.reflection': '🔄 [REFLECTION]',
        'kaelum.cache': '💾 [CACHE]',
        'kaelum.cache_validator': '🔍 [CACHE VALIDATOR]',
        'kaelum.llm': '🤖 [LLM]',
        'kaelum.reward': '⭐ [REWARD]',
        
        'kaelum.coherence_detector': '🔗 [COHERENCE]',
        'kaelum.completeness_detector': '📋 [COMPLETENESS]',
        'kaelum.conclusion_detector': '🎬 [CONCLUSION]',
        'kaelum.domain_classifier': '🏷️ [DOMAIN]',
        'kaelum.repetition_detector': '🔁 [REPETITION]',
        'kaelum.task_classifier': '📝 [TASK TYPE]',
        'kaelum.worker_type_classifier': '🔀 [WORKER TYPE]',
        
        'kaelum.worker': '👷 [WORKER]',
        'kaelum.math_worker': '➗ [MATH]',
        'kaelum.logic_worker': '🧠 [LOGIC]',
        'kaelum.code_worker': '💻 [CODE]',
        'kaelum.factual_worker': '📚 [FACTUAL]',
        'kaelum.creative_worker': '🎨 [CREATIVE]',
        'kaelum.analysis_worker': '🔬 [ANALYSIS]',
    }
    
    def format(self, record):

        logger_name = record.name
        prefix = self.COMPONENT_PREFIXES.get(logger_name, '')
        
        message = super().format(record)
        
        if prefix:
            return f"{prefix} {message}"
        return message

def setup_backend_logging():
    """Configure basic logging for backend API."""

    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_FILE.touch()
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(str(LOG_FILE))
    
    logs_filter = NoLogsEndpointFilter()
    console_handler.addFilter(logs_filter)
    file_handler.addFilter(logs_filter)
    
    formatter = ComponentPrefixFormatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )
    
    for lib in ["httpcore", "httpx", "urllib3", "sentence_transformers",
                "transformers", "huggingface_hub", "filelock"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
