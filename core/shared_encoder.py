"""Shared embedding encoder to avoid loading SentenceTransformer multiple times.

Instead of each component creating its own encoder (wasting ~1GB memory),
we create a single shared instance and pass it around.
"""

from sentence_transformers import SentenceTransformer
from typing import Optional
import logging

logger = logging.getLogger("kaelum.shared_encoder")

_shared_encoder: Optional[SentenceTransformer] = None

def get_shared_encoder(model_name: str = "all-MiniLM-L6-v2", device: str = 'cpu') -> SentenceTransformer:
    """Get or create the shared encoder instance.
    
    Args:
        model_name: SentenceTransformer model name
        device: 'cpu' or 'cuda' - use 'cpu' when vLLM is using GPU
        
    Returns:
        Shared SentenceTransformer instance
    """
    global _shared_encoder
    
    if _shared_encoder is None:
        logger.info(f"Initializing shared encoder: {model_name} on {device}")
        _shared_encoder = SentenceTransformer(model_name, device=device)
        logger.info(f"✓ Shared encoder loaded (will be reused by all components)")
    
    return _shared_encoder

def reset_shared_encoder():
    """Reset the shared encoder (useful for testing or config changes)."""
    global _shared_encoder
    _shared_encoder = None
