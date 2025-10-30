"""Caching layer for reasoning results."""

import hashlib
import json
from functools import lru_cache
from typing import Any, Dict, Optional

# Try Redis, fallback to in-memory
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ReasoningCache:
    """Cache for reasoning results with Redis fallback to LRU."""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            redis_url: Redis connection URL (None = in-memory only)
            ttl: Time-to-live in seconds (default: 1 hour)
            max_size: Max LRU cache size for in-memory fallback
        """
        self.ttl = ttl
        self.max_size = max_size
        self.redis_client = None
        self._memory_cache: Dict[str, Any] = {}

        # Try Redis
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None
        
        # Use LRU cache as primary for speed
        self._lru_get = lru_cache(maxsize=max_size)(self._get_from_memory)

    def get(self, query: str, config: str) -> Optional[Dict]:
        """
        Get cached result.

        Args:
            query: User query
            config: Configuration string

        Returns:
            Cached result or None
        """
        key = self._make_key(query, config)
        
        # Try LRU cache first (fastest)
        try:
            result = self._lru_get(key)
            if result is not None:
                return result
        except TypeError:
            # Unhashable key, skip LRU
            pass

        # Try Redis
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    result = json.loads(data)
                    # Warm up LRU cache
                    self._memory_cache[key] = result
                    return result
            except Exception:
                pass

        return None

    def set(self, query: str, config: str, result: Dict) -> None:
        """
        Cache a result.

        Args:
            query: User query
            config: Configuration string
            result: Result to cache
        """
        key = self._make_key(query, config)

        # Store in memory (LRU)
        self._memory_cache[key] = result
        
        # Clear LRU cache to refresh
        self._lru_get.cache_clear()

        # Store in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    self.ttl,
                    json.dumps(result, default=str)
                )
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._lru_get.cache_clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass

    def _make_key(self, query: str, config: str) -> str:
        """Generate cache key."""
        content = f"{query}:{config}"
        return f"kaelum:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def _get_from_memory(self, key: str) -> Optional[Dict]:
        """LRU-cached memory lookup."""
        return self._memory_cache.get(key)

import hashlib
import json
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ReasoningCache:
    """Cache for reasoning results using Redis."""

    def __init__(
        self,
        use_cache: bool = True,
        host: str = "localhost",
        port: int = 6379,
        ttl: int = 3600
    ):
        """
        Initialize reasoning cache.
        
        Args:
            use_cache: Enable/disable caching
            host: Redis host
            port: Redis port
            ttl: Time-to-live for cached results (seconds)
        """
        self.use_cache = use_cache
        self.ttl = ttl
        self.redis_client = None
        self.in_memory_cache = {}  # Fallback when Redis unavailable
        
        if use_cache:
            self._init_redis(host, port)
    
    def _init_redis(self, host: str, port: int) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            print("Warning: Redis not installed. Using in-memory cache. Install with: pip install redis")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            print(f"Redis cache connected at {host}:{port}")
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}. Using in-memory cache.")
            self.redis_client = None
    
    def get(self, query: str, config_hash: Optional[str] = None) -> Optional[dict]:
        """
        Get cached result for a query.
        
        Args:
            query: The reasoning query
            config_hash: Optional hash of config to include in cache key
            
        Returns:
            Cached result dict or None if not found
        """
        if not self.use_cache:
            return None
        
        key = self._make_key(query, config_hash)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Warning: Redis get failed: {e}")
        
        # Fallback to in-memory
        return self.in_memory_cache.get(key)
    
    def set(self, query: str, result: dict, config_hash: Optional[str] = None) -> None:
        """
        Cache a result.
        
        Args:
            query: The reasoning query
            result: Result dictionary to cache
            config_hash: Optional hash of config to include in cache key
        """
        if not self.use_cache:
            return
        
        key = self._make_key(query, config_hash)
        
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    self.ttl,
                    json.dumps(result)
                )
                return
            except Exception as e:
                print(f"Warning: Redis set failed: {e}")
        
        # Fallback to in-memory (with simple size limit)
        if len(self.in_memory_cache) > 1000:
            # Clear half the cache when limit reached
            keys = list(self.in_memory_cache.keys())
            for k in keys[:500]:
                del self.in_memory_cache[k]
        
        self.in_memory_cache[key] = result
    
    def clear(self) -> None:
        """Clear all cached results."""
        if self.redis_client:
            try:
                # Clear all kaelum keys
                keys = self.redis_client.keys("kaelum:reasoning:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                pass
        
        self.in_memory_cache.clear()
    
    def _make_key(self, query: str, config_hash: Optional[str] = None) -> str:
        """
        Create cache key from query and config.
        
        Args:
            query: The query string
            config_hash: Optional config hash
            
        Returns:
            Cache key string
        """
        # Hash the query
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Include config hash if provided
        if config_hash:
            return f"kaelum:reasoning:{query_hash}:{config_hash}"
        return f"kaelum:reasoning:{query_hash}"
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "enabled": self.use_cache,
            "backend": "redis" if self.redis_client else "memory",
            "in_memory_size": len(self.in_memory_cache)
        }
        
        if self.redis_client:
            try:
                keys = self.redis_client.keys("kaelum:reasoning:*")
                stats["redis_keys"] = len(keys) if keys else 0
            except Exception:
                stats["redis_keys"] = 0
        
        return stats
