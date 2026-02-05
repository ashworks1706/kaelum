"""Central path management for Kaelum to ensure all .kaelum directories are created in project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

KAELUM_DIR = PROJECT_ROOT / ".kaelum"

DEFAULT_CACHE_DIR = str(KAELUM_DIR / "cache")
DEFAULT_ROUTER_DIR = str(KAELUM_DIR / "routing")
DEFAULT_CALIBRATION_DIR = str(KAELUM_DIR / "calibration")
DEFAULT_CACHE_VALIDATION_DIR = str(KAELUM_DIR / "cache_validation")
DEFAULT_ACTIVE_LEARNING_DIR = str(KAELUM_DIR / "active_learning")
DEFAULT_ANALYTICS_DIR = str(KAELUM_DIR / "analytics")
DEFAULT_TREE_CACHE_DIR = str(KAELUM_DIR / "tree_cache")
DEFAULT_LOG_FILE = KAELUM_DIR / "kaelum.log"

KAELUM_DIR.mkdir(parents=True, exist_ok=True)
