import os
from typing import Optional


class ConfigError(Exception):
    """Configuration related errors"""

    pass


# API Keys (secrets from environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model Settings (constants)
GEMINI_MODEL = "gemini-2.5-flash-lite"
OPENROUTER_MODEL = "kwaipilot/kat-coder-pro:free"

# Performance Settings (constants)
BATCH_SIZE = 30
MAX_CONCURRENT = 5
MAX_OUTPUT_TOKENS = 32768
RATE_LIMIT_RPM = 10
LEGACY_BATCH_SIZE = 45
MAX_WORKERS = 5

# App Settings (constants)
PORT = 5000
DEFAULT_EDITION = "th_delhi"
DEFAULT_TOP_COUNT = 20

# Feature Flags (from environment)
USE_ASYNC_OPTIMIZATION = os.getenv("USE_ASYNC_OPTIMIZATION", "true").lower() == "true"
USE_OPENROUTER_FALLBACK = os.getenv("USE_OPENROUTER_FALLBACK", "true").lower() == "true"

# Chat Settings (constants)
CHAT_TEMPERATURE = 0.7
CHAT_MAX_TOKENS = 4000


def validate_config() -> None:
    """Validate required configuration"""
    if not GEMINI_API_KEY:
        raise ConfigError("GEMINI_API_KEY is required. Set it in .env file")

    if USE_OPENROUTER_FALLBACK and not OPENROUTER_API_KEY:
        raise ConfigError(
            "OPENROUTER_API_KEY is required when USE_OPENROUTER_FALLBACK is true"
        )


def get_config_summary() -> dict:
    """Get configuration summary for debugging"""
    return {
        "gemini_configured": bool(GEMINI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "async_optimization": USE_ASYNC_OPTIMIZATION,
        "fallback_enabled": USE_OPENROUTER_FALLBACK,
        "batch_size": BATCH_SIZE,
        "max_concurrent": MAX_CONCURRENT,
        "model": GEMINI_MODEL,
    }
