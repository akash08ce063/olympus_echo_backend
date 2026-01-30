"""
Static Memory Cache for Voice Testing Platform.

This module provides static memory caching for configuration and shared resources,
similar to pranthora_backend's static memory cache pattern.
"""

import json
import os
from typing import Dict, Any, Optional


class StaticMemoryCache:
    """Static memory cache for storing configuration and shared resources."""

    # Static class variables
    config: Dict[str, Any] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, config_file: str = "configurations.json"):
        """Load config into memory at startup."""
        if cls._initialized:
            return

        config_path = os.path.join(os.path.dirname(__file__), config_file)
        try:
            with open(config_path, "r") as f:
                cls.config = json.load(f)
            cls._initialized = True
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    @classmethod
    def get_config(cls, section: str, key: str, default=None):
        """Retrieve configuration value from the static memory cache."""
        if not cls._initialized:
            cls.initialize()
        return cls.config.get(section, {}).get(key, default)

    @classmethod
    def get_section(cls, section: str) -> Dict[str, Any]:
        """Retrieve entire configuration section."""
        if not cls._initialized:
            cls.initialize()
        return cls.config.get(section, {})

    @classmethod
    def get_supabase_url(cls) -> str:
        """Get Supabase URL from config."""
        return cls.get_config("database", "supabase_url")

    @classmethod
    def get_supabase_key(cls) -> str:
        """Get Supabase API key from config."""
        return cls.get_config("database", "supabase_key")

    @classmethod
    def get_pranthora_api_key(cls) -> str:
        """Get Pranthora API key from config."""
        return cls.get_config("pranthora", "api_key")

    @classmethod
    def get_pranthora_base_url(cls) -> str:
        """Get Pranthora base URL from config."""
        return cls.get_config("pranthora", "base_url")

    @classmethod
    def get_database_config(cls) -> Dict[str, str]:
        """Get complete database configuration."""
        return cls.get_section("database")

    @classmethod
    def get_pranthora_config(cls) -> Dict[str, str]:
        """Get complete Pranthora configuration."""
        return cls.get_section("pranthora")

    @classmethod
    def get_audio_chunk_duration_ms(cls) -> float:
        """Get audio chunk duration in milliseconds from config."""
        return float(cls.get_config("audio", "chunk_duration_ms"))

    @classmethod
    def get_connection_sync_timeout_seconds(cls) -> float:
        """Get connection sync timeout in seconds from config."""
        return float(cls.get_config("audio", "connection_sync_timeout_seconds", 10.0))

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if cache has been initialized."""
        return cls._initialized

    @classmethod
    def get_livekit_url(cls) -> str:
        """Get LiveKit URL from config."""
        return cls.get_config("livekit", "url")

    @classmethod
    def get_livekit_api_key(cls) -> str:
        """Get LiveKit API key from config."""
        return cls.get_config("livekit", "api_key")

    @classmethod
    def get_livekit_api_secret(cls) -> str:
        """Get LiveKit API secret from config."""
        return cls.get_config("livekit", "api_secret")

    @classmethod
    def get_livekit_config(cls) -> Dict[str, str]:
        """Get complete LiveKit configuration."""
        return cls.get_section("livekit")


# Initialize on import
StaticMemoryCache.initialize()
