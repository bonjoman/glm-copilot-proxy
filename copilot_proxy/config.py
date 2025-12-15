"""Configuration management for Copilot proxy."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_DIR = Path.home() / ".copilot-proxy"
CONFIG_FILE = CONFIG_DIR / "config.json"


def is_first_run() -> bool:
    """Check if this is the first run (no config file exists)."""
    return not CONFIG_FILE.exists()


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return CONFIG_DIR


def get_config_file() -> Path:
    """Get the configuration file path."""
    return CONFIG_FILE


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR


def load_config() -> Dict[str, Any]:
    """Load configuration from file.

    Returns:
        Dictionary containing configuration, or empty dict if file doesn't exist.
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Dictionary containing configuration to save.
    """
    ensure_config_dir()

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_api_key() -> Optional[str]:
    """Get API key from config file.

    Returns:
        API key if found in config, None otherwise.
    """
    config = load_config()
    api_key = config.get("api_key")
    if api_key:
        return api_key.strip()
    return None


def set_api_key(api_key: str) -> None:
    """Save API key to config file.

    Args:
        api_key: The API key to save.
    """
    config = load_config()
    config["api_key"] = api_key.strip()
    save_config(config)


def get_base_url() -> Optional[str]:
    """Get base URL from config file.

    Returns:
        Base URL if found in config, None otherwise.
    """
    config = load_config()
    base_url = config.get("base_url")
    if base_url:
        return base_url.strip()
    return None


def set_base_url(base_url: str) -> None:
    """Save base URL to config file.

    Args:
        base_url: The base URL to save.
    """
    config = load_config()
    config["base_url"] = base_url.strip()
    save_config(config)


def ensure_complete_config() -> None:
    """Ensure all configuration values exist in config, including defaults."""
    if not CONFIG_FILE.exists():
        return

    config = load_config()
    updated = False

    # Ensure base_url exists (save default if not set)
    if "base_url" not in config:
        config["base_url"] = "https://api.z.ai/api/coding/paas/v4"
        updated = True

    # Ensure context_length exists
    if "context_length" not in config:
        config["context_length"] = 64000
        updated = True

    # Ensure default host exists
    if "default_host" not in config:
        config["default_host"] = "127.0.0.1"
        updated = True

    # Ensure default port exists
    if "default_port" not in config:
        config["default_port"] = 11434
        updated = True

    if updated:
        save_config(config)


def get_context_length() -> int:
    """Get context length from config file.

    Returns:
        Context length if found in config, 64000 otherwise.
    """
    config = load_config()
    return config.get("context_length", 64000)


def set_context_length(context_length: int) -> None:
    """Save context length to config file.

    Args:
        context_length: The context length to save.
    """
    config = load_config()
    config["context_length"] = context_length
    save_config(config)


def get_model_name() -> Optional[str]:
    """Get model name from config file.

    Returns:
        Model name if found in config, None otherwise.
    """
    config = load_config()
    model_name = config.get("model_name")
    if model_name:
        return model_name.strip()
    return None


def set_model_name(model_name: str) -> None:
    """Save model name to config file.

    Args:
        model_name: The model name to save.
    """
    config = load_config()
    config["model_name"] = model_name.strip()
    save_config(config)


def get_temperature() -> Optional[float]:
    """Get temperature from config file.

    Returns:
        Temperature if found in config, None otherwise.
    """
    config = load_config()
    return config.get("temperature")


def set_temperature(temperature: float) -> None:
    """Save temperature to config file.

    Args:
        temperature: The temperature to save.
    """
    config = load_config()
    config["temperature"] = temperature
    save_config(config)