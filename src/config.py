"""Configuration loader.

Reads ``config.yaml`` from the project root and exposes it as a nested dict.
Also resolves the project root so every other module can build absolute paths
regardless of the current working directory.
"""

import os
import yaml

# Project root = one level up from this file (src/ -> project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and return the YAML configuration as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def abspath(relative_path: str) -> str:
    """Resolve a config-relative path to an absolute path under the project root."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


# Loaded once at import time for convenience.
CONFIG = load_config()
