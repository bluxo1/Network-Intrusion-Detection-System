"""
Thin inference adapter for the Flask layer.

Wraps ``src.predict`` so the web app has a small, stable surface:
  * ``models_available()`` - are the trained artifacts present on disk?
  * ``get_predictor()``    - the lazily-loaded, process-wide ``Predictor``.
"""

import os

from src.config import CONFIG, abspath
from src.predict import get_predictor  # re-exported for the app

__all__ = ["get_predictor", "models_available", "missing_artifacts"]


def missing_artifacts() -> list:
    """Return the list of required artifact paths that are not yet on disk."""
    required = [
        CONFIG["artifacts"]["scaler"],
        CONFIG["artifacts"]["encoder"],
        CONFIG["artifacts"]["metadata"],
        CONFIG["artifacts"]["binary_model"],
        CONFIG["artifacts"]["multiclass_model"],
    ]
    return [p for p in required if not os.path.exists(abspath(p))]


def models_available() -> bool:
    """True only if every artifact needed for inference exists."""
    return len(missing_artifacts()) == 0
