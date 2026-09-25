"""
Inference helper: loads the trained artifacts and runs the two-stage
classification used everywhere (evaluation, Flask app, unit tests).

Two-stage / layered strategy
----------------------------
1. The **binary** model decides Normal vs Attack from its sigmoid probability
   against ``inference.attack_threshold``.
2. If (and only if) the traffic is flagged as an attack, the **multi-class**
   model picks the specific attack type among {DOS, PROBE, R2L, U2R}
   (the Normal logit is ignored at this stage).

This keeps a single high-recall gate for "is this an attack at all?" and defers
the finer-grained typing to the specialised model.
"""

from typing import Dict, List, Optional

import json
import math

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CONFIG, abspath
from .model import BinaryClassifier, MultiClassClassifier
from .preprocess import transform
from .schema import BINARY_FLAG_FEATURES, CLASS_NAMES, FEATURE_COLUMNS


class InputValidationError(ValueError):
    """A request contains a feature value the trained model cannot use safely."""


def validate_features(features: dict, metadata: dict) -> dict:
    """Build one complete, finite row from the public 41-feature schema."""
    if not isinstance(features, dict):
        raise InputValidationError("features must be a JSON object")
    unknown = sorted(set(features) - set(FEATURE_COLUMNS))
    if unknown:
        raise InputValidationError(f"unknown feature: {unknown[0]}")

    row = {}
    categories = metadata["categorical_columns"]
    for col in FEATURE_COLUMNS:
        if col in categories:
            default = metadata["categories"][col][0]
            value = features.get(col, default)
            if not isinstance(value, str) or not value or len(value) > 100:
                raise InputValidationError(f"{col} must be a nonempty category (up to 100 characters)")
            # OneHotEncoder intentionally supports categories not seen in fit.
            row[col] = value
            continue

        value = features.get(col, 0)
        if isinstance(value, bool):
            raise InputValidationError(f"{col} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(f"{col} must be numeric") from exc
        if not math.isfinite(number) or number < 0 or number > float(np.finfo(np.float32).max):
            raise InputValidationError(f"{col} must be finite and nonnegative")
        if col.endswith("rate") and number > 1:
            raise InputValidationError(f"{col} must be between 0 and 1")
        if not col.endswith("rate") and not number.is_integer():
            raise InputValidationError(f"{col} must be an integer")
        if col in BINARY_FLAG_FEATURES and number not in (0, 1):
            raise InputValidationError(f"{col} must be 0 or 1")
        if col == "su_attempted" and number not in (0, 1, 2):
            raise InputValidationError("su_attempted must be 0, 1 or 2")
        row[col] = number
    return row


def _build_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct a model from a saved checkpoint dict and load its weights."""
    if ckpt["kind"] == "binary":
        model = BinaryClassifier(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropouts"])
    else:
        model = MultiClassClassifier(
            ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropouts"], ckpt["output_dim"]
        )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


class Predictor:
    """Loads all artifacts once and serves predictions for single rows or batches."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        art = CONFIG["artifacts"]

        self.scaler = joblib.load(abspath(art["scaler"]))
        self.encoder = joblib.load(abspath(art["encoder"]))
        with open(abspath(art["metadata"]), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.class_names: List[str] = self.metadata.get("class_names", CLASS_NAMES)

        # ``weights_only=True`` (the safe unpickler): our checkpoints hold only
        # tensors plus plain Python metadata (ints/lists/str), all of which the
        # restricted loader accepts. This closes the arbitrary-code-execution
        # path a tampered ``.pt`` file would otherwise open at load time.
        bin_ckpt = torch.load(abspath(art["binary_model"]), map_location=self.device, weights_only=True)
        mc_ckpt = torch.load(abspath(art["multiclass_model"]), map_location=self.device, weights_only=True)
        self.binary_model = _build_from_checkpoint(bin_ckpt, self.device)
        self.multiclass_model = _build_from_checkpoint(mc_ckpt, self.device)

        override = CONFIG["inference"].get("attack_threshold")
        self.threshold = float(
            self.metadata.get("attack_threshold", 0.5) if override is None else override
        )
        if not 0 <= self.threshold <= 1:
            raise ValueError("inference.attack_threshold must be between 0 and 1")

    # ------------------------------------------------------------------
    # Low-level: raw feature matrix -> probabilities
    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray):
        """Return (binary_prob[n], multiclass_prob[n,5]) for a feature matrix."""
        xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            bin_prob = torch.sigmoid(self.binary_model(xt)).view(-1).cpu().numpy()
            mc_prob = F.softmax(self.multiclass_model(xt), dim=1).cpu().numpy()
        return bin_prob, mc_prob

    def predict_matrix(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Batch prediction on an already-preprocessed matrix.

        Returns a dict with binary probabilities, multi-class probabilities and
        the final layered class index per sample.
        """
        bin_prob, mc_prob = self._forward(X)
        is_attack = bin_prob >= self.threshold

        # Among attack classes only (indices 1..4), pick the most likely type.
        attack_choice = mc_prob[:, 1:].argmax(axis=1) + 1
        final = np.where(is_attack, attack_choice, 0).astype(np.int64)

        return {
            "binary_prob": bin_prob,
            "multiclass_prob": mc_prob,
            "final_index": final,
            "is_attack": is_attack,
        }

    # ------------------------------------------------------------------
    # High-level: a dict of the 41 raw features -> friendly result
    # ------------------------------------------------------------------
    def predict_features(self, features: Dict[str, object]) -> Dict[str, object]:
        """Predict from a dict of the 41 raw NSL-KDD features.

        Missing numeric fields default to 0; missing categoricals default to the
        first known category. Invalid values raise InputValidationError.
        """
        row = validate_features(features, self.metadata)
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        X = transform(df, self.scaler, self.encoder)
        if not np.isfinite(X).all():
            raise InputValidationError("feature values exceed the supported range")
        out = self.predict_matrix(X)
        if not np.isfinite(out["binary_prob"]).all() or not np.isfinite(out["multiclass_prob"]).all():
            raise InputValidationError("feature values exceed the supported range")

        idx = int(out["final_index"][0])
        label = self.class_names[idx]
        bin_prob = float(out["binary_prob"][0])
        mc_prob = out["multiclass_prob"][0]

        # Confidence is an uncalibrated model score: for Normal, 1 - binary
        # score; for an attack, the selected multi-class softmax score.
        confidence = (1.0 - bin_prob) if idx == 0 else float(mc_prob[idx])

        return {
            "predicted_class": label,
            "is_attack": bool(idx != 0),
            "confidence": round(confidence, 4),
            "attack_probability": round(bin_prob, 4),
            "class_probabilities": {
                name: round(float(p), 4) for name, p in zip(self.class_names, mc_prob)
            },
        }


# Lazily-initialised process-wide singleton so the Flask app loads models once.
_PREDICTOR: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor()
    return _PREDICTOR
