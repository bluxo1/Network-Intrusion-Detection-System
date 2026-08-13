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

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CONFIG, abspath
from .model import BinaryClassifier, MultiClassClassifier
from .preprocess import transform
from .schema import CLASS_NAMES, FEATURE_COLUMNS


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

        # ``weights_only=False`` because our checkpoint stores plain Python
        # metadata (ints/lists) alongside the tensors. The files are produced by
        # our own trainer, so this is safe.
        bin_ckpt = torch.load(abspath(art["binary_model"]), map_location=self.device, weights_only=False)
        mc_ckpt = torch.load(abspath(art["multiclass_model"]), map_location=self.device, weights_only=False)
        self.binary_model = _build_from_checkpoint(bin_ckpt, self.device)
        self.multiclass_model = _build_from_checkpoint(mc_ckpt, self.device)

        self.threshold = float(CONFIG["inference"]["attack_threshold"])

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
        first known category. Returns a JSON-serialisable result dict.
        """
        row = {}
        for col in FEATURE_COLUMNS:
            if col in self.metadata["categorical_columns"]:
                default = self.metadata["categories"][col][0]
                row[col] = str(features.get(col, default))
            else:
                val = features.get(col, 0)
                try:
                    row[col] = float(val)
                except (TypeError, ValueError):
                    row[col] = 0.0

        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        X = transform(df, self.scaler, self.encoder)
        out = self.predict_matrix(X)

        idx = int(out["final_index"][0])
        label = self.class_names[idx]
        bin_prob = float(out["binary_prob"][0])
        mc_prob = out["multiclass_prob"][0]

        # Confidence: for Normal, how sure we are it is benign; for an attack,
        # the specific-type probability from the multi-class model.
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
