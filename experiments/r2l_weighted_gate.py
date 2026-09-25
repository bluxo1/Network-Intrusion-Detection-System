"""One-shot comparison of the deployed gate and an R2L-weighted tree gate.

KDDTrain+ supplies the existing train/validation split and all threshold
selection. KDDTest+ is read only after the candidate fit and threshold are
fixed. This file does not replace any deployed artifact.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve

from src.config import CONFIG, abspath
from src.predict import Predictor


def select_threshold(y_binary: np.ndarray, scores: np.ndarray, ceiling: float) -> float:
    """Match src.train's validation-only F2 choice under the FAR ceiling."""
    false_alarm, recall, thresholds = roc_curve(y_binary, scores)
    positives = float(y_binary.sum())
    negatives = float(len(y_binary) - positives)
    tp = recall * positives
    fp = false_alarm * negatives
    fn = positives - tp
    denominator = 5 * tp + 4 * fn + fp
    f2 = np.divide(5 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    eligible = np.flatnonzero((false_alarm <= ceiling) & np.isfinite(thresholds))
    if len(eligible) == 0:
        raise ValueError("No threshold satisfies the validation false-alarm ceiling")
    return float(thresholds[eligible[np.argmax(f2[eligible])]])


def metrics(y_binary: np.ndarray, y_five: np.ndarray, scores: np.ndarray,
            threshold: float, attack_choice: np.ndarray) -> dict:
    flagged = scores >= threshold
    final = np.where(flagged, attack_choice, 0)
    attack = y_binary == 1
    normal = ~attack
    r2l = y_five == 3
    return {
        "threshold": threshold,
        "attack_recall": float(flagged[attack].mean()),
        "r2l_gate_recall": float(flagged[r2l].mean()),
        "r2l_layered_recall": float((final[r2l] == 3).mean()),
        "five_class_accuracy": float((final == y_five).mean()),
        "normal_false_alarm_rate": float(flagged[normal].mean()),
        "true_attacks": int(attack.sum()),
        "detected_attacks": int(flagged[attack].sum()),
        "true_r2l": int(r2l.sum()),
        "detected_r2l": int(flagged[r2l].sum()),
        "correct_r2l": int((final[r2l] == 3).sum()),
        "true_normal": int(normal.sum()),
        "false_alarms": int(flagged[normal].sum()),
    }


def main() -> None:
    data_path = Path(abspath(CONFIG["paths"]["processed_dir"])) / "dataset.npz"
    with np.load(data_path) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        yb_train = data["yb_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        yb_val = data["yb_val"]
        # Deliberately do not load X_test until model/threshold are frozen.

    sample_weight = np.ones(len(y_train), dtype=np.float64)
    sample_weight[y_train == 3] = 10.0
    candidate = HistGradientBoostingClassifier(
        loss="log_loss", max_iter=100, max_leaf_nodes=31,
        learning_rate=0.1, l2_regularization=1.0,
        early_stopping=False, random_state=42,
    )
    candidate.fit(X_train, yb_train.astype(int), sample_weight=sample_weight)

    candidate_val = candidate.predict_proba(X_val)[:, 1]
    ceiling = float(CONFIG["inference"]["max_validation_false_alarm_rate"])
    candidate_threshold = select_threshold(yb_val, candidate_val, ceiling)

    baseline = Predictor(device="cpu")
    baseline_val_out = baseline.predict_matrix(X_val)
    baseline_val = baseline_val_out["binary_prob"]
    val_attack_choice = baseline_val_out["multiclass_prob"][:, 1:].argmax(axis=1) + 1
    baseline_threshold = float(baseline.threshold)
    validation = {
        "deployed_mlp": metrics(yb_val, y_val, baseline_val, baseline_threshold, val_attack_choice),
        "r2l_weighted_hgb": metrics(yb_val, y_val, candidate_val, candidate_threshold, val_attack_choice),
    }

    # Selection is complete. No threshold/model change follows test evaluation.
    with np.load(data_path) as data:
        X_test = data["X_test"]
        y_test = data["y_test"]
        yb_test = data["yb_test"]
    candidate_test = candidate.predict_proba(X_test)[:, 1]
    baseline_test_out = baseline.predict_matrix(X_test)
    baseline_test = baseline_test_out["binary_prob"]
    test_attack_choice = baseline_test_out["multiclass_prob"][:, 1:].argmax(axis=1) + 1
    result = {
        "protocol": {
            "training": "KDDTrain+ 85% stratified split; preprocessors fit on training only",
            "selection": "KDDTrain+ 15% validation; maximize F2 with FAR <= 1%",
            "final_evaluation": "KDDTest+ after candidate/threshold locked; rerun for layered metrics without selection changes",
            "candidate": "HistGradientBoostingClassifier(loss=log_loss, max_iter=100, max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1.0, early_stopping=False, random_state=42); R2L sample weight 10, all other rows 1",
        },
        "validation": validation,
        "kddtest_plus": {
            "deployed_mlp": metrics(yb_test, y_test, baseline_test, baseline_threshold, test_attack_choice),
            "r2l_weighted_hgb": metrics(yb_test, y_test, candidate_test, candidate_threshold, test_attack_choice),
        },
    }
    out_path = Path("experiments/r2l_weighted_gate_results.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
