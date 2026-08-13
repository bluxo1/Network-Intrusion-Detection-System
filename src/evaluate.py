"""
Evaluate the trained models on two splits:

1. **In-distribution validation** - the same 15% held out of KDDTrain+ during
   training. This is what the "98%+ detection rate / minimal false alarms"
   target refers to.
2. **KDDTest+** - the official test set, which deliberately introduces novel
   attack variants (especially R2L/U2R) absent from training. Accuracy here is
   the honest measure of generalisation and, as widely documented, sits far
   below the in-distribution figure for every model class.

For each split we report the binary detection metrics (accuracy, precision,
recall = detection rate, F1, ROC-AUC, false-alarm rate) and the multi-class
per-class precision/recall/F1 + confusion matrix.

Outputs ``reports/metrics.json`` and, if plotting libs are available,
``reports/confusion_matrix.png`` for the KDDTest+ split.

Run with:  ``python -m src.evaluate``
"""

import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import CONFIG, abspath
from .predict import Predictor
from .schema import CLASS_NAMES


def _load_arrays():
    npz = os.path.join(abspath(CONFIG["paths"]["processed_dir"]), "dataset.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"{npz} not found. Run `python -m src.preprocess` (and `python -m src.train`) first."
        )
    return np.load(npz)


def _val_split(X_train, y_train, yb_train):
    """Reconstruct the exact stratified validation split used by src/train.py."""
    tcfg = CONFIG["training"]
    idx = np.arange(len(X_train))
    _, val_idx = train_test_split(
        idx, test_size=tcfg["val_split"], random_state=tcfg["seed"], stratify=y_train
    )
    return X_train[val_idx], y_train[val_idx], yb_train[val_idx]


def _binary_metrics(yb, is_attack, bin_prob):
    normal_mask = yb == 0
    try:
        auc = roc_auc_score(yb, bin_prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(yb, is_attack)),
        "precision": float(precision_score(yb, is_attack, zero_division=0)),
        "recall_detection_rate": float(recall_score(yb, is_attack, zero_division=0)),
        "f1": float(f1_score(yb, is_attack, zero_division=0)),
        "roc_auc": float(auc),
        "false_alarm_rate": float(is_attack[normal_mask].mean()) if normal_mask.any() else float("nan"),
    }


def _save_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Best-effort heatmap; silently skipped if plotting libs are missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover
        print(f"[evaluate] skipping confusion-matrix plot ({exc})")
        return

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.title("NIDS confusion matrix (KDDTest+)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[evaluate] saved confusion matrix -> {out_path}")


def evaluate_split(predictor: Predictor, X, y, yb, name: str, cm_path: str = None) -> dict:
    """Evaluate both models on one split, print a report, return a metrics dict."""
    out = predictor.predict_matrix(X)
    final = out["final_index"]
    is_attack = out["is_attack"].astype(int)

    bin_m = _binary_metrics(yb, is_attack, out["binary_prob"])
    labels = list(range(len(CLASS_NAMES)))
    mc_acc = float(accuracy_score(y, final))
    report = classification_report(y, final, labels=labels, target_names=CLASS_NAMES,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y, final, labels=labels)

    print(f"\n############ {name} ############")
    print("--- Binary (Normal vs Attack) ---")
    for k in ["accuracy", "precision", "recall_detection_rate", "f1", "roc_auc", "false_alarm_rate"]:
        print(f"  {k:22s}: {bin_m[k]:.4f}")
    print("--- Multi-class (layered) ---")
    print(f"  accuracy              : {mc_acc:.4f}")
    print(classification_report(y, final, labels=labels, target_names=CLASS_NAMES, zero_division=0))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(cm)

    if cm_path:
        _save_confusion_matrix(cm, cm_path)

    return {
        "binary": bin_m,
        "multiclass": {"accuracy": mc_acc, "per_class": report, "confusion_matrix": cm.tolist()},
    }


def main() -> None:
    data = _load_arrays()
    predictor = Predictor()

    # In-distribution validation split (the 98%+ target regime).
    Xv, yv, ybv = _val_split(data["X_train"], data["y_train"], data["yb_train"])
    val_metrics = evaluate_split(predictor, Xv, yv, ybv, "IN-DISTRIBUTION VALIDATION (15% of KDDTrain+)")

    # Official KDDTest+ (cross-distribution generalisation).
    reports_dir = abspath("reports")
    os.makedirs(reports_dir, exist_ok=True)
    test_metrics = evaluate_split(
        predictor, data["X_test"], data["y_test"], data["yb_test"],
        "KDDTest+ (official test set, contains novel attacks)",
        cm_path=os.path.join(reports_dir, "confusion_matrix.png"),
    )

    with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"in_distribution_validation": val_metrics, "kddtest_plus": test_metrics}, f, indent=2)
    print(f"\n[evaluate] saved metrics -> {os.path.join(reports_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
