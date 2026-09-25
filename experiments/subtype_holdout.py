"""Training-only robustness check for the R2L-weighted binary gate.

Every row comes from KDDTrain+. In each prespecified fold, the named R2L
subtypes are removed from fit and validation entirely, then evaluated with
separate known-subtype traffic. KDDTest+ and deployed artifacts are unused.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.config import CONFIG, abspath
from src.dataset import NIDSDataset
from src.model import BinaryClassifier
from src.preprocess import add_class_column, fit_preprocessors, load_raw, transform
from src.schema import CLASS_NAMES
from src.train import calibrate_threshold, set_seed, train_loop

from experiments.r2l_weighted_gate import select_threshold


# Chosen from KDDTrain+ label counts before seeing this experiment's outcomes.
# The first tests several uncommon R2L signatures; the second holds out the
# dominant training signature while fitting on the remaining R2L subtypes.
FOLDS = {
    "holdout_three_rare": ("guess_passwd", "warezmaster", "imap"),
    "holdout_warezclient": ("warezclient",),
}


def score_mlp(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).view(-1).numpy()


def measure(y: np.ndarray, subtype: np.ndarray, score: np.ndarray,
            threshold: float, held_names: tuple[str, ...]) -> dict:
    flagged = score >= threshold
    attack = y != 0
    normal = ~attack
    held = np.isin(subtype, held_names)
    known_attack = attack & ~held
    return {
        "threshold": float(threshold),
        "attack_recall": float(flagged[attack].mean()),
        "heldout_r2l_recall": float(flagged[held].mean()) if held.any() else None,
        "known_attack_recall": float(flagged[known_attack].mean()),
        "normal_false_alarm_rate": float(flagged[normal].mean()),
        "attack_detected": int(flagged[attack].sum()),
        "attack_total": int(attack.sum()),
        "heldout_r2l_detected": int(flagged[held].sum()),
        "heldout_r2l_total": int(held.sum()),
        "known_attack_detected": int(flagged[known_attack].sum()),
        "known_attack_total": int(known_attack.sum()),
        "normal_false_alarms": int(flagged[normal].sum()),
        "normal_total": int(normal.sum()),
    }


def run_fold(df, labels: np.ndarray, held_names: tuple[str, ...]) -> dict:
    seed = int(CONFIG["training"]["seed"])
    raw_names = df["label"].to_numpy()
    held_mask = np.isin(raw_names, held_names)
    if not held_mask.any() or not np.all(labels[held_mask] == 3):
        raise ValueError("Held-out names must identify existing R2L rows")
    known_idx = np.flatnonzero(~held_mask)
    held_idx = np.flatnonzero(held_mask)
    # The 15% evaluation slice and the 15%-of-remaining validation slice
    # contain only known subtypes. All held-out subtype rows join evaluation.
    remaining_idx, known_eval_idx = train_test_split(
        known_idx, test_size=0.15, random_state=seed, stratify=labels[known_idx],
    )
    fit_idx, val_idx = train_test_split(
        remaining_idx, test_size=0.15, random_state=seed,
        stratify=labels[remaining_idx],
    )
    eval_idx = np.concatenate([known_eval_idx, held_idx])
    assert not np.isin(raw_names[fit_idx], held_names).any()
    assert not np.isin(raw_names[val_idx], held_names).any()
    assert len(np.unique(np.concatenate([fit_idx, val_idx, eval_idx]))) == len(df)

    scaler, encoder, _ = fit_preprocessors(df.iloc[fit_idx])
    X_fit = transform(df.iloc[fit_idx], scaler, encoder)
    X_val = transform(df.iloc[val_idx], scaler, encoder)
    X_eval = transform(df.iloc[eval_idx], scaler, encoder)
    y_fit = labels[fit_idx]
    y_val = labels[val_idx]
    y_eval = labels[eval_idx]
    yb_fit = (y_fit != 0).astype(np.float32)
    yb_val = (y_val != 0).astype(np.float32)

    set_seed(seed)
    mcfg = CONFIG["model"]
    tcfg = CONFIG["training"]
    baseline = BinaryClassifier(
        X_fit.shape[1], mcfg["binary_hidden"], mcfg["binary_dropout"],
    )
    train_loader = DataLoader(
        NIDSDataset(X_fit, yb_fit, binary=True),
        batch_size=tcfg["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        NIDSDataset(X_val, yb_val, binary=True),
        batch_size=tcfg["batch_size"], shuffle=False,
    )
    pos_weight = torch.tensor([(len(yb_fit) - yb_fit.sum()) / yb_fit.sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    baseline = train_loop(
        baseline, train_loader, val_loader, criterion, torch.device("cpu"),
        tcfg, task="binary",
    )
    ceiling = float(CONFIG["inference"]["max_validation_false_alarm_rate"])
    baseline_threshold = calibrate_threshold(
        baseline, X_val, yb_val, torch.device("cpu"), ceiling,
    )

    sample_weight = np.where(y_fit == 3, 10.0, 1.0)
    candidate = HistGradientBoostingClassifier(
        loss="log_loss", max_iter=100, max_leaf_nodes=31,
        learning_rate=0.1, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )
    candidate.fit(X_fit, yb_fit.astype(int), sample_weight=sample_weight)
    candidate_val_score = candidate.predict_proba(X_val)[:, 1]
    candidate_threshold = select_threshold(yb_val, candidate_val_score, ceiling)

    baseline_val_score = score_mlp(baseline, X_val)
    baseline_eval_score = score_mlp(baseline, X_eval)
    candidate_eval_score = candidate.predict_proba(X_eval)[:, 1]
    return {
        "heldout_subtypes": list(held_names),
        "heldout_counts": {name: int((raw_names[held_idx] == name).sum()) for name in held_names},
        "split_rows": {
            "fit": len(fit_idx), "validation": len(val_idx),
            "evaluation_known": len(known_eval_idx),
            "evaluation_heldout": len(held_idx),
        },
        "validation_known_only": {
            "baseline_mlp": measure(y_val, raw_names[val_idx], baseline_val_score,
                                    baseline_threshold, held_names),
            "r2l_weighted_hgb": measure(y_val, raw_names[val_idx], candidate_val_score,
                                        candidate_threshold, held_names),
        },
        "evaluation": {
            "baseline_mlp": measure(y_eval, raw_names[eval_idx], baseline_eval_score,
                                    baseline_threshold, held_names),
            "r2l_weighted_hgb": measure(y_eval, raw_names[eval_idx], candidate_eval_score,
                                        candidate_threshold, held_names),
        },
    }


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    df = add_class_column(load_raw(abspath(CONFIG["paths"]["train_file"])))
    lookup = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    labels = df["class"].map(lookup).to_numpy(dtype=np.int64)
    result = {
        "protocol": {
            "data": "KDDTrain+ only; KDDTest+ and deployed artifacts not read",
            "split": "For each prespecified fold, all held-out subtype rows to evaluation; remaining rows 15% stratified evaluation, then 15% stratified validation, remainder fit; seed 42",
            "preprocessing": "StandardScaler and OneHotEncoder fit on fit rows only",
            "baseline": "Repo BinaryClassifier and src.train train_loop (Adam, StepLR, early stopping, BCE pos_weight), fit from scratch per fold",
            "candidate": "HistGradientBoostingClassifier(loss=log_loss, max_iter=100, max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1.0, early_stopping=False, random_state=42); R2L sample weight 10, others 1",
            "threshold": "Maximize validation F2 under <=1% normal FAR, independently per gate; no held-out subtype row used",
        },
        "folds": {},
    }
    for fold_name, held_names in FOLDS.items():
        print(f"\n=== {fold_name}: {held_names} ===", flush=True)
        result["folds"][fold_name] = run_fold(df, labels, held_names)
        Path("experiments/subtype_holdout_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(result["folds"][fold_name]["evaluation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
