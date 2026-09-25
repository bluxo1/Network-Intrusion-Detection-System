"""Explain R2L errors on KDDTest+ without selecting or changing a model.

Run ``python -m src.r2l_diagnostics`` after downloading the NSL-KDD files and
training. This is an evaluation-only report: KDDTest+ scores must not be used
to choose thresholds, features, hyperparameters, or model checkpoints.
"""

import hashlib
import json
import os
from collections import Counter

import numpy as np

from .config import CONFIG, abspath
from .predict import Predictor
from .preprocess import load_raw, transform
from .schema import CLASS_NAMES, map_label_to_class


QUANTILES = (0, 10, 25, 50, 75, 90, 100)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scores(values):
    """Summarize a fixed set of scores; leave empty groups explicitly null."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "mean": float(values.mean()),
        "quantiles": {
            f"p{q}": float(v)
            for q, v in zip(QUANTILES, np.percentile(values, QUANTILES))
        },
    }


def analyze_r2l(labels, true_classes, binary_scores, family_scores,
                threshold, training_subtypes):
    """Return the two-stage R2L error breakdown from aligned model outputs.

    ``family_scores`` has all five class columns. The deployed predictor ignores
    its Normal column when the binary gate passes, so this does too.
    """
    labels = np.asarray(labels, dtype=str)
    true_classes = np.asarray(true_classes, dtype=str)
    binary_scores = np.asarray(binary_scores, dtype=np.float64)
    family_scores = np.asarray(family_scores, dtype=np.float64)
    count = len(labels)
    if (true_classes.shape != (count,) or binary_scores.shape != (count,)
            or family_scores.shape != (count, len(CLASS_NAMES))):
        raise ValueError("labels, classes, and model scores must have aligned rows")
    if not 0 <= threshold <= 1 or not np.isfinite(binary_scores).all() or not np.isfinite(family_scores).all():
        raise ValueError("threshold and model scores must be finite and valid")

    r2l = true_classes == "R2L"
    normal = true_classes == "Normal"
    passes = binary_scores >= threshold
    attack_choice = np.asarray(CLASS_NAMES)[family_scores[:, 1:].argmax(axis=1) + 1]
    gate_miss = r2l & ~passes
    family_error = r2l & passes & (attack_choice != "R2L")
    correct = r2l & passes & (attack_choice == "R2L")

    subtypes = {}
    for subtype in sorted(set(labels[r2l])):
        mask = r2l & (labels == subtype)
        routed = mask & passes
        subtypes[subtype] = {
            "support": int(mask.sum()),
            "seen_in_training_file": subtype in training_subtypes,
            "gate_misses": int((mask & gate_miss).sum()),
            "gate_passes": int(routed.sum()),
            "family_errors_after_gate": int((mask & family_error).sum()),
            "correct_r2l": int((mask & correct).sum()),
            "recall": float((mask & correct).sum() / mask.sum()),
            "binary_scores": _scores(binary_scores[mask]),
            "gate_miss_binary_scores": _scores(binary_scores[mask & gate_miss]),
            "family_choices_after_gate": dict(sorted(Counter(attack_choice[routed]).items())),
        }

    total_r2l = int(r2l.sum())
    gate_passes = int((r2l & passes).sum())
    return {
        "threshold": float(threshold),
        "test_rows": count,
        "normal_rows": int(normal.sum()),
        "normal_false_alarms": int((normal & passes).sum()),
        "normal_false_alarm_rate": float((normal & passes).sum() / normal.sum()) if normal.any() else None,
        "r2l_rows": total_r2l,
        "gate_misses": int(gate_miss.sum()),
        "gate_passes": gate_passes,
        "gate_recall": float(gate_passes / total_r2l) if total_r2l else None,
        "family_errors_after_gate": int(family_error.sum()),
        "family_accuracy_after_gate": float(correct.sum() / gate_passes) if gate_passes else None,
        "correct_r2l": int(correct.sum()),
        "end_to_end_r2l_recall": float(correct.sum() / total_r2l) if total_r2l else None,
        "family_choices_after_gate": dict(sorted(Counter(attack_choice[r2l & passes]).items())),
        "binary_score_distributions": {
            "normal_all": _scores(binary_scores[normal]),
            "r2l_all": _scores(binary_scores[r2l]),
            "r2l_gate_misses": _scores(binary_scores[gate_miss]),
            "r2l_family_errors_after_gate": _scores(binary_scores[family_error]),
            "r2l_correct": _scores(binary_scores[correct]),
        },
        "r2l_family_score_distributions_after_gate": {
            "correct_r2l_score": _scores(family_scores[correct, CLASS_NAMES.index("R2L")]),
            "wrong_r2l_score": _scores(family_scores[family_error, CLASS_NAMES.index("R2L")]),
        },
        "subtypes": subtypes,
    }


def _markdown(report):
    r = report["results"]
    lines = [
        "# KDDTest+ R2L error analysis",
        "",
        "**Evaluation only.** KDDTest+ is a held-out test set. Do not use this report to select a threshold, feature set, checkpoint, or hyperparameters; make those choices using training/validation data.",
        "",
        f"Binary threshold: {r['threshold']:.6f} (saved/active inference threshold).",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
        f"| True R2L records | {r['r2l_rows']} |",
        f"| Rejected by binary gate as Normal | {r['gate_misses']} |",
        f"| Passed binary gate | {r['gate_passes']} |",
        f"| Passed gate, assigned wrong attack family | {r['family_errors_after_gate']} |",
        f"| Correctly assigned R2L | {r['correct_r2l']} |",
        "",
        f"R2L gate recall: {r['gate_recall']:.1%}; conditional family accuracy: {r['family_accuracy_after_gate']:.1%}; end-to-end R2L recall: {r['end_to_end_r2l_recall']:.1%}.",
        f"Normal false alarms at the same threshold: {r['normal_false_alarms']}/{r['normal_rows']} ({r['normal_false_alarm_rate']:.1%}).",
        "",
        "## Attack subtypes",
        "",
        "'Seen' means the raw subtype occurs anywhere in KDDTrain+; the validation split is not distinguished here.",
        "",
        "| Subtype | Seen | Rows | Gate misses | Family errors | Correct R2L | Recall | Median binary score |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subtype, row in sorted(r["subtypes"].items(), key=lambda pair: (-pair[1]["support"], pair[0])):
        median = row["binary_scores"]["quantiles"]["p50"]
        lines.append(f"| {subtype} | {'yes' if row['seen_in_training_file'] else 'no'} | {row['support']} | {row['gate_misses']} | {row['family_errors_after_gate']} | {row['correct_r2l']} | {row['recall']:.1%} | {median:.3f} |")
    lines += [
        "",
        "The companion JSON contains score quantiles by outcome and subtype, plus the attack-family choices after the gate. Scores are model outputs, not calibrated probabilities.",
        "",
        "## Provenance",
        "",
    ]
    for key, value in report["source_sha256"].items():
        lines.append(f"- `{key}` SHA-256: `{value}`")
    return "\n".join(lines) + "\n"


def main():
    train_path = abspath(CONFIG["paths"]["train_file"])
    test_path = abspath(CONFIG["paths"]["test_file"])
    train = load_raw(train_path)
    test = load_raw(test_path)
    predictor = Predictor(device="cpu")
    X = transform(test, predictor.scaler, predictor.encoder)
    outputs = predictor.predict_matrix(X)
    labels = test["label"].astype(str).str.strip().str.lower().to_numpy()
    classes = np.asarray([map_label_to_class(label) for label in labels])
    training_subtypes = set(train["label"].astype(str).str.strip().str.lower())
    results = analyze_r2l(labels, classes, outputs["binary_prob"],
                          outputs["multiclass_prob"], predictor.threshold,
                          training_subtypes)
    provenance = {"KDDTrain+.txt": train_path, "KDDTest+.txt": test_path}
    for name, artifact in CONFIG["artifacts"].items():
        provenance[name] = abspath(artifact)
    report = {
        "dataset": "NSL-KDD KDDTest+",
        "purpose": "evaluation only; never tune on KDDTest+",
        "source_sha256": {name: _sha256(path) for name, path in provenance.items()},
        "results": results,
    }
    reports_dir = abspath("reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "r2l_diagnostics.json")
    md_path = os.path.join(reports_dir, "r2l_diagnostics.md")
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    with open(md_path, "w", encoding="utf-8") as output:
        output.write(_markdown(report))
    print(f"R2L: {results['correct_r2l']}/{results['r2l_rows']} correct; "
          f"{results['gate_misses']} gate misses; "
          f"{results['family_errors_after_gate']} family errors")
    print(f"Reports: {json_path}, {md_path}")


if __name__ == "__main__":
    main()
