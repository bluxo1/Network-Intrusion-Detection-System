"""Checks that R2L errors are attributed to the correct inference stage."""

import numpy as np

from src.r2l_diagnostics import analyze_r2l


def test_r2l_stage_attribution_and_subtype_counts():
    labels = ["guess_passwd", "guess_passwd", "snmpguess", "normal", "normal"]
    classes = ["R2L", "R2L", "R2L", "Normal", "Normal"]
    binary = np.array([0.1, 0.7, 0.9, 0.2, 0.6])
    # Columns follow Normal, DOS, PROBE, R2L, U2R. Normal must be ignored
    # after the binary gate passes, even if its softmax score is largest.
    family = np.array([
        [0.8, 0.05, 0.05, 0.08, 0.02],
        [0.7, 0.05, 0.04, 0.2, 0.01],
        [0.05, 0.7, 0.1, 0.1, 0.05],
        [0.6, 0.1, 0.1, 0.1, 0.1],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    ])
    result = analyze_r2l(labels, classes, binary, family, 0.5, {"guess_passwd"})

    assert result["r2l_rows"] == 3
    assert result["gate_misses"] == 1
    assert result["gate_passes"] == 2
    assert result["family_errors_after_gate"] == 1
    assert result["correct_r2l"] == 1
    assert result["end_to_end_r2l_recall"] == 1 / 3
    assert result["normal_false_alarms"] == 1
    assert result["subtypes"]["guess_passwd"]["seen_in_training_file"] is True
    assert result["subtypes"]["snmpguess"]["seen_in_training_file"] is False
    assert result["family_choices_after_gate"] == {"DOS": 1, "R2L": 1}
    assert result["binary_score_distributions"]["r2l_gate_misses"]["quantiles"]["p50"] == 0.1
