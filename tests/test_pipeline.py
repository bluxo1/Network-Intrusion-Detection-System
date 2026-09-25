"""Checks that validation rows cannot influence fitted preprocessing."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG
from src.preprocess import run
from src.schema import ALL_COLUMNS, NUMERIC_COLUMNS


def test_preprocessing_fits_only_training_partition(tmp_path, monkeypatch):
    labels = ["normal", "neptune", "satan", "guess_passwd", "rootkit"] * 10
    classes = np.tile(np.arange(5), 10)
    _, val_idx = train_test_split(
        np.arange(50), test_size=0.2, random_state=42, stratify=classes,
    )
    rows = []
    for index, label in enumerate(labels):
        row = {name: 0 for name in ALL_COLUMNS}
        row.update(duration=index, protocol_type="tcp", service="http", flag="SF",
                   label=label, difficulty=0)
        if index == val_idx[0]:
            row["service"] = "validation_only"
        rows.append(row)
    frame = pd.DataFrame(rows, columns=ALL_COLUMNS)
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"
    frame.to_csv(train_file, index=False, header=False)
    frame.head(10).to_csv(test_file, index=False, header=False)

    monkeypatch.setitem(CONFIG["paths"], "train_file", str(train_file))
    monkeypatch.setitem(CONFIG["paths"], "test_file", str(test_file))
    monkeypatch.setitem(CONFIG["paths"], "models_dir", str(tmp_path / "models"))
    monkeypatch.setitem(CONFIG["paths"], "processed_dir", str(tmp_path / "processed"))
    monkeypatch.setitem(CONFIG["training"], "val_split", 0.2)
    for name, suffix in (
        ("scaler", "scaler.pkl"), ("encoder", "encoder.pkl"),
        ("label_encoder", "label_encoder.pkl"), ("metadata", "metadata.json"),
    ):
        monkeypatch.setitem(CONFIG["artifacts"], name, str(tmp_path / "models" / suffix))

    run()

    with np.load(tmp_path / "processed" / "dataset.npz") as arrays:
        assert arrays["X_train"].shape[0] == 40
        assert arrays["X_val"].shape[0] == 10
    scaler = joblib.load(tmp_path / "models" / "scaler.pkl")
    with open(tmp_path / "models" / "metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    train_idx, _ = train_test_split(
        np.arange(50), test_size=0.2, random_state=42, stratify=classes,
    )
    duration_position = NUMERIC_COLUMNS.index("duration")
    assert np.isclose(scaler.mean_[duration_position], np.mean(train_idx))
    assert not np.isclose(scaler.mean_[duration_position], np.mean(np.arange(50)))
    assert "validation_only" not in metadata["categories"]["service"]
