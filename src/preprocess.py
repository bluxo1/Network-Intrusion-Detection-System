"""
Data loading and preprocessing for NSL-KDD.

Responsibilities
----------------
1. Read the raw ``KDDTrain+.txt`` / ``KDDTest+.txt`` files.
2. Map the raw attack label to the 5 canonical classes.
3. One-hot encode the categorical columns and z-score scale the numeric ones,
   fitting **only on the training split** and reusing those fitted objects
   everywhere else (train/val/test/inference) so the pipeline is identical.
4. Persist the fitted encoder, scaler and a metadata JSON so the Flask app can
   reproduce the exact same transformation at inference time.

Running this module as a script performs the full fit + transform + save and
writes processed arrays to ``data/processed/`` for the trainer to consume.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, abspath
from .schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    map_label_to_class,
)


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    """Load a raw NSL-KDD file into a DataFrame with named columns.

    The files are comma-separated with no header; the last two columns are the
    attack label and a difficulty score. The difficulty score is dropped.
    """
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    df = df.drop(columns=["difficulty"])
    return df


def add_class_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a canonical ``class`` column (string) derived from the raw label."""
    df = df.copy()
    df["class"] = df["label"].apply(map_label_to_class)
    return df


def _class_to_index(series: pd.Series) -> np.ndarray:
    """Encode canonical class names to fixed integer indices (see CLASS_NAMES)."""
    mapping = {name: i for i, name in enumerate(CLASS_NAMES)}
    return series.map(mapping).to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Fitting the preprocessors
# ---------------------------------------------------------------------------
def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder that tolerates categories unseen during fit."""
    # sklearn >= 1.2 uses ``sparse_output``; older versions use ``sparse``.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_preprocessors(train_df: pd.DataFrame):
    """Fit the scaler and encoder on the training data.

    Returns ``(scaler, encoder, metadata)``.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLUMNS].to_numpy(dtype=np.float64))

    encoder = _make_encoder()
    encoder.fit(train_df[CATEGORICAL_COLUMNS])

    # Human-readable names for the one-hot columns, e.g. "protocol_type=tcp".
    onehot_names = []
    for col, cats in zip(CATEGORICAL_COLUMNS, encoder.categories_):
        onehot_names.extend([f"{col}={c}" for c in cats])

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoded_feature_names": list(NUMERIC_COLUMNS) + onehot_names,
        "input_dim": len(NUMERIC_COLUMNS) + len(onehot_names),
        "class_names": CLASS_NAMES,
        # Category lists power the dropdowns in the web form.
        "categories": {
            col: [str(c) for c in cats]
            for col, cats in zip(CATEGORICAL_COLUMNS, encoder.categories_)
        },
    }
    return scaler, encoder, metadata


def transform(df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder) -> np.ndarray:
    """Apply the fitted scaler + encoder and return the final feature matrix.

    Column order is always ``[scaled numeric ... | one-hot categorical ...]`` to
    match ``metadata['encoded_feature_names']``.
    """
    numeric = scaler.transform(df[NUMERIC_COLUMNS].to_numpy(dtype=np.float64))
    categorical = encoder.transform(df[CATEGORICAL_COLUMNS])
    return np.hstack([numeric, categorical]).astype(np.float32)


# ---------------------------------------------------------------------------
# Script entry point: full pipeline -> saved artifacts + processed arrays
# ---------------------------------------------------------------------------
def run() -> None:
    train_path = abspath(CONFIG["paths"]["train_file"])
    test_path = abspath(CONFIG["paths"]["test_file"])

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "NSL-KDD files not found.\n"
            f"  expected: {train_path}\n"
            f"            {test_path}\n"
            "Run `python data/download_data.py` first to fetch the dataset."
        )

    print(f"[preprocess] loading train: {train_path}")
    train_df = add_class_column(load_raw(train_path))
    print(f"[preprocess] loading test : {test_path}")
    test_df = add_class_column(load_raw(test_path))

    print(f"[preprocess] train rows={len(train_df)}  test rows={len(test_df)}")
    print("[preprocess] class distribution (train):")
    print(train_df["class"].value_counts().to_string())

    scaler, encoder, metadata = fit_preprocessors(train_df)
    print(f"[preprocess] encoded input dimension = {metadata['input_dim']}")

    X_train = transform(train_df, scaler, encoder)
    X_test = transform(test_df, scaler, encoder)

    # Multi-class integer targets (0..4).
    y_train = _class_to_index(train_df["class"])
    y_test = _class_to_index(test_df["class"])
    # Binary targets: 0 = Normal, 1 = Attack.
    yb_train = (y_train != 0).astype(np.float32)
    yb_test = (y_test != 0).astype(np.float32)

    # ---- persist artifacts -------------------------------------------------
    models_dir = abspath(CONFIG["paths"]["models_dir"])
    processed_dir = abspath(CONFIG["paths"]["processed_dir"])
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    joblib.dump(scaler, abspath(CONFIG["artifacts"]["scaler"]))
    joblib.dump(encoder, abspath(CONFIG["artifacts"]["encoder"]))
    # The "label encoder" is just the ordered class-name list; store it too so
    # the app never has to hard-code the ordering.
    joblib.dump(CLASS_NAMES, abspath(CONFIG["artifacts"]["label_encoder"]))
    with open(abspath(CONFIG["artifacts"]["metadata"]), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.savez_compressed(
        os.path.join(processed_dir, "dataset.npz"),
        X_train=X_train,
        y_train=y_train,
        yb_train=yb_train,
        X_test=X_test,
        y_test=y_test,
        yb_test=yb_test,
    )

    print(f"[preprocess] saved scaler/encoder/metadata to {models_dir}")
    print(f"[preprocess] saved processed arrays to {processed_dir}/dataset.npz")


if __name__ == "__main__":
    run()
