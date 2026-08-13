"""
Form/UI helpers for the Flask layer.

Defines how the 41 NSL-KDD features are grouped and rendered in the web form,
sensible defaults for a benign connection, and a parser that turns submitted
form data into the plain ``{feature: value}`` dict the ``Predictor`` expects.

The actual scaling/encoding lives in ``src.preprocess.transform`` and is reused
by the predictor, so there is a single preprocessing implementation.
"""

from typing import Dict, List

from src.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

# --- Feature groups mirror the NSL-KDD documentation and the project spec ---
FEATURE_GROUPS: Dict[str, List[str]] = {
    "Basic connection": [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    ],
    "Content": [
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login",
    ],
    "Time-based traffic": [
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    ],
    "Host-based traffic": [
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    ],
}

# Binary 0/1 flag features (rendered with a 0/1 constraint).
BINARY_FLAG_FEATURES = {
    "land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login",
}

# Default values describing a typical benign HTTP connection. Pre-filling the
# form makes the 41-field UI usable instead of daunting.
DEFAULTS: Dict[str, object] = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 215, "dst_bytes": 45076, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
    "is_guest_login": 0, "count": 1, "srv_count": 1, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 9, "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.11,
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}


def is_rate(feature: str) -> bool:
    """Rate features are bounded in [0, 1] and rendered with a 0.01 step."""
    return feature.endswith("rate")


def build_field_specs(metadata: dict) -> List[dict]:
    """Build a render-ready spec for every grouped feature.

    Each spec carries the input type, default and (for categoricals) the list of
    valid options pulled from the fitted encoder's ``metadata['categories']``.
    """
    categories = metadata.get("categories", {})
    groups = []
    for group_name, features in FEATURE_GROUPS.items():
        fields = []
        for feat in features:
            spec = {"name": feat, "default": DEFAULTS.get(feat, 0), "label": feat}
            if feat in CATEGORICAL_COLUMNS:
                spec["type"] = "select"
                spec["options"] = categories.get(feat, [])
            elif is_rate(feat):
                spec.update(type="number", step="0.01", min="0", max="1")
            elif feat in BINARY_FLAG_FEATURES:
                spec.update(type="number", step="1", min="0", max="1")
            else:
                spec.update(type="number", step="1", min="0")
            fields.append(spec)
        groups.append({"name": group_name, "fields": fields})
    return groups


def parse_form(form) -> Dict[str, object]:
    """Extract the 41 feature values from a submitted Flask form.

    Values are returned as raw strings/numbers; type coercion and default-filling
    for any omitted field happen inside ``Predictor.predict_features``.
    """
    features: Dict[str, object] = {}
    for feat in FEATURE_COLUMNS:
        if feat in form and str(form.get(feat)).strip() != "":
            features[feat] = form.get(feat)
        else:
            features[feat] = DEFAULTS.get(feat, 0)
    return features
