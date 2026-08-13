"""
NSL-KDD schema definitions.

This module is the single source of truth for:
  * the 41 feature column names (+ label + difficulty),
  * which columns are categorical vs numeric,
  * the mapping from raw attack labels to the 5 canonical classes.

Both the training pipeline (``src/``) and the Flask app (``app/``) import from
here so the two can never drift apart.
"""

# ---------------------------------------------------------------------------
# Column layout of KDDTrain+.txt / KDDTest+.txt
# 41 features, then the attack label, then a "difficulty" score we discard.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# The two trailing columns in the raw files.
ALL_COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"]

# The three symbolic (categorical) features.
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Everything else is numeric.
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# ---------------------------------------------------------------------------
# Canonical 5-class taxonomy.
# Index positions are fixed and used everywhere as the label encoding:
#   0=Normal, 1=DOS, 2=PROBE, 3=R2L, 4=U2R
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "DOS", "PROBE", "R2L", "U2R"]

# Raw label -> canonical class. Covers every attack that appears in either
# KDDTrain+ or KDDTest+ (the test set introduces novel attacks not in train).
ATTACK_CATEGORY = {
    # --- Normal ---
    "normal": "Normal",
    # --- Denial of Service ---
    "back": "DOS",
    "land": "DOS",
    "neptune": "DOS",
    "pod": "DOS",
    "smurf": "DOS",
    "teardrop": "DOS",
    "mailbomb": "DOS",
    "apache2": "DOS",
    "processtable": "DOS",
    "udpstorm": "DOS",
    "worm": "DOS",
    # --- Probe ---
    "ipsweep": "PROBE",
    "nmap": "PROBE",
    "portsweep": "PROBE",
    "satan": "PROBE",
    "mscan": "PROBE",
    "saint": "PROBE",
    # --- Remote to Local ---
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    # --- User to Root ---
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def map_label_to_class(raw_label: str) -> str:
    """Map a raw NSL-KDD label to one of the 5 canonical class names.

    Unknown labels are conservatively treated as an attack ("DOS") rather than
    silently normal, so a never-before-seen label is never dismissed.
    """
    return ATTACK_CATEGORY.get(str(raw_label).strip().lower(), "DOS")
