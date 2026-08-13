"""
Download the NSL-KDD dataset (KDDTrain+ / KDDTest+).

The NSL-KDD files are hosted on several public mirrors. This script tries each
mirror in turn and saves the plain comma-separated ``.txt`` files into this
``data/`` directory, where the preprocessing pipeline expects them.

Usage:
    python data/download_data.py
"""

import gzip
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# (filename, [candidate URLs]) - first mirror that returns a valid file wins.
TARGETS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Train.csv",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
        "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Test.csv",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt",
    ],
}

# Each raw record has 41 features + label + difficulty = 43 comma-separated fields.
EXPECTED_FIELDS = 43


def _looks_valid(text: str) -> bool:
    """Sanity-check that the downloaded text is a proper NSL-KDD file."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 1000:
        return False
    n_fields = len(lines[0].split(","))
    return n_fields == EXPECTED_FIELDS


def _fetch(url: str) -> str:
    """Fetch a URL as text, requesting gzip so the ~19 MB files transfer as
    ~2 MB. Uses ``requests`` when available (it auto-decompresses); otherwise
    falls back to ``urllib`` with manual gzip handling.
    """
    headers = {"User-Agent": "nids-pytorch/1.0", "Accept-Encoding": "gzip"}
    try:
        import requests  # optional; listed in requirements.txt

        resp = requests.get(url, headers=headers, timeout=180)
        resp.raise_for_status()
        return resp.text
    except ImportError:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="replace")


def download_one(filename: str, urls) -> bool:
    dest = os.path.join(HERE, filename)
    if os.path.exists(dest) and _looks_valid(open(dest, "r", encoding="utf-8").read()):
        print(f"[download] {filename} already present - skipping")
        return True

    for url in urls:
        try:
            print(f"[download] fetching {filename} from {url}")
            text = _fetch(url)
            if not _looks_valid(text):
                print(f"[download]   -> unexpected format from {url}, trying next mirror")
                continue
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            n_rows = len([ln for ln in text.splitlines() if ln.strip()])
            print(f"[download]   -> saved {filename} ({n_rows} rows)")
            return True
        except Exception as exc:  # noqa: BLE001 - report and try next mirror
            print(f"[download]   -> failed ({exc}); trying next mirror")
    return False


def main() -> int:
    ok = True
    for filename, urls in TARGETS.items():
        if not download_one(filename, urls):
            ok = False
            print(f"[download] ERROR: could not obtain {filename} from any mirror")
    if ok:
        print("\n[download] NSL-KDD ready. Next: python -m src.train")
        return 0
    print(
        "\n[download] Some files could not be downloaded automatically.\n"
        "Download KDDTrain+.txt and KDDTest+.txt manually (e.g. from the UNB\n"
        "NSL-KDD page or a GitHub mirror) and place them in the data/ folder."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
