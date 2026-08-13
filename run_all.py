"""
One-shot pipeline: download -> preprocess -> train -> evaluate.

Convenience wrapper so you can go from an empty checkout to trained, evaluated
models with a single command:

    python run_all.py

Each stage is skipped-friendly: preprocessing regenerates only if needed, and
training/evaluation always use the cached processed arrays.
"""

import runpy
import sys

from src import evaluate, preprocess, train


def main() -> int:
    print("=" * 70)
    print("STEP 1/4  Download NSL-KDD")
    print("=" * 70)
    # download_data.py is a standalone script; run it in-process.
    try:
        runpy.run_path("data/download_data.py", run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            print("Download step reported a problem; continuing if data already exists.")

    print("\n" + "=" * 70)
    print("STEP 2/4  Preprocess")
    print("=" * 70)
    preprocess.run()

    print("\n" + "=" * 70)
    print("STEP 3/4  Train")
    print("=" * 70)
    train.main()

    print("\n" + "=" * 70)
    print("STEP 4/4  Evaluate")
    print("=" * 70)
    evaluate.main()

    print("\nAll done. Start the web app with:  python app/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
