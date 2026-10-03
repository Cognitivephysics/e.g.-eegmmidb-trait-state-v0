#!/usr/bin/env python3
"""Write FREEZE.json — code hashes + result hash + env pins."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = (
    "run_experiment.py",
    "plot_measurements.py",
    "requirements-eeg.txt",
    "README.md",
    "REPRODUCE.md",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _py_versions() -> dict:
    out = {}
    for mod in ("mne", "numpy", "scipy", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except ImportError:
            out[mod] = None
    out["python"] = platform.python_version()
    return out


def main() -> None:
    result = HERE / "runs" / "RESULT_LATEST.json"
    freeze = {
        "protocol": "EEGMMIDB_TRAIT_STATE_v0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {name: _sha(HERE / name) for name in FILES if (HERE / name).is_file()},
        "result_sha256": _sha(result) if result.is_file() else None,
        "result_path": str(result),
        "environment": _py_versions(),
        "dataset": "PhysioNet eegmmidb 1.0.0 R01",
        "reproduce": "See REPRODUCE.md; run: python run_experiment.py --full --no-prefetch",
    }
    out = HERE / "FREEZE.json"
    out.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), **freeze}, indent=2))


if __name__ == "__main__":
    main()
