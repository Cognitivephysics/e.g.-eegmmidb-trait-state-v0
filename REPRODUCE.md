# Reproduce EEGMMIDB trait–state v0

This repository is **standalone** (flat layout). Paths below are relative to the repo root.

| Item | Path |
|------|------|
| Analysis (frozen v0) | `run_experiment.py` |
| Figures | `plot_measurements.py` |
| Seal / hashes | `FREEZE.json` |
| Published numbers | `runs/RESULT_LATEST.json` |
| Reader digest | `runs/RESULT_DIGEST.md` |

## Steps

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-eeg.txt

python run_experiment.py --full --no-prefetch
python plot_measurements.py --raw
python seal_freeze.py
```

Network is required for the first PhysioNet download. Compare your `runs/RESULT_LATEST.json` to the bundled reference (verdict, `delta_d_pr`, bootstrap CI).

Verify code hashes against `FREEZE.json`:

```bash
python3 -c "import json; f=json.load(open('FREEZE.json')); print(f['files_sha256']['run_experiment.py'])"
shasum -a 256 run_experiment.py
```

## What we do *not* claim

Reproduction confirms the **pipeline on open data**, not validation of “Cognitive Physics” as physics. See `runs/RESULT_DIGEST.md`.
