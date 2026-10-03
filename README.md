# EEGMMIDB trait–state v0 (reproducible)

Open-data test: **PhysioNet EEG Motor Movement/Imagery**, run **R01** (eyes-open), preregistered **10-D** features, between- vs within-unit covariance geometry and **participation-ratio** \(d_{PR}\).

**Primary result (N=78 after QC):** \(\widehat{\Delta d_{PR}} \approx 2.47\), bootstrap 95% CI **[1.92, 3.01]** — see [`runs/RESULT_LATEST.json`](runs/RESULT_LATEST.json) and [`runs/RESULT_DIGEST.md`](runs/RESULT_DIGEST.md).

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-eeg.txt

python run_experiment.py --full --no-prefetch
python plot_measurements.py --raw
python seal_freeze.py
```

Network required for first PhysioNet download. Compare your `runs/RESULT_LATEST.json` to the bundled reference.

Details: **[REPRODUCE.md](REPRODUCE.md)** · Code seal: **[FREEZE.json](FREEZE.json)**

## Files

| File | Purpose |
|------|---------|
| `paper/Cognitive_Physics_Measurement_Program_Oct2026.pdf` | Preregistration / measurement proposal (Oct 2026) |
| `run_experiment.py` | Frozen analysis v0 |
| `plot_measurements.py` | Figures (`runs/figures/`) |
| `requirements-eeg.txt` | Python dependencies |

## Citation / context

Joel Peña Muñoz Jr. — measurement proposal *Cognitive Physics: A Measurement Proposal* (trait–state decomposition). This repo is **execution-first**; read the digest for **do-not-overclaim** limits.

## License

Add your license (e.g. MIT for code; cite PhysioNet for EEG data).
