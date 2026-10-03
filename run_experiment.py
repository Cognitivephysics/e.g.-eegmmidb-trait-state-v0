#!/usr/bin/env python3
"""PhysioNet EEGMMIDB R01 — trait/state covariance geometry (main (7) protocol v0)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "runs"
DATA_DIR = EXP_DIR / "data" / "eegbci"
CACHE_DIR = EXP_DIR / "data" / "feature_cache"
STATUS_PATH = OUT_DIR / "RUN_STATUS.json"

SFREQ = 160.0
EPOCH_SEC = 2.0
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30))
ALPHA = (8, 13)
N_BOOT = 2000
PREREG = {
    "dataset": "PhysioNet eegmmidb 1.0.0",
    "run": "R01 eyes-open baseline",
    "epoch_sec": EPOCH_SEC,
    "sfreq_hz": SFREQ,
    "reference": "common_average",
    "alpha_hilbert_hz": list(ALPHA),
    "plv_pair": ("F4", "P3"),
    "n_features": 10,
    "hypothesis": "dPR(within) - dPR(between) > 0",
}


def _d_pr(cov: np.ndarray) -> float:
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2
    w = np.linalg.eigvalsh(cov)
    w = np.maximum(w, 0.0)
    tr = w.sum()
    if tr <= 1e-12:
        return float("nan")
    return float(tr**2 / (w**2).sum())


def _lz_complexity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8:
        return float("nan")
    med = np.median(x)
    s = "".join("1" if v > med else "0" for v in x)
    i, k, l = 0, 1, 1
    n = len(s)
    c = 1
    while True:
        if i + k > n - 1 or l + k > n - 1:
            c += 1
            break
        if s[i : i + k] == s[l : l + k]:
            k += 1
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    i = 0
                    k = 1
            else:
                l += 1
                if l == n:
                    c += 1
                    break
    return c / (n / math.log2(n) if n > 1 else 1.0)


def _perm_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size - (m - 1) * delay
    if n < 5:
        return float("nan")
    patterns: dict[tuple[int, ...], int] = {}
    for i in range(n):
        key = tuple(np.argsort(x[i : i + m * delay : delay]).tolist())
        patterns[key] = patterns.get(key, 0) + 1
    p = np.array(list(patterns.values()), dtype=float)
    p /= p.sum()
    h = -np.sum(p * np.log2(p + 1e-15))
    h_max = math.log2(math.factorial(m))
    return float(h / h_max) if h_max > 0 else float("nan")


def _bandpower_epoch(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    """Channel-averaged band power (log10 µV²)."""
    nperseg = min(int(sfreq), data.shape[1])
    psds = []
    for ch in data:
        f, pxx = signal.welch(ch, fs=sfreq, nperseg=nperseg)
        mask = (f >= fmin) & (f <= fmax)
        if not mask.any():
            continue
        psds.append(float(np.trapezoid(pxx[mask], f[mask])))
    if not psds:
        return float("nan")
    return float(np.log10(np.mean(psds) + 1e-20))


def _alpha_phases(data: np.ndarray, sfreq: float) -> np.ndarray:
    sos = signal.butter(4, ALPHA, btype="bandpass", fs=sfreq, output="sos")
    filt = signal.sosfiltfilt(sos, data, axis=1)
    return np.angle(signal.hilbert(filt, axis=1))


def _kuramoto_stats(phases: np.ndarray) -> tuple[float, float]:
    r = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return float(np.mean(r)), float(np.std(r))


def _plv_pair(phases: np.ndarray, idx_a: int, idx_b: int) -> float:
    d = phases[idx_a] - phases[idx_b]
    return float(np.abs(np.mean(np.exp(1j * d))))


def _heig_epoch(data: np.ndarray) -> float:
    c = np.cov(data)
    w = np.linalg.eigvalsh(c)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 1e-15:
        return float("nan")
    q = w / s
    ent = -np.sum(q * np.log(q + 1e-15))
    k = data.shape[0]
    return float(ent / math.log(k) if k > 1 else 0.0)


def _features_one_epoch(raw_data: np.ndarray, sfreq: float, ch_names: list[str], idx_f4: int, idx_p3: int) -> np.ndarray:
    feats = [
        _bandpower_epoch(raw_data, sfreq, lo, hi) for lo, hi in BANDS
    ]
    phases = _alpha_phases(raw_data, sfreq)
    r_mean, r_std = _kuramoto_stats(phases)
    feats.extend([r_mean, r_std, _plv_pair(phases, idx_f4, idx_p3)])
    feats.append(_heig_epoch(raw_data))
    avg = np.mean(raw_data, axis=0)
    feats.extend([_lz_complexity(avg), _perm_entropy(avg, m=3, delay=1)])
    return np.array(feats, dtype=float)


def _init_eegbci_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(DATA_DIR)
    try:
        from mne import set_config

        set_config("MNE_DATASETS_EEGBCI_PATH", str(DATA_DIR), set_env=True)
    except Exception:
        pass
    return DATA_DIR


def _status(msg: str, **extra) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"at": datetime.now(timezone.utc).isoformat(), "message": msg, **extra}
    STATUS_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc), flush=True)


def _load_subject_epochs(subject: int, *, epoch_sec: float, eegbci_path: Path) -> tuple[np.ndarray, list[str]] | None:
    cache = CACHE_DIR / f"S{subject:03d}_R01.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        names = z["ch_names"]
        ch = names.tolist() if hasattr(names, "tolist") else list(names)
        return z["feats"], [str(c) for c in ch]

    from mne.io import read_raw_edf
    from mne.datasets import eegbci

    paths = eegbci.load_data(
        subjects=subject,
        runs=[1],
        path=eegbci_path,
        update_path=True,
        verbose=False,
    )
    raw = read_raw_edf(paths[0], preload=True, verbose=False)
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
    raw.pick(picks="eeg", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if abs(raw.info["sfreq"] - SFREQ) > 1:
        raw.resample(SFREQ, verbose=False)
    data = raw.get_data()
    sf = float(raw.info["sfreq"])
    ch_names = raw.ch_names
    names_up = [c.upper().replace(".", "") for c in ch_names]

    def _idx(name: str) -> int:
        t = name.upper()
        for i, c in enumerate(names_up):
            if c == t or c.endswith(t):
                return i
        raise KeyError(name)

    try:
        i_f4, i_p3 = _idx("F4"), _idx("P3")
    except KeyError:
        i_f4, i_p3 = 0, min(1, data.shape[0] - 1)

    n_samp = int(epoch_sec * sf)
    rows: list[np.ndarray] = []
    for start in range(0, data.shape[1] - n_samp + 1, n_samp):
        seg = data[:, start : start + n_samp]
        if np.max(np.abs(seg)) > 200e-6:  # QC: 200 µV
            continue
        rows.append(_features_one_epoch(seg, sf, ch_names, i_f4, i_p3))
    if len(rows) < 2:
        return None
    feats = np.vstack(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=feats, ch_names=np.array(ch_names, dtype=object))
    return feats, ch_names


def _cov_geometries(feats_by_subject: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Equations (3) and (4) with equal unit weight."""
    subjects = sorted(feats_by_subject.keys())
    n = len(subjects)
    p = feats_by_subject[subjects[0]].shape[1]
    means = np.array([feats_by_subject[s].mean(axis=0) for s in subjects])
    grand = means.mean(axis=0)
    between = (means - grand).T @ (means - grand) / max(n - 1, 1)
    within_acc = np.zeros((p, p))
    for s in subjects:
        x = feats_by_subject[s]
        ns = x.shape[0]
        mu = x.mean(axis=0)
        xc = x - mu
        within_acc += xc.T @ xc / max(ns - 1, 1)
    within = within_acc / n
    return between, within


def _scale_pooled(feats_by_subject: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    all_rows = np.vstack([feats_by_subject[s] for s in sorted(feats_by_subject)])
    mu = np.nanmean(all_rows, axis=0)
    sd = np.nanstd(all_rows, axis=0)
    sd[sd < 1e-8] = 1.0
    out = {}
    for s, x in feats_by_subject.items():
        out[s] = (x - mu) / sd
    return out


def _bootstrap_delta(feats_by_subject: dict[int, np.ndarray], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    subjects = sorted(feats_by_subject.keys())
    n = len(subjects)
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=n, replace=True)
        sub = {int(s): feats_by_subject[int(s)] for s in draw}
        b, w = _cov_geometries(sub)
        deltas.append(_d_pr(w) - _d_pr(b))
    arr = np.array(deltas)
    return {
        "n": n_boot,
        "delta_mean": float(np.nanmean(arr)),
        "ci95_low": float(np.nanpercentile(arr, 2.5)),
        "ci95_high": float(np.nanpercentile(arr, 97.5)),
    }


def _null_shuffle_labels(feats_by_subject: dict[int, np.ndarray], n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_rows = []
    labels = []
    for s, x in feats_by_subject.items():
        for row in x:
            all_rows.append(row)
            labels.append(s)
    all_rows = np.array(all_rows)
    labels = np.array(labels)
    n = len(feats_by_subject)
    deltas = []
    for _ in range(n_perm):
        perm = labels.copy()
        rng.shuffle(perm)
        fake: dict[int, list] = {s: [] for s in feats_by_subject}
        for row, lab in zip(all_rows, perm):
            fake[lab].append(row)
        fake = {s: np.vstack(v) for s, v in fake.items() if len(v) >= 2}
        if len(fake) < 2:
            continue
        b, w = _cov_geometries(fake)
        deltas.append(_d_pr(w) - _d_pr(b))
    arr = np.array(deltas)
    return {"n_perm": n_perm, "delta_mean": float(np.nanmean(arr)), "delta_std": float(np.nanstd(arr))}


def _random_projection_stress(feats_by_subject: dict[int, np.ndarray], seed: int, n_proj: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    subjects = sorted(feats_by_subject.keys())
    p = feats_by_subject[subjects[0]].shape[1]
    deltas = []
    for _ in range(n_proj):
        w = rng.standard_normal(p)
        w /= np.linalg.norm(w) + 1e-12
        proj = {s: (x @ w).reshape(-1, 1) for s, x in feats_by_subject.items()}
        b, wi = _cov_geometries(proj)
        deltas.append(_d_pr(wi) - _d_pr(b))
    arr = np.array(deltas)
    return {
        "n_projections": n_proj,
        "delta_mean": float(np.nanmean(arr)),
        "delta_std": float(np.nanstd(arr)),
    }


def _loo_delta(feats_by_subject: dict[int, np.ndarray]) -> dict:
    subjects = sorted(feats_by_subject.keys())
    deltas = []
    for drop in subjects:
        sub = {s: feats_by_subject[s] for s in subjects if s != drop}
        if len(sub) < 3:
            continue
        b, w = _cov_geometries(sub)
        deltas.append(_d_pr(w) - _d_pr(b))
    arr = np.array(deltas)
    return {"n": len(deltas), "delta_mean": float(np.nanmean(arr)), "delta_std": float(np.nanstd(arr))}


def run(*, max_subjects: int, n_boot: int, n_perm: int, seed: int, prefetch: bool) -> dict:
    eegbci_path = _init_eegbci_path()
    _status("started", max_subjects=max_subjects, n_boot=n_boot, n_perm=n_perm, data_path=str(eegbci_path))
    if prefetch:
        from mne.datasets import eegbci

        _status("prefetch_eegbci", subjects=f"1..{max_subjects}")
        eegbci.load_data(
            subjects=list(range(1, max_subjects + 1)),
            runs=[1],
            path=eegbci_path,
            update_path=True,
            verbose=False,
        )
        _status("prefetch_done")
    feats: dict[int, np.ndarray] = {}
    skipped = []
    for sub in range(1, max_subjects + 1):
        try:
            out = _load_subject_epochs(sub, epoch_sec=EPOCH_SEC, eegbci_path=eegbci_path)
        except Exception as e:  # noqa: BLE001
            skipped.append({"subject": sub, "error": str(e)[:200]})
            _status("subject_error", subject=sub, error=str(e)[:120])
            continue
        if out is None:
            skipped.append({"subject": sub, "error": "insufficient_epochs"})
            continue
        feats[sub] = out[0]
        _status("subject_ok", subject=sub, n_epochs=int(feats[sub].shape[0]), loaded=len(feats))
    if len(feats) < 5:
        raise SystemExit(f"Too few subjects loaded ({len(feats)}); need network for eegbci download.")

    feats = _scale_pooled(feats)
    between, within = _cov_geometries(feats)
    d_b, d_w = _d_pr(between), _d_pr(within)
    delta = d_w - d_b

    _status("analysis", n_subjects=len(feats))
    boot = _bootstrap_delta(feats, n_boot, seed)
    _status("bootstrap_done", **boot)
    null = _null_shuffle_labels(feats, n_perm, seed + 1)
    stress = _random_projection_stress(feats, seed + 2, n_proj=48)
    loo = _loo_delta(feats)

    w_b = np.linalg.eigvalsh(between)
    w_w = np.linalg.eigvalsh(within)
    w_b = np.maximum(w_b, 0)[::-1]
    w_w = np.maximum(w_w, 0)[::-1]
    pc12_b = float(w_b[:2].sum() / (w_b.sum() + 1e-15))
    pc12_w = float(w_w[:2].sum() / (w_w.sum() + 1e-15))

    if boot["ci95_low"] > 0:
        verdict = "supports_primary"
    elif boot["ci95_high"] < 0:
        verdict = "supports_reversed"
    else:
        verdict = "unresolved"

    return {
        "experiment": "EEGMMIDB_TRAIT_STATE_v0",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "prereg": PREREG,
        "n_subjects": len(feats),
        "n_epochs_total": int(sum(x.shape[0] for x in feats.values())),
        "skipped": skipped,
        "d_pr_between": d_b,
        "d_pr_within": d_w,
        "delta_d_pr": delta,
        "pc12_variance_fraction_between": pc12_b,
        "pc12_variance_fraction_within": pc12_w,
        "bootstrap_delta": boot,
        "null_shuffle_participant_labels": null,
        "stress_random_1d_projection": stress,
        "leave_one_participant_out": loo,
        "data_path": str(eegbci_path),
        "verdict": verdict,
        "interpretation": {
            "methodologist_note": "Gap depends on 10-D feature engineering; see null + 1D projection.",
            "panel_honest": "Empirical numbers on open data; naming is secondary to this result.",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run EEGMMIDB trait–state dimensionality test")
    p.add_argument("--max-subjects", type=int, default=109, help="Max subject IDs 1..N to load")
    p.add_argument("--bootstrap", type=int, default=N_BOOT)
    p.add_argument("--perm", type=int, default=500, help="Label-shuffle null iterations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="20 subjects, 400 bootstrap")
    p.add_argument("--no-prefetch", action="store_true", help="Skip batch download (use cache)")
    p.add_argument("--full", action="store_true", help="109 subjects, 2000 bootstrap, 1000 perm")
    args = p.parse_args()
    if args.full:
        args.max_subjects = 109
        args.bootstrap = 2000
        args.perm = 1000
    if args.quick:
        args.max_subjects = min(args.max_subjects, 20)
        args.bootstrap = min(args.bootstrap, 400)
        args.perm = min(args.perm, 200)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run(
        max_subjects=args.max_subjects,
        n_boot=args.bootstrap,
        n_perm=args.perm,
        seed=args.seed,
        prefetch=not args.no_prefetch,
    )
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT_DIR / f"RESULT_{tag}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "RESULT_LATEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(path), "verdict": result["verdict"], "delta_d_pr": result["delta_d_pr"]}))


if __name__ == "__main__":
    main()
