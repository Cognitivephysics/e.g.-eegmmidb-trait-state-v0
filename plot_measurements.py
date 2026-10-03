#!/usr/bin/env python3
"""Figures from cached features + RESULT_LATEST (no re-download)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as exp  # noqa: E402

CACHE_DIR = exp.CACHE_DIR
RESULT = HERE / "runs" / "RESULT_LATEST.json"
OUT = HERE / "runs" / "figures"

FEATURE_LABELS = [
    "δ 1–4 Hz",
    "θ 4–8 Hz",
    "α 8–13 Hz",
    "β 13–30 Hz",
    "Kuramoto R̄",
    "Kuramoto σ",
    "PLV F4–P3",
    "H_eig",
    "LZ",
    "H_perm",
]


def _load_feats() -> dict[int, np.ndarray]:
    feats: dict[int, np.ndarray] = {}
    for p in sorted(CACHE_DIR.glob("S*_R01.npz")):
        sid = int(p.stem.split("_")[0][1:])
        z = np.load(p, allow_pickle=True)
        feats[sid] = z["feats"]
    if len(feats) < 5:
        raise SystemExit(f"Need feature cache in {CACHE_DIR} — run ./joel eeg-trait-state --full first")
    return feats


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_dpr_bar(result: dict, d_b: float, d_w: float) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Between-unit\n(participant means)", "Within-unit\n(epoch variation)"]
    vals = [d_b, d_w]
    colors = ["#2c5282", "#c05621"]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylabel(r"Effective dimensionality $d_{PR}$")
    ax.set_title("EEGMMIDB R01 — participation ratio")
    delta = result.get("delta_d_pr", vals[1] - vals[0])
    boot = result.get("bootstrap_delta") or {}
    ax.annotate(
        rf"$\Delta d_{{PR}}$ = {delta:.2f}" + "\n"
        rf"95% CI [{boot.get('ci95_low', 0):.2f}, {boot.get('ci95_high', 0):.2f}]",
        xy=(0.5, max(vals) * 0.55),
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = OUT / "fig1_dpr_between_vs_within.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eigen_spectra(between: np.ndarray, within: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    eb = np.maximum(np.linalg.eigvalsh(between), 0)[::-1]
    ew = np.maximum(np.linalg.eigvalsh(within), 0)[::-1]
    x = np.arange(1, len(eb) + 1)
    ax.plot(x, eb / eb.sum(), "o-", label="Between-unit", color="#2c5282", linewidth=2)
    ax.plot(x, ew / ew.sum(), "s-", label="Within-unit", color="#c05621", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("Component index (eigenvalue rank)")
    ax.set_ylabel("Fraction of total variance")
    ax.set_title("Covariance spectra (normalized eigenvalues)")
    ax.legend()
    fig.tight_layout()
    path = OUT / "fig2_covariance_eigen_spectra.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bootstrap(feats: dict[int, np.ndarray], result: dict, n_boot: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    subjects = sorted(feats.keys())
    n = len(subjects)
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=n, replace=True)
        sub = {int(s): feats[int(s)] for s in draw}
        b, w = exp._cov_geometries(sub)
        deltas.append(exp._d_pr(w) - exp._d_pr(b))
    arr = np.array(deltas)
    boot = result.get("bootstrap_delta") or {}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr, bins=40, color="#4a5568", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Δ = 0")
    if boot.get("ci95_low") is not None:
        ax.axvline(boot["ci95_low"], color="#c05621", linestyle=":", linewidth=2)
        ax.axvline(boot["ci95_high"], color="#c05621", linestyle=":", linewidth=2, label="95% CI (full run)")
    ax.set_xlabel(r"Bootstrap $\Delta d_{PR}$ (within − between)")
    ax.set_ylabel("Count")
    ax.set_title(f"Participant bootstrap (n={n_boot} for figure; full run used 2000)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig3_bootstrap_delta_dpr.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _pick_subject(default: int, feats: dict[int, np.ndarray]) -> int:
    if default in feats:
        return default
    return max(feats.keys(), key=lambda s: feats[s].shape[0])


def _load_raw_r01(subject: int):
    from mne.channels import make_standard_montage
    from mne.datasets import eegbci
    from mne.datasets.eegbci import standardize
    from mne.io import read_raw_edf

    eegbci_path = exp._init_eegbci_path()
    paths = eegbci.load_data(
        subjects=subject,
        runs=[1],
        path=eegbci_path,
        update_path=True,
        verbose=False,
    )
    raw = read_raw_edf(paths[0], preload=True, verbose=False)
    raw.pick(picks="eeg", verbose=False)
    standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if abs(raw.info["sfreq"] - exp.SFREQ) > 1:
        raw.resample(exp.SFREQ, verbose=False)
    return raw


def _first_clean_epoch(raw) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (data ch×time µV, times s, sfreq, start_sample)."""
    data = raw.get_data()
    sf = float(raw.info["sfreq"])
    n_samp = int(exp.EPOCH_SEC * sf)
    for start in range(0, data.shape[1] - n_samp + 1, n_samp):
        seg = data[:, start : start + n_samp]
        if np.max(np.abs(seg)) <= 200e-6:
            times = np.arange(n_samp) / sf
            return seg * 1e6, times, sf, start
    seg = data[:, :n_samp]
    times = np.arange(n_samp) / sf
    return seg * 1e6, times, sf, 0


def plot_raw_waveform_and_topomap(subject: int) -> Path:
    from mne.viz import plot_topomap
    from scipy import signal

    raw = _load_raw_r01(subject)
    seg_uv, times, sf, start = _first_clean_epoch(raw)
    ch_names = raw.ch_names
    # Alpha band power per channel (8–13 Hz) for topomap
    sos = signal.butter(4, exp.ALPHA, btype="bandpass", fs=sf, output="sos")
    alpha_pow = []
    for ch in seg_uv:
        f, pxx = signal.welch(signal.sosfiltfilt(sos, ch), fs=sf, nperseg=min(len(ch), int(sf)))
        mask = (f >= exp.ALPHA[0]) & (f <= exp.ALPHA[1])
        alpha_pow.append(float(np.trapezoid(pxx[mask], f[mask])))
    alpha_pow = np.array(alpha_pow)

    pick_names = [n for n in ("Fz", "Cz", "Pz", "Oz", "F3", "F4", "C3", "C4", "P3", "P4") if n in ch_names]
    if len(pick_names) < 4:
        pick_names = ch_names[: min(10, len(ch_names))]
    idx = [ch_names.index(n) for n in pick_names]

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.35)
    ax_w = fig.add_subplot(gs[0])
    offset = 0.0
    step = np.max(np.ptp(seg_uv[idx], axis=1)) * 1.15 + 5
    for i, ci in enumerate(idx):
        ax_w.plot(times, seg_uv[ci] + offset, color="#2d3748", linewidth=0.7)
        ax_w.text(-0.02, offset, pick_names[i], transform=ax_w.get_yaxis_transform(), ha="right", va="center", fontsize=8)
        offset += step
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Channels (stacked, µV + offset)")
    ax_w.set_title(f"Example 2 s epoch — S{subject:03d} R01 eyes-open (sample {start}, CAR)")
    ax_w.set_xlim(times[0], times[-1])

    ax_t = fig.add_subplot(gs[1])
    plot_topomap(alpha_pow, raw.info, axes=ax_t, show=False, contours=4)
    ax_t.set_title("α band power (8–13 Hz) — same epoch")
    fig.tight_layout()
    path = OUT / f"fig4_raw_epoch_S{subject:03d}_waveform_topomap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary_panel(result: dict, d_b: float, d_w: float, between: np.ndarray, within: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Between", "Within"], [d_b, d_w], color=["#2c5282", "#c05621"])
    axes[0].set_ylabel(r"$d_{PR}$")
    axes[0].set_title("Effective dimensionality")
    eb = np.maximum(np.linalg.eigvalsh(between), 0)[::-1]
    ew = np.maximum(np.linalg.eigvalsh(within), 0)[::-1]
    axes[1].plot(eb / eb.sum(), "o-", label="Between")
    axes[1].plot(ew / ew.sum(), "s-", label="Within")
    axes[1].set_title("Variance fractions")
    axes[1].set_xlabel("Component")
    axes[1].legend()
    n = result.get("n_subjects", "?")
    ep = result.get("n_epochs_total", "?")
    fig.suptitle(f"EEGMMIDB trait–state v0 — N={n} participants, {ep} epochs", fontsize=11, y=1.02)
    fig.tight_layout()
    path = OUT / "fig0_summary_panel.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot EEG measurement figures from cache")
    p.add_argument("--bootstrap", type=int, default=400, help="Bootstrap draws for histogram figure")
    p.add_argument("--open", action="store_true", help="Open figures folder in Finder")
    p.add_argument("--raw", action="store_true", help="Add waveform + α topomap from one EDF")
    p.add_argument("--subject", type=int, default=1, help="Subject ID for --raw (default 1)")
    args = p.parse_args()
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT.read_text()) if RESULT.is_file() else {}
    feats = _load_feats()
    feats = exp._scale_pooled(feats)
    between, within = exp._cov_geometries(feats)
    d_b = exp._d_pr(between)
    d_w = exp._d_pr(within)
    paths = [
        plot_summary_panel(result, d_b, d_w, between, within),
        plot_dpr_bar(result, d_b, d_w),
        plot_eigen_spectra(between, within),
        plot_bootstrap(feats, result, args.bootstrap, seed=42),
    ]
    if args.raw:
        sub = _pick_subject(args.subject, feats)
        try:
            paths.append(plot_raw_waveform_and_topomap(sub))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"raw_plot_error": str(exc)[:300]}), file=sys.stderr)
    manifest = {
        "figures": [str(x) for x in paths],
        "result_json": str(RESULT),
        "n_subjects_cache": len(feats),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if args.open:
        import subprocess

        subprocess.run(["open", str(OUT)], check=False)


if __name__ == "__main__":
    main()
