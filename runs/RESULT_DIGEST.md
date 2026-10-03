# EEGMMIDB trait–state run — reader digest

Source: `RESULT_LATEST.json` (EEGMMIDB_TRAIT_STATE_v0, finished 2026-10-03 UTC).

---

## For your research friend (results prose)

We preregistered a trait–state comparison on open PhysioNet data: EEG Motor Movement/Imagery **R01** (one-minute eyes-open baseline), **2 s** non-overlapping epochs, **common-average** reference, and a **10-dimensional** feature vector (band powers, alpha-phase Kuramoto summaries, F4–P3 phase locking, within-epoch channel eigenvalue entropy, and two complexity summaries). Participants were units; epochs were repeated observations. We estimated **between-unit** and **within-unit** feature covariances with equal unit weighting (Eq. 3–4) and compared their **participation-ratio effective dimensionality** \(d_{PR}\).

After quality control, **78** of **109** candidate participants contributed **1,322** epochs (remaining IDs had too few usable epochs under the amplitude rule). In this sample, **\(d_{PR}\) was higher within units than between units**: \(\hat d_{PR,\mathrm{within}} \approx 6.01\), \(\hat d_{PR,\mathrm{between}} \approx 3.54\), \(\widehat{\Delta d} \approx 2.47\). A participant-level bootstrap (**2,000** resamples) gave a **95%** interval **[1.92, 3.01]**, which lies entirely above zero under the preregistered decision rule (**supports_primary**). Leave-one-participant-out estimates were stable (mean \(\widehat{\Delta d} \approx 2.47\), SD \(\approx 0.04\)). Permuting epoch labels across participants reduced the mean \(\widehat{\Delta d}\) to about **0.83**, indicating that part of the contrast depends on preserving subject identity—not only on marginal feature structure. Random **one-dimensional** projections of the scaled features yielded \(\widehat{\Delta d} \approx 0\), so the observed gap is tied to the **multivariate feature design**, not to a single scalar projection.

**Plain takeaway:** On this public resting-EEG protocol, **within-person epoch geometry is more high-dimensional than between-person geometry in the chosen feature space**, with uncertainty that excludes zero under our prespecified test. That supports the **ordering** we risked in the preregistration; it does **not** by itself establish a general law of mind or matter, nor does it remove the need for feature ablations and harsher nulls.

---

## Do not overclaim (reviewer box)

1. **This is a preregistered **feature-space** comparison on **one** open dataset (EEGMMIDB R01), not a universal trait–state law or a validation of the name “Cognitive Physics.”**
2. **Large \(\widehat{\Delta d}\) depends on the **10-D engineering** (shuffle and 1D projection stress tests show identity and multivariate structure matter); anatomy and baseline subject effects are expected confounds, not disproof of honest reporting.**
3. **Claim only what passed prespecification: ordering of \(d_{PR}\) with bootstrap CI above zero; **not** latent “traits/states” in a psychological sense until reliability, measurement error, and independent replication are reported.**

---

## One-line caption (figures / slides)

Resting EEG (N=78, 1,322 epochs): \(d_{PR,\mathrm{within}} > d_{PR,\mathrm{between}}\); \(\widehat{\Delta d}=2.47\) [1.92, 3.01] bootstrap; preregistered primary ordering supported in 10-D features.
