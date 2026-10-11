# Collective withdrawal of training data in clinical AI

Code for "Collective withdrawal of training data degrades clinical AI performance for the withdrawing group" (Porter, Krieg, Kirkpatrick, Verjans; submitted to NEJM AI, 2026).

No patient data are included. MIMIC-IV v3.1 and eICU-CRD v2.0 are available to credentialed users through PhysioNet under their data use agreements. The scripts expect the raw PhysioNet csv.gz files under `~/mimic/hosp` and `~/mimic/eicu`, and write derived cohorts to `~/mimic/derived` and results to `~/mimic/results`. Only aggregate results (per-group summaries) leave the analysis machine.

## Scripts, in the order they were run

| Script | What it does | Output folder |
|---|---|---|
| `extract_cohort.py` | MIMIC-IV 30-day readmission cohort, 122 features | `derived/cohort_latest.parquet` |
| `exp_v5_partitions.py` | Withdrawal of real partitions (race, language, insurance, anchor year) and of 25 pseudo-communities refitted on training patients, with patient-matched and admission-matched random controls, four remedies, clinical metrics and feature-space isolation (`COMMUNITIES=1`) | `results/v5_partitions` |
| `exp_v6_groupaudit.py` | Group-level verification: reference audit against ten exact retrains and a query-only permutation audit; noise floor | `results/v6_groupaudit` |
| `exp_v7_eicu_consortium.py` | eICU consortium withdrawal (regions, teaching status, bed-size bands) with stay-matched controls | `results/v7_eicu_consortium` |
| `exp_v8_modelclass.py` | Exact retraining contrast with gradient-boosted trees and logistic regression | `results/v8_modelclass` |
| `exp_v9_verify_trainfit.py` | Loss-based membership inference and offline LiRA (16 shadows, over-trained control) on the refitted communities | `results/v9_verify_trainfit` |
| `exp_v10_fu_trainfit.py` | Fair Unlearning (Chen et al. 2026) on the refitted communities, FU-race and FU-community | `results/v10_fu_trainfit` |
| `exp_eicu.py` | Single-hospital withdrawal on eICU (64 hospitals), builds the eICU cohort | `results/eicu_*` |
| `exp_v2_sameheldout.py`, `exp_v3_verify.py`, `exp_v4_fairunlearn.py` | Earlier runs on pseudo-communities fitted on all patients; kept for provenance | `results/sameheldout_*`, `verify_*`, `fu_*` |
| `figure1.py` | Figure 1 | |

Every experiment writes one JSON per run atomically, skips finished runs on restart, caches model checkpoints, and prints a summary table from the files on disk. `QUICK=1` runs a two-group smoke test. `ROOT` sets the data root (default `~/mimic`).

## Environment

Python 3.10 or later, PyTorch 2.x (Apple MPS or CUDA or CPU), pandas, pyarrow, scikit-learn, scipy, matplotlib. The full set of experiments takes about three hours on an Apple M4 laptop.

## Licence

MIT.
