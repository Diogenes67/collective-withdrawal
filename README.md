# The cost of collective data withdrawal in clinical AI

Code for "The Cost of Collective Data Withdrawal in Clinical AI" (Porter, Krieg, Kirkpatrick, Verjans; submitted to NEJM AI, 2026).

No patient data are included. MIMIC-IV v3.1 and eICU-CRD v2.0 are available to credentialed users through PhysioNet under their data use agreements. The scripts expect the raw PhysioNet csv.gz files under `~/mimic/hosp` and `~/mimic/eicu`, write derived cohorts to `~/mimic/derived` and results to `~/mimic/results`. All analyses were run locally; only aggregate results leave the analysis machine. `aggregate/figure_data_v5.csv` holds every value plotted in Figures 1 and 2, and `aggregate/v27_churn/` the aggregate results behind Table S18.

## Scripts behind the manuscript, in the order they were run

| Script | What it does | Output folder |
|---|---|---|
| `extract_cohort.py` | MIMIC-IV 30-day readmission cohort, 122 features | `derived/cohort_latest.parquet` |
| `exp_eicu.py` | Builds the eICU in-hospital mortality cohort (and single-hospital withdrawal, not reported) | `derived/eicu_cohort.parquet` |
| `exp_v5_partitions.py` | Withdrawal of 25 clustered groups and race, language, insurance and anchor-year partitions, with patient- and admission-matched random controls; isolation ratio | `results/v5_partitions` |
| `exp_v6_groupaudit.py` | Group-level audits A and B (Appendix) | `results/v6_groupaudit` |
| `exp_v7_eicu_consortium_v72.py` | eICU consortium withdrawal with early stopping and uncapped stay-matched controls | `results/v7_eicu_consortium_es` |
| `exp_v8_modelclass.py` | Exact-retraining contrast for gradient-boosted trees and logistic regression | `results/v8_modelclass` |
| `exp_v9_verify_trainfit.py` | Loss-based and reference-model (LiRA) membership inference | `results/v9_verify_trainfit` |
| `exp_v10_fu_trainfit.py` | Fine-tuning, NegGrad+ and Fair Unlearning on the clustered groups | `results/v10_fu_trainfit` |
| `exp_v11_confinement.py` (v11.1) | Contribution confinement on MIMIC-IV: control component, recalibration, residual component, representation head, fine-tuned copy, data-size curve | `results/v11_confinement_1` |
| `exp_v12_eicu_federated_v122.py` | FedAvg federation over 206 eICU hospitals, federated withdrawal and confinement inside the federation | `results/v12_eicu_federated_es` |
| `exp_v13_intervals.py` (v13.1) | Dependence-aware bootstrap intervals (hospital-cluster on eICU; group then patient on MIMIC-IV); run `MEMBERSHIP_ONLY=1` first | `results/v13_intervals_1` |
| `exp_v13b_matched_flags.py` | Decision metrics at a matched flag rate | `results/v13_intervals_1` |
| `exp_v13c_extra_contrasts.py` | Additional eICU contrasts for the permission-state table, from cached predictions (`CHUNK=0..4`, then combine) | `results/v13_intervals_1` |
| `exp_v14_commonbase.py` (v14.1) | Five groups excluded from one common base; cross-fitted adaptation | `results/v14_commonbase_1` |
| `exp_v14b_pooled_adapt.py` | The same component fitted to the pooled model and to the base | `results/v14b_pooled_adapt` |
| `exp_v15_preproc.py` | Preprocessing refitted on retained rows for the five most isolated groups | `results/v15_preproc` |
| `exp_v16_strict.py` | Contribution confinement with the retained-only shared pipeline on MIMIC-IV (vocabulary, imputation, scaling and base refitted without the group), with a provenance record per run | `results/v16_strict` |
| `exp_v17_crossfit.py` | Cross-fitted residual component for the pooled model (the residual-only adapted pooled comparator) | `results/v17_crossfit` |
| `exp_v18_landmark.py` | eICU contrasts restricted to stays still in the unit at 24 hours | `results/v18_landmark` |
| `exp_v19_protocol.py` | Validation-chosen component (residual, head or fine-tuned copy) on the strict MIMIC-IV base: the protocol | `results/v19_protocol` |
| `exp_v20_eicu_protocol.py` (`run_v20.sh`) | The protocol on the centralised eICU model, original pipeline | `results/v20_eicu_protocol` |
| `exp_v21_sisa.py` (`run_v21.sh`) | Group-aligned sharded training on MIMIC-IV | `results/v21_sisa` |
| `exp_v22_unlearning_flags.py` (`run_v22_v23.sh`) | Unlearning methods against the same method on random removal, at a matched flag rate | `results/v22_unlearning_flags` |
| `exp_v23_eicu_strict.py` (`run_v22_v23.sh`) | The protocol on the centralised eICU model with the retained-only pipeline, with provenance | `results/v23_eicu_strict` |
| `exp_v24_symmetric_mimic.py`, `exp_v25_symmetric_eicu.py` (`run_v24_v25.sh`) | Adapted pooled comparator given the same choice of component, cross-fitted | `results/v24_symmetric_mimic`, `results/v25_symmetric_eicu` |
| `exp_v26_eicu_table2.py` | eICU cells of Table 2 on the strict pipeline | `results/v26_eicu_table2` |
| `exp_v27_churn.py` | Other patients' predictions after a withdrawal, from cached checkpoints (no training) | `results/v27_churn` |
| `figures_v5.py` | Figures 1 and 2 and `figure_data_v5.csv` | |
| `si_tables_v6.py` | Supplementary Tables S1 to S17 (Table S18 is built from `results/v27_churn/churn.json`) | |

The residual component in v11.1, v13.1, v14.1 and v14b leaves the shared model's logit effectively unpenalised and lets a logit-only recalibration compete in cross-validation. Earlier versions (v11.0, v13.0, v14.0) penalised the logit and are superseded. Superseded scripts kept for provenance: `exp_v7_eicu_consortium.py` and `exp_v7_eicu_consortium_v71.py` (fixed-epoch eICU training; v7.0 capped one control), `exp_v2_sameheldout.py`, `exp_v3_verify.py`, `exp_v4_fairunlearn.py` and `figure1.py` (earlier analyses on groups fitted to all patients), and `figures_v4.py`, `si_tables_v4.py` and `aggregate/figure_data_v4.csv` (the 18 October figures and tables).

Every experiment writes one JSON per run atomically, skips finished runs on restart, caches model checkpoints, and prints a summary from the files on disk. `QUICK=1` runs a smoke test. `ROOT` sets the data root (default `~/mimic`). Run one copy of a script at a time; two concurrent copies compete for threads and write to the same folders.

`RESULTS_MANIFEST.md` maps each number in the manuscript to the result file it comes from, with checksums of those files.

## Environment

Python 3.10 or later, PyTorch 2.x (Apple MPS, CUDA or CPU), pandas, pyarrow, scikit-learn, matplotlib. The full set of experiments takes about eight hours on an Apple M4 laptop.

## Licence

MIT.
