# The cost of collective data withdrawal from readmission and mortality prediction models

Code for "The Cost of Collective Data Withdrawal from Readmission and Mortality Prediction Models" (Porter, Krieg, Kirkpatrick, O'Donnell, Verjans; submitted to NEJM AI, 2026).

## Plain-language summary

Hospitals use computer programs to predict which patients are at risk, for example of dying in intensive care or coming back to hospital after going home. These programs learn from past patients' records.

Communities and hospitals can have the right to say "stop using our data". For Indigenous peoples this is part of Indigenous data sovereignty. It is the right to decide how data about their people, lands and resources are used, including the right to say no or to take the data back. It matters because data and blood samples given by Indigenous people have been used for research their communities never agreed to. Who speaks for a community, and how, is for that community to decide.

We asked a simple question. If a group takes its data back, do the predictions get worse for that group's own patients?

Using two large public hospital datasets from the United States, we found they can. When a group of hospitals, or a group of patients who were medically different from everyone else, took its data out, the program became less accurate for those patients than if the same number of random patients had been removed. The more different the group, the bigger the loss.

We then tested a different design. The main program is built without the group's data, and the group's data go into a small separate add-on used only for that group's patients. Predictions with the add-on were about as good as with the usual approach. If the group later wants out, the add-on is deleted and nothing else has to be rebuilt.

The groups in this study were hospitals and groups of patients built from United States records. None of them is an Indigenous nation or community. No Australian data were used, and no Indigenous community took part in or made decisions in this study. Any real use would have to be designed with the communities whose data are involved.

## Data and code

No patient data are included. MIMIC-IV v3.1 and eICU-CRD v2.0 are available to credentialed users through PhysioNet under their data use agreements. The scripts expect the raw PhysioNet csv.gz files under `~/mimic/hosp` and `~/mimic/eicu`, write derived cohorts to `~/mimic/derived` and results to `~/mimic/results`. All analyses were run locally; only aggregate results leave the analysis machine. `aggregate/v9_figures/figure_data_v9.csv` holds every value plotted in Figures 1, 2 and S1, and the other folders in `aggregate/` hold copies of the smaller aggregate result files named in `RESULTS_MANIFEST.md`.

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
| `exp_v14_commonbase.py` (v14.1) | Five groups excluded from one common base on the standard pipeline (superseded by v33, Note 15; its common bases are the multi-group arms of Table S18); cross-fitted adaptation | `results/v14_commonbase_1` |
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
| `exp_v28_multiplicity.py` (v28.1) | Holm and Benjamini-Hochberg adjustment of the 55 main-text contrasts of the v8 manuscript, from the result files (Note 14, Table S19) | `results/v28_multiplicity` |
| `exp_v29_seeds_equivalence.py` | Per-seed estimates, between-seed SD and seed-inclusive intervals for the main contrasts, and post hoc equivalence tests at ±0.005 AUROC, from cached predictions (Note 14, Tables S20 and S21); `SEEDS` and `TAG` select the seed set | `results/v29_seeds_equivalence`, `results/v29_seeds_equivalence_s10` |
| `exp_v30_seed_core.py` (`run_seeds_overnight.sh`) | Pooled model, exact retrain and random-set retrain for seeds 3 to 9 with v5's pipeline, then the v16, v17, v19 and v24 scripts for those seeds (`SEEDS`, `SUMMARY=0`) | `results/v30_seed_core`, checkpoints in `results/v5_partitions/ckpt` |
| `exp_v31_fold_manifests.py` | Provenance of every cross-fitting fold model (v17, v20): both candidate half splits tested against the cached model, the resolved split's excluded patients written to a local manifest, aggregate gaps and checkpoint SHA-256 to `manifest.csv` (Note 5, Table S22) | `results/v31_fold_manifests` |
| `exp_v32_pooled_full_adapt.py` | Full pooled-adaptation comparator: the pooled model with a component fitted on all of the group's training patients, type chosen on a held-back fifth, against the adapted cross-fitted comparator and the confined system (Note 5, Tables S11, S12 and S23); `PART`, `SEEDS` | `results/v32_pooled_full_adapt` |
| `exp_v33_strict_multigroup.py` | Several groups withdrawn at once on the retained-only pipeline: one common base without 5, 10 or 15 most isolated or random groups, each group's component chosen by validation (Note 4, Tables S10 and S14); `SEEDS`, `SIZES`, `FAMILIES` | `results/v33_strict_multigroup` |
| `figures_v5.py` | Figures 1 and 2 and `figure_data_v5.csv` of the October drafts (superseded by `figures_v9.py`) | |
| `exp_v34_isolation.py` | Withdrawal penalty against feature-space isolation for all 49 groups, equal-pool isolation ratio (Figure 1D) | `results/v34_isolation` |
| `exp_v35_landmark.py` | eICU contrasts among stays still in the unit at 24 hours, the primary eICU cohort, from cached predictions | `results/v35_landmark` |
| `exp_v36_others.py` | Other patients' AUROC under each single-group base (Table 3 row A1) | `results/v36_others` |
| `boot_draws.py` | Module used by the scripts above: keeps every bootstrap draw and adds exact percentile p-values | `draws*.npz` beside each script's results |
| `exp_v37_exact_p.py` | Exact bootstrap p-values, Holm and Benjamini-Hochberg over 71 contrasts (Note 14, Table S19); `FAMILY=v9` | `results/v37_exact_p` |
| `export_cluster_assign.py` | Writes the 25 clustered groups as fitted in v5, for v38 | `derived/cluster_assign_v5.csv` |
| `exp_v38_tuned_unlearning.py` | NegGrad+ and Fair Unlearning tuned on retained validation data (Note 3, Tables S7 and S27); `GRID=ext`, `ONLY_GROUPS` | `results/v38_tuned_unlearning_ext` |
| `exp_v38m_matched.py` | Tuned unlearning at exact retraining's level of removal, against exact retraining (post hoc) | `results/v38_tuned_unlearning_ext/matched.json` |
| `exp_v39_eicu_seed_preds.py` | Prediction cache for eICU seeds 3 to 9, for the ten-seed check (Tables S20 and S21) | `results/v39_eicu_seed_core` |
| `exp_v40_boot10k.py` | 10,000-draw bootstrap for the three contrasts whose Holm decision depended on resolution | `results/v40_boot10k` |
| `exp_v41_baseline.py` | Baseline characteristics by dataset and split (Table S25) | `results/v41_baseline` |
| `exp_v42_strict_penalty.py` | Withdrawal penalty with the whole pipeline refitted in both arms (Table S28) | `results/v42_strict_penalty` |
| `exp_v43_isolation_adjusted.py` | Adjusted regressions of the penalty on the equal-pool isolation ratio (Note 16, Table S26) | `results/v43_isolation_adjusted` |
| `exp_v44_participant_flow.py` | Participant flow and participants and outcome events by group and consortium (Tables S29 to S31) | `results/v44_participant_flow` |
| `exp_v45_mixed_queue.py` | Other patients' flags when a group's component is deleted and one top-decile queue is recomputed over all held-out patients (Table S18, Note 13); cached predictions, no training | `results/v45_mixed_queue` |
| `exp_v46_isolation_clusters.py` | Isolation slope within the 25 clustered groups, which do not overlap (Note 16) | `results/v46_isolation_clusters` |
| `figures_v9.py` | Figures 1, 2 and S1 of the submitted manuscript and `figure_data_v9.csv` | `nejmai_build/v9_figures` (copy in `aggregate/v9_figures/`) |
| `run_air_rerun.sh`, `run_three_seed_summaries.sh` | Drivers for the November 2026 rerun of every eICU analysis and the tuned-unlearning grid on the analysis machine, and for the three-seed summaries reported in the Appendix | `results/air_rerun.log`, `results/three_seed_summaries.log` |
| `si_tables_v6.py` | Supplementary Tables S1 to S17 (Table S18 is built from `results/v27_churn/churn.json`, Table S19 by `exp_v28_multiplicity.py`) | |

The residual component in v11.1, v13.1, v14.1 and v14b leaves the shared model's logit effectively unpenalised and lets a logit-only recalibration compete in cross-validation. Earlier versions (v11.0, v13.0, v14.0) penalised the logit and are superseded. Superseded scripts kept for provenance: `exp_v7_eicu_consortium.py` and `exp_v7_eicu_consortium_v71.py` (fixed-epoch eICU training; v7.0 capped one control), `exp_v2_sameheldout.py`, `exp_v3_verify.py`, `exp_v4_fairunlearn.py` and `figure1.py` (earlier analyses on groups fitted to all patients), and `figures_v4.py`, `si_tables_v4.py` and `aggregate/figure_data_v4.csv` (the 18 October figures and tables).

Every experiment writes one JSON per run atomically, skips finished runs on restart, caches model checkpoints, and prints a summary from the files on disk. `QUICK=1` runs a smoke test. `ROOT` sets the data root (default `~/mimic`). Run one copy of a script at a time; two concurrent copies compete for threads and write to the same folders.

`RESULTS_MANIFEST.md` maps each number in the manuscript to the result file it comes from, with checksums of those files.

## Environment

Python 3.10 or later with the package versions pinned in `requirements.txt` (the environment on the analysis machine, an Apple M4 laptop with PyTorch on MPS; CUDA or CPU also work). Run the scripts in the order of the table above; `run_air_rerun.sh` and `run_three_seed_summaries.sh` give the exact commands for the eICU chain and the three-seed summaries. The full set of experiments takes about eight hours. The release that matches the submitted manuscript is tagged `v1.0-submission`.

## Licence

MIT.
