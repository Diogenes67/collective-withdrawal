# Results manifest

Each manuscript number is re-derived from the files below. Checksums are the first 16 hexadecimal characters of SHA-256 of the result file on the analysis machine, recomputed on 27 November 2026 (v45 and v46 added 28 November 2026) after every eICU analysis and the tuned-unlearning analyses were rerun on that machine (18 to 24 November 2026). All result files listed were produced on the analysis machine. Result files are not in this repository; they are aggregate files produced by the scripts named. Copies of the smaller aggregate files are in `aggregate/`.

| Manuscript item | Result file | Script | SHA-256 (16) |
|---|---|---|---|
| Clustered-group and census withdrawal costs; group sizes; isolation ratios; test AUROC; Table S1 | `results/v5_partitions/results.csv` | `exp_v5_partitions.py` | e767fbeb08bc9323 |
| Group-level audits; Table S12 | `results/v6_groupaudit/summary.csv` | `exp_v6_groupaudit.py` | 23bd76376b3b9302 |
| eICU centralised penalty per consortium, fine-tuning, NegGrad+; Table S2 | `results/v7_eicu_consortium_es/summary.csv` | `exp_v7_eicu_consortium_v72.py` | 0123a76e2bf787d9 |
| Gradient-boosted trees and logistic regression | `results/v8_modelclass/paired.json` | `exp_v8_modelclass.py` | 58c6bd93acd3bf35 |
| Membership inference (0.587, 0.499, 0.600); Table S11 | `results/v9_verify_trainfit/summary.csv` | `exp_v9_verify_trainfit.py` | 74753461f3ab28fa |
| NegGrad+ and Fair Unlearning against random removal; Table S10 | `results/v10_fu_trainfit/paired.json` | `exp_v10_fu_trainfit.py` | 5ca6840d3ab4c05c |
| Confinement on MIMIC-IV per group, data-size curve, recalibration, ECE, component selection, checksums; Tables S3, S4; Figure 2A and 2C | `results/v11_confinement_1/results.csv` | `exp_v11_confinement.py` | 08a655865b496cee |
| Federation per consortium, full-test AUROC 0.835; Table S2; Figure 1C | `results/v12_eicu_federated_es/summary.csv` | `exp_v12_eicu_federated_v122.py` | b13434ad78582e58 |
| All dependence-aware AUROC intervals in the abstract, Results and Table 2; Table S7; Figure 1A | `results/v13_intervals_1/intervals.json` | `exp_v13_intervals.py` | a59b086fece597dd |
| Events unflagged per 1,000, sensitivity, net benefit at a matched flag rate; Table 2; Table S5; Figure 1B | `results/v13_intervals_1/matched_flags.json` | `exp_v13b_matched_flags.py` | 0cd3e199494302f6 |
| eICU state-3 contrasts and federation against centralised; Table 2 | `results/v13_intervals_1/extra_contrasts.json` | `exp_v13c_extra_contrasts.py` | 69ca7e9a4115fcc0 |
| Multi-group analysis on the standard pipeline (superseded, Note 15) and cross-fitted analyses; common bases of Table S18 | `results/v14_commonbase_1/paired.json` | `exp_v14_commonbase.py` | 4dd371461c102e47 |
| Pooled-plus-component comparison; Table S8; Figure 2D | `results/v14b_pooled_adapt/paired.json`, `summary.csv` | `exp_v14b_pooled_adapt.py` | 14aeb57e60a1b676, 95dcf9ca831f92c3 |
| Preprocessing sensitivity; Table S9 | `results/v15_preproc/summary.csv` | `exp_v15_preproc.py` | cd5748f40b743d52 |
| Retained-only shared pipeline on MIMIC-IV: strict base, confinement, provenance; Table 2 state 3 (MIMIC-IV); Tables S10, S15 | `results/v16_strict/intervals.json`, `provenance.csv` | `exp_v16_strict.py` | 5e7a780757e24b28, 6bfbc15a9555e809 |
| Cross-fitted residual component for the pooled model (residual-only comparator); Table S13 | `results/v17_crossfit/intervals.json` | `exp_v17_crossfit.py` | b1d1b2ba0c818cec |
| The 24-hour landmark, v8 values (superseded in the main text by v35); Note 8; Table S6 | `results/v18_landmark/all_contrasts.json`, `landmark_contrasts.json` | `exp_v18_landmark.py` | e4b70ace6d1faa99, d775a7c2858d7daf |
| Validation-chosen components on MIMIC-IV, state 2; Table 2; Table 3 row A2; Table S11; Figure 2 | `results/v19_protocol/intervals.json`, `summary.csv` | `exp_v19_protocol.py` | 56b250d95dc13348, ed4044fdbb9e09de |
| The protocol on the centralised eICU model, original pipeline (Table 2 uses the strict pipeline, v23 and v26) | `results/v20_eicu_protocol/intervals.json` | `exp_v20_eicu_protocol.py` | b35cab8051a417e6 |
| Group-aligned sharded training; Table 1 row 6; Note 6; Table S17 | `results/v21_sisa/intervals.json` | `exp_v21_sisa.py` | c4ab853e9619c414 |
| Unlearning methods at a matched flag rate; Table 1 row 3; Table S7 | `results/v22_unlearning_flags/intervals.json` | `exp_v22_unlearning_flags.py` | 11924f06016a0dc7 |
| Retained-only pipeline on the centralised eICU model: strict base, chosen component, provenance; Table 2; Tables S10, S12 | `results/v23_eicu_strict/intervals.json`, `provenance.csv` | `exp_v23_eicu_strict.py` | 22b5f395704289e5, 2b8b9ed1a4b6aeba |
| Adapted pooled comparator with the same choice of component, MIMIC-IV; Table 2; Table S13; Figure 2D | `results/v24_symmetric_mimic/intervals.json` | `exp_v24_symmetric_mimic.py` | a2a97f0f0e01978b |
| Adapted pooled comparator, eICU; Table 2; Table S13 | `results/v25_symmetric_eicu/intervals.json` | `exp_v25_symmetric_eicu.py` | f24c12a62a96e6b8 |
| eICU cells of Table 2 on the strict pipeline | `results/v26_eicu_table2/all.json` | `exp_v26_eicu_table2.py` | 61511bee7c19a9a1 |
| Other patients' predictions after a withdrawal; Table 1, other patients' column; Note 13; Table S18 | `results/v27_churn/churn.json` (aggregates in `aggregate/v27_churn/`) | `exp_v27_churn.py` | fb8bafec45ebc293 |
| Multiplicity adjustment of the 55 main-text contrasts; Note 14; Table S19 | `results/v28_multiplicity/multiplicity.csv` (copy in `aggregate/v28_multiplicity/`) | `exp_v28_multiplicity.py` | a49ab3efdc0a3dd5 |
| Training randomness, three seeds (Table S20) and equivalence tests (Table S21); Note 14 | `results/v29_seeds_equivalence/seeds.csv`, `equivalence.csv` | `exp_v29_seeds_equivalence.py` | 89661c9ed6662a5a, 82ab1f7ab4eda33a |
| Training randomness, ten seeds on MIMIC-IV (Table S20) and equivalence tests (Table S21); Note 14 | `results/v29_seeds_equivalence_s10/seeds.csv`, `equivalence.csv` | `exp_v30_seed_core.py`, `run_seeds_overnight.sh`, `exp_v29_seeds_equivalence.py` | ff4fd4088b25cf3a, aeb557a1a5308f2f |
| Fold-model provenance, 500 MIMIC-IV and 60 eICU fold models; Note 5; Table S22 | `results/v31_fold_manifests/manifest.csv` | `exp_v31_fold_manifests.py` | 84359e20c73e198a |
| Full pooled-adaptation comparator; Note 5; Tables S11, S12 and S23 | `results/v32_pooled_full_adapt/summary_mimic.csv`, `summary_eicu.csv`, `intervals.json` | `exp_v32_pooled_full_adapt.py` | e40e594fc1494be1, 81d5d1885ae7cd72, d5824253be4f9546 |
| Several groups withdrawn at once, retained-only pipeline; Note 4; Tables S10 and S14 | `results/v33_strict_multigroup/summary.csv`, `intervals.json` | `exp_v33_strict_multigroup.py` | 15b6328e13c44333, b8aec30453c23555 |
| Isolation regression over 49 groups (slope, Spearman, family intercepts, equal-pool ratios); Figure 1D | `results/v34_isolation/regression.json`, `isolation.csv`, `figure_data.csv` | `exp_v34_isolation.py` | 88c590c125cf79ba, 60e004a318e5f8de, a63dffc0f8a914ed |
| eICU 24-hour landmark, primary eICU cohort: penalty, confinement contrasts, lower in N of 10, deaths unflagged; Table 1 and Table 2 eICU rows; Figure 1C | `results/v35_landmark/contrasts_landmark.json`, `per_consortium.csv`, `table2_eicu_landmark.md` | `exp_v35_landmark.py` | 8c811939ea2ea1a2, 49c1d554486a4bd2, 2e8f1ac8feb00369 |
| eICU all held-out stays, sensitivity cohort | `results/v35_landmark/contrasts_all.json` | `exp_v35_landmark.py` | 30c42c6c120835e7 |
| Other patients' AUROC under each single-group base; Table 3 row A1 | `results/v36_others/intervals.json`, `per_group.csv` | `exp_v36_others.py`, `boot_draws.py` | 9e9386f7dc7b3a53, b57fd975043f33b5 |
| Exact bootstrap p-values, Holm and Benjamini-Hochberg over 71 contrasts; Note 14; Table S19 | `results/v37_exact_p/multiplicity.csv`, `table_S19.md` | `exp_v37_exact_p.py` (FAMILY=v9; p from v40 where no draw of 1,000 crossed zero and the Holm decision depended on it) | a0df5e0fff345edf, 816e3aec6568edef |
| 10,000-draw bootstrap for the three contrasts whose Holm decision depended on resolution (v19 readmissions unflagged, confined minus pooled; v21 confined minus sharded ensemble; v33 fifteen random groups); reproduces the 1,000-draw intervals; Note 14 | `results/v40_boot10k/v19.json`, `v21.json`, `v33.json` | `exp_v40_boot10k.py` | 0d1714554753b200, e9190c7764d5440a, 06f4e5d531f36293 |
| Withdrawal penalty under complete-pipeline retraining of both arms, the primary penalty (abstract 0.007 and 0.010; Results; Table 1 row 2; Table 3); Note 1; Table S28 | `results/v42_strict_penalty/intervals.json` | `exp_v42_strict_penalty.py` | c97f17981b074e31 |
| Adjusted isolation regressions (48 matched groups; family, dataset, size and withdrawn-fraction terms); Note 16; Table S26 | `results/v43_isolation_adjusted/regressions.json` | `exp_v43_isolation_adjusted.py` | ab07e4bde7b5fc1a |
| Tuned unlearning on the extended grid (forget weight down to 0, learning rate down to 1e-5, 1 epoch); main text Results; Table 1 row 3; Note 3; Tables S7 and S27 | `results/v38_tuned_unlearning_ext/intervals.json`, `frontier.csv`, `summary.csv` | `exp_v38_tuned_unlearning.py` (GRID=ext) | 15492540f7cf656d, 0253d24536946110, 6ca7b86e1a10689e |
| Removal-matched comparison with exact retraining (post hoc); Results; Note 3 | `results/v38_tuned_unlearning_ext/matched.json` | `exp_v38m_matched.py` | e6bc3453fb087178 |
| Training randomness, ten seeds on eICU (Table S20) and equivalence tests (Table S21); Note 14 | `results/v29_seeds_equivalence_eicu_s10/seeds.csv`, `equivalence.csv` | `exp_v7_eicu_consortium_v72.py` (SEEDS=3,...,9), `exp_v39_eicu_seed_preds.py`, `exp_v29_seeds_equivalence.py` (ESEEDS=0,...,9) | 58fa83526a7e9832, a5305b4b7097be7c |
| Baseline characteristics by dataset and split; Table S25 | `results/v41_baseline/table_baseline.csv` | `exp_v41_baseline.py` | 51ce7586903121a1 |
| Participant flow, participants and outcome events by group and consortium; Note 1; Tables S29 to S31 | `results/v44_participant_flow/flow.json`, `groups_mimic.csv`, `groups_eicu.csv` | `exp_v44_participant_flow.py` | 3fe890593aa5648e, f9b6701c4751366a, 49e3745ee413a6fd |
| Figures 1, 2 and S1; every plotted value | `aggregate/v9_figures/figure_data_v9.csv` (figures in `aggregate/v9_figures/`) | `figures_v9.py` | 2b83ce87ebd38bc7 |
| Other patients' flags after component deletion with one shared top-decile queue (0.76 and 3.02 per 1,000); Table 1 row 7; Table S18; Note 13 | `results/v45_mixed_queue/summary.txt`, `per_group.csv` | `exp_v45_mixed_queue.py` | 833b2dce1748e9e6, bcaf533a98f3e9ef |
| Isolation slope within the 25 clustered groups (0.019; 0.011 to 0.026); abstract; Results; Note 16 | `results/v46_isolation_clusters/summary.json` | `exp_v46_isolation_clusters.py` | 19f20a2ce51fe96d |
