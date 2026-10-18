# Results manifest

Each manuscript number is re-derived from the files below. Checksums are the first 16 hexadecimal characters of SHA-256 of the result file on the analysis machine on 18 October 2026. Result files are not in this repository; they are aggregate files produced by the scripts named.

| Manuscript item | Result file | Script | SHA-256 (16) |
|---|---|---|---|
| Clustered-group and census withdrawal costs; group sizes; isolation ratios; test AUROC; Table S1 | `results/v5_partitions/results.csv` | `exp_v5_partitions.py` | e767fbeb08bc9323 |
| Group-level audits; Table S12 | `results/v6_groupaudit/summary.csv` | `exp_v6_groupaudit.py` | 23bd76376b3b9302 |
| eICU centralised penalty per consortium, fine-tuning, NegGrad+; Table S2 | `results/v7_eicu_consortium_es/summary.csv` | `exp_v7_eicu_consortium_v72.py` | e19014eb8e6b7405 |
| Gradient-boosted trees and logistic regression | `results/v8_modelclass/paired.json` | `exp_v8_modelclass.py` | 01b86384608a4121 |
| Membership inference (0.587, 0.499, 0.600); Table S11 | `results/v9_verify_trainfit/summary.csv` | `exp_v9_verify_trainfit.py` | 74753461f3ab28fa |
| NegGrad+ and Fair Unlearning against random removal; Table S10 | `results/v10_fu_trainfit/paired.json` | `exp_v10_fu_trainfit.py` | 5ca6840d3ab4c05c |
| Confinement on MIMIC-IV per group, data-size curve, recalibration, ECE, component selection, checksums; Tables S3, S4; Figure 2A and 2C | `results/v11_confinement_1/results.csv` | `exp_v11_confinement.py` | 08a655865b496cee |
| Federation per consortium, full-test AUROC 0.835; Table S2; Figure 1C | `results/v12_eicu_federated_es/summary.csv` | `exp_v12_eicu_federated_v122.py` | b13434ad78582e58 |
| All dependence-aware AUROC intervals in the abstract, Results and Table 2; Table S7; Figure 1A | `results/v13_intervals_1/intervals.json` | `exp_v13_intervals.py` | 974bcba21aa54d81 |
| Events unflagged per 1,000, sensitivity, net benefit at a matched flag rate; Table 2; Table S5; Figure 1B | `results/v13_intervals_1/matched_flags.json` | `exp_v13b_matched_flags.py` | d320683731582ac2 |
| eICU state-3 contrasts and federation against centralised; Table 2 | `results/v13_intervals_1/extra_contrasts.json` | `exp_v13c_extra_contrasts.py` | 69ca7e9a4115fcc0 |
| Multi-group and cross-fitted analyses; Table S8 | `results/v14_commonbase_1/paired.json` | `exp_v14_commonbase.py` | 4dd371461c102e47 |
| Pooled-plus-component comparison; Table S8; Figure 2D | `results/v14b_pooled_adapt/paired.json`, `summary.csv` | `exp_v14b_pooled_adapt.py` | 14aeb57e60a1b676, 95dcf9ca831f92c3 |
| Preprocessing sensitivity; Table S9 | `results/v15_preproc/summary.csv` | `exp_v15_preproc.py` | cd5748f40b743d52 |
