**Table S19.** Multiplicity adjustment for the contrasts reported in the main text. Two-sided p-values are exact percentile p-values from the stored bootstrap draws (41 of 71 contrasts); where no draw crossed zero (30 contrasts) the p-value is below the resolution of 1,000 draws and the normal approximation to the interval is given, adjusted over all 71 contrasts by Holm's method (family-wise error rate) and by the Benjamini-Hochberg procedure (false discovery rate). Unadjusted intervals are those reported in the text and tables.

| Contrast | Estimate (95% interval) | p | Holm-adjusted p | Benjamini-Hochberg q |
|---|---|---|---|---|
| **Withdrawal cost, eICU (24-hour landmark, primary)** | | | | |
| Consortium withdrawal minus matched random removal, 10 consortia | -0.0049 (-0.0064 to -0.0032) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Consortium withdrawal minus pooled model | -0.0074 (-0.0090 to -0.0058) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Deaths unflagged at a matched flag rate, withdrawal minus random removal | +0.69 (+0.18 to +1.23) | 0.010 | 0.330 | 0.018 |
| Deaths unflagged at a matched flag rate, withdrawal minus pooled model | +1.26 (+0.63 to +1.79) | <0.0001 (approx.) | 0.001 | <0.0001 |
| **Remedies (24-hour landmark, primary)** | | | | |
| Federation, consortium withdrawal minus random removal | -0.0008 (-0.0023 to +0.0011) | 0.454 | 1.000 | 0.597 |
| Federation without the consortium minus full federation | -0.0054 (-0.0069 to -0.0037) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Federation, deaths unflagged, without the consortium minus full federation | +0.62 (+0.16 to +1.00) | 0.008 | 0.280 | 0.015 |
| **Confinement, eICU (24-hour landmark, primary)** | | | | |
| Federation: consortium component minus full federation | +0.0208 (+0.0165 to +0.0261) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Federation: consortium-specific gain (component minus control component) | +0.0099 (+0.0068 to +0.0132) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Federation: deaths unflagged, component minus full federation | -2.94 (-3.89 to -1.81) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Centralised model: base alone (retained-only pipeline) minus pooled model | -0.0092 (-0.0112 to -0.0066) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Centralised model: confined minus pooled model | -0.0004 (-0.0018 to +0.0008) | 0.510 | 1.000 | 0.635 |
| Centralised model: confined minus adapted pooled comparator | -0.0017 (-0.0025 to -0.0008) | <0.0001 (approx.) | 0.005 | 0.0003 |
| Centralised model: deaths unflagged, confined minus pooled model | -0.17 (-0.57 to +0.60) | 0.866 | 1.000 | 0.897 |
| Centralised model: deaths unflagged, confined minus adapted pooled comparator | +0.05 (-0.27 to +0.58) | 0.510 | 1.000 | 0.635 |
| Centralised model: deaths unflagged, base minus pooled model | +1.87 (+0.67 to +2.64) | 0.0002 (approx.) | 0.009 | 0.0005 |
| **Withdrawal cost, MIMIC-IV** | | | | |
| Group withdrawal minus matched random removal, 25 clustered groups | -0.0106 (-0.0157 to -0.0061) | <0.0001 (approx.) | 0.0007 | <0.0001 |
| Group withdrawal minus pooled model | -0.0083 (-0.0132 to -0.0039) | 0.0004 (approx.) | 0.018 | 0.001 |
| Group withdrawal minus random removal, readmissions unflagged at a matched flag rate | +1.06 (+0.12 to +2.21) | 0.022 | 0.660 | 0.037 |
| Race groups, withdrawal minus random removal (bootstrap over 7 groups) | -0.0014 (-0.0087 to +0.0045) | 0.774 | 1.000 | 0.837 |
| Race groups other than white, withdrawal minus random removal (bootstrap over 6 groups) | +0.0019 (-0.0012 to +0.0059) | 0.307 | 1.000 | 0.431 |
| Gradient-boosted trees, withdrawal minus random removal (bootstrap over groups) | -0.0081 (-0.0131 to -0.0043) | 0.0003 (approx.) | 0.014 | 0.0008 |
| Logistic regression, withdrawal minus random removal (bootstrap over groups) | -0.0024 (-0.0040 to -0.0008) | 0.006 | 0.222 | 0.012 |
| **Withdrawal cost, eICU (all stays, sensitivity)** | | | | |
| Consortium withdrawal minus matched random removal, 10 consortia | -0.0066 (-0.0084 to -0.0045) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Consortium withdrawal minus pooled model | -0.0096 (-0.0114 to -0.0074) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Deaths unflagged at a matched flag rate, withdrawal minus pooled model | +1.76 (+1.09 to +2.29) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Deaths unflagged at a matched flag rate, withdrawal minus random removal | +1.07 (+0.56 to +1.54) | <0.0001 (approx.) | 0.0009 | <0.0001 |
| **Remedies** | | | | |
| Tuned NegGrad+ on the group minus tuned NegGrad+ on random removal | -0.0010 (-0.0020 to +0.0001) | 0.078 | 1.000 | 0.123 |
| Tuned Fair Unlearning on the group minus tuned Fair Unlearning on random removal | +0.0016 (-0.0008 to +0.0039) | 0.162 | 1.000 | 0.240 |
| Tuned NegGrad+, readmissions unflagged, group minus random removal | +0.06 (-0.13 to +0.29) | 0.496 | 1.000 | 0.635 |
| Tuned Fair Unlearning, readmissions unflagged, group minus random removal | -0.45 (-1.18 to +0.10) | 0.090 | 1.000 | 0.139 |
| NegGrad+ at exact retraining's removal minus exact retraining (bootstrap over groups) | -0.0065 (-0.0114 to -0.0014) | 0.010 | 0.326 | 0.018 |
| Fair Unlearning at exact retraining's removal minus exact retraining (bootstrap over groups) | -0.0047 (-0.0104 to +0.0010) | 0.102 | 1.000 | 0.155 |
| **Remedies (all stays, sensitivity)** | | | | |
| Federation, consortium withdrawal minus random removal | -0.0032 (-0.0049 to -0.0010) | 0.001 (approx.) | 0.043 | 0.002 |
| Federation without the consortium minus full federation | -0.0076 (-0.0092 to -0.0056) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Federation, deaths unflagged, without the consortium minus full federation | +0.78 (+0.42 to +1.18) | <0.0001 (approx.) | 0.003 | 0.0002 |
| Federated penalty, teaching-status consortia | +0.0008 (-0.0019 to +0.0039) | 0.530 | 1.000 | 0.649 |
| **Confinement, MIMIC-IV** | | | | |
| Base alone (retained-only pipeline) minus pooled model | -0.0068 (-0.0123 to -0.0019) | 0.006 | 0.222 | 0.012 |
| Base with the validation-chosen component minus pooled model | +0.0105 (+0.0059 to +0.0155) | <0.0001 (approx.) | 0.0009 | <0.0001 |
| Control component minus base | +0.0012 (-0.0002 to +0.0024) | 0.076 | 1.000 | 0.123 |
| Pooled model with a cross-fitted residual component minus pooled model | +0.0111 (+0.0069 to +0.0155) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Adapted pooled comparator minus pooled model | +0.0098 (+0.0048 to +0.0137) | <0.0001 (approx.) | 0.0009 | <0.0001 |
| Confined system minus adapted pooled comparator | +0.0008 (-0.0035 to +0.0057) | 0.778 | 1.000 | 0.837 |
| Confined system minus residual-only comparator | -0.0006 (-0.0042 to +0.0027) | 0.742 | 1.000 | 0.823 |
| Full pooled comparator (component fitted on all of the group's data) minus pooled model | +0.0015 (-0.0012 to +0.0045) | 0.310 | 1.000 | 0.431 |
| Confined system minus full pooled comparator | +0.0090 (+0.0035 to +0.0146) | 0.004 | 0.152 | 0.008 |
| Five most isolated groups, base minus pooled model | -0.0216 (-0.0358 to -0.0051) | 0.018 | 0.558 | 0.031 |
| Five most isolated groups, confined minus residual-only comparator | -0.0021 (-0.0138 to +0.0113) | 0.692 | 1.000 | 0.796 |
| Five most isolated groups, confined minus adapted pooled comparator | -0.0006 (-0.0150 to +0.0216) | 0.872 | 1.000 | 0.897 |
| Five most isolated groups, full pooled comparator minus pooled model | +0.0090 (+0.0014 to +0.0193) | 0.012 | 0.384 | 0.021 |
| Deletion of the component: confined minus base | +0.0173 (+0.0123 to +0.0222) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Readmissions unflagged at a matched flag rate, confined minus pooled model | -1.56 (-2.75 to -0.70) | 0.0006 | 0.025 | 0.001 |
| Readmissions unflagged, confined minus adapted pooled comparator | -0.22 (-0.96 to +0.59) | 0.696 | 1.000 | 0.796 |
| Readmissions unflagged, base minus pooled model | +0.76 (-0.51 to +1.50) | 0.396 | 1.000 | 0.530 |
| **Several groups withdrawn at once, MIMIC-IV** | | | | |
| Five random groups, confined on the common base minus single-exclusion confined system | +0.0013 (-0.0031 to +0.0060) | 0.618 | 1.000 | 0.731 |
| Ten random groups, confined on the common base minus single-exclusion confined system | -0.0000 (-0.0056 to +0.0058) | 0.922 | 1.000 | 0.922 |
| Five most isolated groups, confined on the common base minus single-exclusion confined system | -0.0112 (-0.0276 to -0.0002) | 0.044 | 1.000 | 0.073 |
| Fifteen random groups, confined on the common base minus single-exclusion confined system | -0.0098 (-0.0181 to -0.0035) | 0.001 | 0.040 | 0.002 |
| **Confinement, eICU (all stays, sensitivity)** | | | | |
| Federation: consortium component minus full federation | +0.0220 (+0.0176 to +0.0257) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Federation: consortium-specific gain (component minus control component) | +0.0125 (+0.0083 to +0.0156) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Federation: deaths unflagged, component minus full federation | -2.69 (-3.50 to -1.91) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Centralised model: base alone (retained-only pipeline) minus pooled model | -0.0112 (-0.0135 to -0.0082) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| Centralised model: confined minus pooled model | +0.0001 (-0.0013 to +0.0011) | 0.910 | 1.000 | 0.922 |
| Centralised model: confined minus adapted pooled comparator | -0.0013 (-0.0020 to -0.0006) | 0.0003 (approx.) | 0.013 | 0.0008 |
| Centralised model: full pooled comparator minus pooled model | -0.0000 (-0.0014 to +0.0012) | 0.852 | 1.000 | 0.897 |
| Centralised model: deaths unflagged, confined minus pooled model | -0.18 (-0.51 to +0.41) | 0.706 | 1.000 | 0.796 |
| Centralised model: deaths unflagged, confined minus adapted pooled comparator | +0.06 (-0.17 to +0.63) | 0.296 | 1.000 | 0.429 |
| Centralised model: deaths unflagged, base minus pooled model | +2.28 (+1.19 to +2.98) | <0.0001 (approx.) | <0.0001 | <0.0001 |
| **Sharded training** | | | | |
| Group-aligned ensemble minus pooled model, before deletion | -0.0039 (-0.0173 to +0.0072) | 0.566 | 1.000 | 0.681 |
| Group-aligned ensemble minus pooled model, after deleting the group's shard | -0.0071 (-0.0210 to +0.0044) | 0.316 | 1.000 | 0.431 |
| Confined system minus group-aligned ensemble | +0.0144 (+0.0058 to +0.0261) | 0.0004 | 0.017 | 0.0010 |
