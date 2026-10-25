**Table S19.** Multiplicity adjustment for the contrasts reported in the main text. Two-sided p-values derived from each 95% bootstrap interval under a normal approximation, adjusted over all 55 contrasts by Holm's method (family-wise error rate) and by the Benjamini-Hochberg procedure (false discovery rate). Unadjusted intervals are those reported in the text and tables.

| Contrast | Estimate (95% interval) | p | Holm-adjusted p | Benjamini-Hochberg q |
|---|---|---|---|---|
| **Withdrawal cost, MIMIC-IV** | | | | |
| Group withdrawal minus matched random removal, 25 clustered groups | -0.0106 (-0.0157 to -0.0061) | <0.0001 | 0.0006 | <0.0001 |
| Group withdrawal minus pooled model | -0.0083 (-0.0132 to -0.0039) | 0.0004 | 0.014 | 0.0010 |
| Group withdrawal minus random removal, readmissions unflagged at a matched flag rate | +1.06 (+0.12 to +2.21) | 0.047 | 1.000 | 0.078 |
| Race groups, withdrawal minus random removal (bootstrap over 7 groups) | -0.0014 (-0.0087 to +0.0045) | 0.677 | 1.000 | 0.792 |
| Race groups other than white, withdrawal minus random removal (bootstrap over 6 groups) | +0.0019 (-0.0012 to +0.0059) | 0.297 | 1.000 | 0.419 |
| Gradient-boosted trees, withdrawal minus random removal (bootstrap over groups) | -0.0081 (-0.0131 to -0.0043) | 0.0003 | 0.010 | 0.0007 |
| Logistic regression, withdrawal minus random removal (bootstrap over groups) | -0.0024 (-0.0040 to -0.0008) | 0.004 | 0.118 | 0.008 |
| **Withdrawal cost, eICU** | | | | |
| Consortium withdrawal minus matched random removal, 10 consortia | -0.0066 (-0.0084 to -0.0045) | <0.0001 | <0.0001 | <0.0001 |
| Consortium withdrawal minus pooled model | -0.0096 (-0.0114 to -0.0074) | <0.0001 | <0.0001 | <0.0001 |
| Deaths unflagged at a matched flag rate, withdrawal minus pooled model | +1.76 (+1.09 to +2.29) | <0.0001 | <0.0001 | <0.0001 |
| Deaths unflagged at a matched flag rate, withdrawal minus random removal | +1.07 (+0.56 to +1.54) | <0.0001 | 0.0006 | <0.0001 |
| 24-hour landmark, withdrawal minus random removal | -0.0049 (-0.0064 to -0.0032) | <0.0001 | <0.0001 | <0.0001 |
| **Remedies** | | | | |
| NegGrad+ on the group minus NegGrad+ on random removal | -0.0361 (-0.0477 to -0.0253) | <0.0001 | <0.0001 | <0.0001 |
| Fair Unlearning on the group minus Fair Unlearning on random removal | -0.0404 (-0.0539 to -0.0288) | <0.0001 | <0.0001 | <0.0001 |
| NegGrad+, readmissions unflagged, group minus random removal | +3.85 (+2.19 to +5.69) | <0.0001 | 0.0006 | <0.0001 |
| Fair Unlearning, readmissions unflagged, group minus random removal | +4.69 (+2.72 to +6.40) | <0.0001 | <0.0001 | <0.0001 |
| Federation, consortium withdrawal minus random removal | -0.0032 (-0.0049 to -0.0010) | 0.001 | 0.034 | 0.002 |
| Federation without the consortium minus full federation | -0.0076 (-0.0092 to -0.0056) | <0.0001 | <0.0001 | <0.0001 |
| Federation, deaths unflagged, without the consortium minus full federation | +0.78 (+0.42 to +1.18) | <0.0001 | 0.002 | 0.0002 |
| Federated penalty, teaching-status consortia | +0.0008 (-0.0019 to +0.0039) | 0.583 | 1.000 | 0.699 |
| Federated penalty at the 24-hour landmark | -0.0008 (-0.0022 to +0.0011) | 0.364 | 1.000 | 0.488 |
| **Confinement, MIMIC-IV** | | | | |
| Base alone (retained-only pipeline) minus pooled model | -0.0068 (-0.0123 to -0.0019) | 0.011 | 0.256 | 0.018 |
| Base with the validation-chosen component minus pooled model | +0.0105 (+0.0059 to +0.0155) | <0.0001 | 0.0006 | <0.0001 |
| Control component minus base | +0.0012 (-0.0002 to +0.0024) | 0.063 | 1.000 | 0.098 |
| Pooled model with a cross-fitted residual component minus pooled model | +0.0111 (+0.0069 to +0.0155) | <0.0001 | <0.0001 | <0.0001 |
| Adapted pooled comparator minus pooled model | +0.0098 (+0.0048 to +0.0137) | <0.0001 | 0.0006 | <0.0001 |
| Confined system minus adapted pooled comparator | +0.0008 (-0.0035 to +0.0057) | 0.747 | 1.000 | 0.822 |
| Confined system minus residual-only comparator | -0.0006 (-0.0042 to +0.0027) | 0.746 | 1.000 | 0.822 |
| Full pooled comparator (component fitted on all of the group's data) minus pooled model | +0.0015 (-0.0012 to +0.0045) | 0.310 | 1.000 | 0.426 |
| Confined system minus full pooled comparator | +0.0090 (+0.0035 to +0.0146) | 0.002 | 0.046 | 0.003 |
| Five most isolated groups, base minus pooled model | -0.0216 (-0.0358 to -0.0051) | 0.006 | 0.152 | 0.011 |
| Five most isolated groups, confined minus residual-only comparator | -0.0021 (-0.0138 to +0.0113) | 0.742 | 1.000 | 0.822 |
| Five most isolated groups, confined minus adapted pooled comparator | -0.0006 (-0.0150 to +0.0216) | 0.951 | 1.000 | 0.969 |
| Five most isolated groups, full pooled comparator minus pooled model | +0.0090 (+0.0014 to +0.0193) | 0.051 | 1.000 | 0.082 |
| Deletion of the component: confined minus base | +0.0173 (+0.0123 to +0.0222) | <0.0001 | <0.0001 | <0.0001 |
| Readmissions unflagged at a matched flag rate, confined minus pooled model | -1.56 (-2.75 to -0.70) | 0.003 | 0.084 | 0.006 |
| Readmissions unflagged, confined minus adapted pooled comparator | -0.22 (-0.96 to +0.59) | 0.585 | 1.000 | 0.699 |
| Readmissions unflagged, base minus pooled model | +0.76 (-0.51 to +1.50) | 0.141 | 1.000 | 0.209 |
| **Several groups withdrawn at once, MIMIC-IV** | | | | |
| Five random groups, confined on the common base minus single-exclusion confined system | +0.0013 (-0.0031 to +0.0060) | 0.567 | 1.000 | 0.699 |
| Ten random groups, confined on the common base minus single-exclusion confined system | -0.0000 (-0.0056 to +0.0058) | 0.998 | 1.000 | 0.998 |
| Five most isolated groups, confined on the common base minus single-exclusion confined system | -0.0112 (-0.0276 to -0.0002) | 0.109 | 1.000 | 0.166 |
| Fifteen random groups, confined on the common base minus single-exclusion confined system | -0.0098 (-0.0181 to -0.0035) | 0.008 | 0.209 | 0.015 |
| **Confinement, eICU** | | | | |
| Federation: consortium component minus full federation | +0.0220 (+0.0176 to +0.0257) | <0.0001 | <0.0001 | <0.0001 |
| Federation: consortium-specific gain (component minus control component) | +0.0125 (+0.0083 to +0.0156) | <0.0001 | <0.0001 | <0.0001 |
| Federation: deaths unflagged, component minus full federation | -2.69 (-3.50 to -1.91) | <0.0001 | <0.0001 | <0.0001 |
| Centralised model: base alone (retained-only pipeline) minus pooled model | -0.0112 (-0.0135 to -0.0082) | <0.0001 | <0.0001 | <0.0001 |
| Centralised model: confined minus pooled model | +0.0001 (-0.0013 to +0.0011) | 0.893 | 1.000 | 0.945 |
| Centralised model: confined minus adapted pooled comparator | -0.0013 (-0.0020 to -0.0006) | 0.0003 | 0.010 | 0.0007 |
| Centralised model: full pooled comparator minus pooled model | -0.0000 (-0.0014 to +0.0012) | 0.941 | 1.000 | 0.969 |
| Centralised model: deaths unflagged, confined minus pooled model | -0.18 (-0.51 to +0.41) | 0.435 | 1.000 | 0.570 |
| Centralised model: deaths unflagged, confined minus adapted pooled comparator | +0.06 (-0.17 to +0.63) | 0.785 | 1.000 | 0.847 |
| Centralised model: deaths unflagged, base minus pooled model | +2.28 (+1.19 to +2.98) | <0.0001 | <0.0001 | <0.0001 |
| **Sharded training** | | | | |
| Group-aligned ensemble minus pooled model, before deletion | -0.0039 (-0.0173 to +0.0072) | 0.530 | 1.000 | 0.677 |
| Group-aligned ensemble minus pooled model, after deleting the group's shard | -0.0071 (-0.0210 to +0.0044) | 0.276 | 1.000 | 0.400 |
| Confined system minus group-aligned ensemble | +0.0144 (+0.0058 to +0.0261) | 0.005 | 0.146 | 0.010 |
