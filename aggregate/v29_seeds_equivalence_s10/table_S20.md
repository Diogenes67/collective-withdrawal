**Table S20.** Training randomness. Each contrast estimated seed by seed on the full held-out data, with the between-seed standard deviation and an interval that adds the seed component to the evaluation bootstrap (SE_total = sqrt(SE_boot² + s²/k), k the number of seeds; normal approximation). The evaluation-only interval is the manuscript's two-stage (MIMIC-IV) or hospital-cluster (eICU) percentile bootstrap, recomputed here from the cached predictions.

| Contrast | Seeds | Seed mean (95% evaluation bootstrap) | Per-seed estimates, lowest to highest | Between-seed SD | 95% interval with the seed component |
|---|---|---|---|---|---|
| MIMIC-IV, 25 groups: Withdrawal minus matched random removal (exact retraining) | 10 | −0.0114 (−0.0163 to −0.0072) | −0.0139 to −0.0085 | 0.0015 | −0.0162 to −0.0067 |
| MIMIC-IV, 25 groups: Withdrawal minus pooled model | 10 | −0.0099 (−0.0143 to −0.0059) | −0.0146 to −0.0054 | 0.0026 | −0.0144 to −0.0054 |
| MIMIC-IV, 25 groups: Confined system minus adapted pooled comparator | 10 | −0.0008 (−0.0043 to +0.0021) | −0.0042 to +0.0042 | 0.0023 | −0.0043 to +0.0026 |
| MIMIC-IV, 25 groups: Confined system minus pooled model | 10 | +0.0087 (+0.0043 to +0.0131) | +0.0035 to +0.0113 | 0.0023 | +0.0039 to +0.0134 |
| MIMIC-IV, 25 groups: Confined system minus residual-only comparator | 10 | −0.0019 (−0.0047 to +0.0005) | −0.0049 to +0.0014 | 0.0017 | −0.0047 to +0.0009 |
| MIMIC-IV, 25 groups: Deletion of the component: confined minus base | 10 | +0.0173 (+0.0120 to +0.0222) | +0.0155 to +0.0189 | 0.0010 | +0.0122 to +0.0223 |
| MIMIC-IV, five most isolated groups: Withdrawal minus matched random removal (exact retraining) | 10 | −0.0249 (−0.0414 to −0.0079) | −0.0311 to −0.0209 | 0.0041 | −0.0412 to −0.0086 |
| MIMIC-IV, five most isolated groups: Withdrawal minus pooled model | 10 | −0.0219 (−0.0356 to −0.0057) | −0.0309 to −0.0066 | 0.0079 | −0.0375 to −0.0062 |
| MIMIC-IV, five most isolated groups: Confined system minus adapted pooled comparator | 10 | −0.0039 (−0.0167 to +0.0103) | −0.0168 to +0.0077 | 0.0080 | −0.0181 to +0.0103 |
| MIMIC-IV, five most isolated groups: Confined system minus pooled model | 10 | −0.0029 (−0.0137 to +0.0104) | −0.0067 to +0.0020 | 0.0033 | −0.0145 to +0.0088 |
| MIMIC-IV, five most isolated groups: Confined system minus residual-only comparator | 10 | −0.0051 (−0.0140 to +0.0045) | −0.0100 to +0.0005 | 0.0031 | −0.0146 to +0.0044 |
| MIMIC-IV, five most isolated groups: Deletion of the component: confined minus base | 10 | +0.0170 (−0.0031 to +0.0302) | +0.0058 to +0.0257 | 0.0067 | −0.0005 to +0.0344 |
