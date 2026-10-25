**Table S20.** Training randomness. Each contrast estimated seed by seed on the full held-out data, with the between-seed standard deviation and an interval that adds the seed component to the evaluation bootstrap (SE_total = sqrt(SE_boot² + s²/3); normal approximation; three seeds, so s has two degrees of freedom). The evaluation-only interval is the manuscript's two-stage (MIMIC-IV) or hospital-cluster (eICU) percentile bootstrap, recomputed here from the cached predictions.

| Contrast | Seed mean (95% evaluation bootstrap) | Seed 0 | Seed 1 | Seed 2 | Between-seed SD | 95% interval with the seed component |
|---|---|---|---|---|---|---|
| MIMIC-IV, 25 groups: Withdrawal minus matched random removal (exact retraining) | −0.0106 (−0.0157 to −0.0061) | −0.0102 | −0.0101 | −0.0116 | 0.0008 | −0.0156 to −0.0057 |
| MIMIC-IV, 25 groups: Withdrawal minus pooled model | −0.0083 (−0.0132 to −0.0039) | −0.0080 | −0.0054 | −0.0116 | 0.0031 | −0.0142 to −0.0025 |
| MIMIC-IV, 25 groups: Confined system minus adapted pooled comparator | +0.0008 (−0.0035 to +0.0057) | −0.0017 | −0.0003 | +0.0042 | 0.0031 | −0.0049 to +0.0064 |
| MIMIC-IV, 25 groups: Confined system minus pooled model | +0.0105 (+0.0059 to +0.0155) | +0.0100 | +0.0102 | +0.0113 | 0.0007 | +0.0057 to +0.0153 |
| MIMIC-IV, 25 groups: Confined system minus residual-only comparator | −0.0006 (−0.0042 to +0.0027) | −0.0012 | −0.0019 | +0.0014 | 0.0017 | −0.0046 to +0.0035 |
| MIMIC-IV, 25 groups: Deletion of the component: confined minus base | +0.0173 (+0.0123 to +0.0222) | +0.0176 | +0.0155 | +0.0189 | 0.0017 | +0.0120 to +0.0227 |
| MIMIC-IV, five most isolated groups: Withdrawal minus matched random removal (exact retraining) | −0.0239 (−0.0396 to −0.0085) | −0.0281 | −0.0218 | −0.0218 | 0.0036 | −0.0397 to −0.0081 |
| MIMIC-IV, five most isolated groups: Withdrawal minus pooled model | −0.0210 (−0.0357 to −0.0065) | −0.0259 | −0.0066 | −0.0306 | 0.0127 | −0.0413 to −0.0007 |
| MIMIC-IV, five most isolated groups: Confined system minus adapted pooled comparator | −0.0006 (−0.0150 to +0.0216) | −0.0054 | −0.0040 | +0.0077 | 0.0072 | −0.0211 to +0.0200 |
| MIMIC-IV, five most isolated groups: Confined system minus pooled model | +0.0004 (−0.0111 to +0.0170) | +0.0008 | +0.0020 | −0.0015 | 0.0018 | −0.0138 to +0.0146 |
| MIMIC-IV, five most isolated groups: Confined system minus residual-only comparator | −0.0021 (−0.0138 to +0.0113) | −0.0026 | −0.0042 | +0.0005 | 0.0024 | −0.0148 to +0.0106 |
| MIMIC-IV, five most isolated groups: Deletion of the component: confined minus base | +0.0220 (+0.0050 to +0.0368) | +0.0257 | +0.0155 | +0.0248 | 0.0057 | +0.0054 to +0.0386 |
| eICU, 10 consortia: Consortium withdrawal minus matched random removal (exact retraining) | −0.0066 (−0.0084 to −0.0046) | −0.0055 | −0.0074 | −0.0070 | 0.0010 | −0.0089 to −0.0044 |
| eICU, 10 consortia: Consortium withdrawal minus pooled model | −0.0096 (−0.0115 to −0.0076) | −0.0100 | −0.0089 | −0.0099 | 0.0006 | −0.0117 to −0.0075 |
| eICU, 10 consortia: Federation: consortium withdrawal minus random removal | −0.0032 (−0.0049 to −0.0010) | −0.0027 | −0.0030 | −0.0039 | 0.0006 | −0.0052 to −0.0011 |
| eICU, 10 consortia: Centralised model: confined (strict) minus adapted pooled comparator | −0.0013 (−0.0020 to −0.0005) | −0.0016 | −0.0011 | −0.0013 | 0.0002 | −0.0021 to −0.0005 |
| eICU, 10 consortia: Centralised model: confined (strict) minus pooled model | +0.0001 (−0.0013 to +0.0012) | −0.0012 | +0.0006 | +0.0008 | 0.0011 | −0.0017 to +0.0018 |
