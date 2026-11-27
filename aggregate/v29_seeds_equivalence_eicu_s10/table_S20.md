**Table S20.** Training randomness. Each contrast estimated seed by seed on the full held-out data, with the between-seed standard deviation and an interval that adds the seed component to the evaluation bootstrap (SE_total = sqrt(SE_boot² + s²/k), k the number of seeds; normal approximation). The evaluation-only interval is the manuscript's two-stage (MIMIC-IV) or hospital-cluster (eICU) percentile bootstrap, recomputed here from the cached predictions.

| Contrast | Seeds | Seed mean (95% evaluation bootstrap) | Per-seed estimates, lowest to highest | Between-seed SD | 95% interval with the seed component |
|---|---|---|---|---|---|
| eICU, 10 consortia: Consortium withdrawal minus matched random removal (exact retraining) | 10 | −0.0066 (−0.0082 to −0.0043) | −0.0076 to −0.0049 | 0.0010 | −0.0086 to −0.0045 |
| eICU, 10 consortia: Consortium withdrawal minus pooled model | 10 | −0.0092 (−0.0110 to −0.0070) | −0.0105 to −0.0077 | 0.0010 | −0.0113 to −0.0072 |
| eICU, 10 consortia: Federation: consortium withdrawal minus random removal | 3 | −0.0032 (−0.0049 to −0.0010) | −0.0039 to −0.0027 | 0.0006 | −0.0052 to −0.0011 |
| eICU, 10 consortia: Centralised model: confined (strict) minus adapted pooled comparator | 10 | −0.0011 (−0.0018 to −0.0005) | −0.0019 to +0.0000 | 0.0006 | −0.0019 to −0.0004 |
| eICU, 10 consortia: Centralised model: confined (strict) minus pooled model | 10 | +0.0002 (−0.0011 to +0.0013) | −0.0012 to +0.0014 | 0.0007 | −0.0011 to +0.0015 |
