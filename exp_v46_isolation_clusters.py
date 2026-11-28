"""
exp_v46_isolation_clusters.py. Slope of the withdrawal penalty on the equal-pool isolation ratio within the 25 clustered
MIMIC-IV groups, which partition the training patients and so do not overlap (external review, 28 Nov 2026). The 48-group
bootstrap of exp_v34_isolation.py resamples overlapping groups as independent; this is the within-family check the main
text leads with. Reads results/v34_isolation/figure_data.csv (one row per group: ratio_equal, penalty). OLS slope,
5,000 bootstrap draws over groups, RandomState(0), Spearman correlation. Writes results/v46_isolation_clusters/summary.json.
"""
import os, json, numpy as np, pandas as pd
R = "results"
d = pd.read_csv(f"{R}/v34_isolation/figure_data.csv")
c = d[d.family_group == "clustered"]; assert len(c) == 25, len(c)
x, y = c.ratio_equal.values, c.penalty.values
slope = float(np.polyfit(x, y, 1)[0]); rs = np.random.RandomState(0); bs = []
for _ in range(5000):
    i = rs.randint(0, len(x), len(x))
    if np.ptp(x[i]) > 0: bs.append(np.polyfit(x[i], y[i], 1)[0])
out = dict(n_groups=25, slope=slope, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_boot=len(bs),
           spearman=float(pd.Series(x).rank().corr(pd.Series(y).rank())), ratio_range=[float(x.min()), float(x.max())])
os.makedirs(f"{R}/v46_isolation_clusters", exist_ok=True)
json.dump(out, open(f"{R}/v46_isolation_clusters/summary.json", "w"), indent=1); print(out)
