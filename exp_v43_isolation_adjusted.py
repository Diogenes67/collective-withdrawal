"""
exp_v43_isolation_adjusted.py. Adjusted regressions of the withdrawal penalty on the equal-pool isolation ratio (external review,
statistical point 2), from results/v34_isolation/figure_data.csv (aggregate). Models, each by OLS with the slope interval from 5,000
bootstrap draws over groups (RandomState(0)): (a) all 49 groups; (b) 48 groups without the white group, whose control was capped;
(c) (b) plus dataset (MIMIC-IV or eICU) and family intercepts; (d) (c) plus log group size and the withdrawn fraction of training
units (patients on MIMIC-IV, stays on eICU). Writes results/v43_isolation_adjusted/{summary.txt, regressions.json}.
"""
import os, json
import numpy as np, pandas as pd
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); OUT = f"{ROOT}/results/v43_isolation_adjusted"; os.makedirs(OUT, exist_ok=True)
D = pd.read_csv(f"{ROOT}/results/v34_isolation/figure_data.csv")
iso = pd.read_csv(f"{ROOT}/results/v34_isolation/isolation.csv")[["dataset", "family", "group", "n_retained"]]
D = D.merge(iso, on=["dataset", "family", "group"], how="left")
D["log_n"] = np.log(D.n_members); D["frac"] = D.n_members / (D.n_members + D.n_retained)
def design(d, cov):
    X = [np.ones(len(d)), d.ratio_equal.values]; names = ["const", "ratio"]
    if "fam" in cov:
        for f in sorted(d.family_group.unique())[1:]: X.append((d.family_group == f).astype(float).values); names.append(f"fam:{f}")
        for ds in sorted(d.dataset.unique())[1:]:
            col = (d.dataset == ds).astype(float).values
            if not any(np.allclose(col, x) for x in X[2:]): X.append(col); names.append(f"ds:{ds}")
    if "size" in cov: X += [d.log_n.values, d.frac.values]; names += ["log_n", "frac"]
    return np.column_stack(X), names
def fit(d, cov):
    X, names = design(d, cov); b, *_ = np.linalg.lstsq(X, d.penalty.values, rcond=None); return b[1], names
models = {"a_all49": (D, ()), "b_48_no_white": (D[~D.control_capped], ()), "c_48_family_dataset": (D[~D.control_capped], ("fam",)),
          "d_48_family_dataset_size": (D[~D.control_capped], ("fam", "size")), "e_all49_family_dataset_size": (D, ("fam", "size"))}
res = {}; L = ["v43 adjusted isolation regressions (OLS; 5,000 bootstrap draws over groups)"]
for k, (d, cov) in models.items():
    d = d.reset_index(drop=True); est, names = fit(d, cov); r = np.random.RandomState(0); bs = []
    for _ in range(5000):
        idx = r.choice(len(d), len(d), True); dd = d.iloc[idx].reset_index(drop=True)
        try: bs.append(fit(dd, cov)[0])
        except Exception: bs.append(np.nan)
    bs = np.array(bs); bs = bs[np.isfinite(bs)]; lo, hi = np.percentile(bs, [2.5, 97.5]); p = min(1, max(2 / len(bs), 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    res[k] = dict(n=len(d), slope=float(est), ci=[float(lo), float(hi)], p=float(p), terms=names)
    L.append(f"  {k:32s} n={len(d)} slope {est:+.4f} [{lo:+.4f}, {hi:+.4f}] p {p:.4f}  terms {names}")
json.dump(res, open(f"{OUT}/regressions.json", "w"), indent=1); open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
