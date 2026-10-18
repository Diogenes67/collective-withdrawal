"""exp_v13c: extra eICU contrasts for the permission-state table, from the v13.1 cached predictions (no models, no torch).
Hospital-cluster bootstrap of the held-out set exactly as v13 (one hospital draw feeds all ten consortia; AUROC per consortium is the
mean over seeds; contrasts are means over consortia). Run in chunks: CHUNK=k (0..4) draws 200 bootstraps with RandomState(1000+k)
and writes results/v13_intervals_1/extra_chunk{k}.npz; CHUNK=combine writes extra_contrasts.json.
"""
import os, json, numpy as np, pandas as pd
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); OUT = f"{ROOT}/results/v13_intervals_1"
CHUNK = os.environ.get("CHUNK", "combine"); NPER = 200; SEEDS = [0, 1, 2]
ARMS = ["original", "exact_group", "fl_global", "fl_without", "fl_resid"]
PAIRS = {"exact_group minus original (centralised, state 3 vs state 1)": ("exact_group", "original"),
         "fl_without minus fl_global (federated, state 3 vs state 1)": ("fl_without", "fl_global"),
         "fl_global minus original (federation vs centralised pooled)": ("fl_global", "original"),
         "fl_resid minus original (federated state 2 vs centralised pooled)": ("fl_resid", "original")}
def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["uniquepid", "hospitalid", "died"])
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values; y_te = df.died.values[is_test].astype(int); hosp_te = df.hospitalid.values[is_test]
P = np.load(f"{OUT}/preds_eicu.npz", allow_pickle=True)["P"].item(); assert len(P[next(iter(P))]) == len(y_te)
mem = json.load(open(f"{OUT}/membership_eicu.json")); keys = list(mem)
hospitals = np.unique(hosp_te); h_rows = {h: np.where(hosp_te == h)[0] for h in hospitals}
def contrasts(rows, hs):
    acc = {k: [] for k in PAIRS}
    for key in keys:
        m = np.isin(hs, mem[key]); rr = rows[m]
        if len(rr) < 50 or len(np.unique(y_te[rr])) < 2: continue
        a = {arm: np.nanmean([auc(y_te[rr], P[(key, s, arm)][rr]) for s in SEEDS]) for arm in ARMS}
        for k, (x, z) in PAIRS.items(): acc[k].append(a[x] - a[z])
    return {k: float(np.mean(v)) for k, v in acc.items()}
if CHUNK == "combine":
    pt = contrasts(np.arange(len(y_te)), hosp_te); B = {k: [] for k in PAIRS}
    for c in range(5):
        Z = np.load(f"{OUT}/extra_chunk{c}.npz", allow_pickle=True)["B"].item()
        for k in PAIRS: B[k] += Z[k]
    res = {f"eICU | all | {k}": dict(point=pt[k], ci=[float(np.percentile(B[k], 2.5)), float(np.percentile(B[k], 97.5))], n_boot=len(B[k]), unit="hospital cluster") for k in PAIRS}
    json.dump(res, open(f"{OUT}/extra_contrasts.json", "w"), indent=1)
    for k, v in res.items(): print(f"{k:75s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] n={v['n_boot']}")
else:
    c = int(CHUNK); r = np.random.RandomState(1000 + c); B = {k: [] for k in PAIRS}
    for b in range(NPER):
        drawn = r.choice(hospitals, len(hospitals), True); rows = np.concatenate([h_rows[h] for h in drawn]); hs = np.concatenate([np.full(len(h_rows[h]), h) for h in drawn])
        for k, v in contrasts(rows, hs).items(): B[k].append(v)
    np.savez_compressed(f"{OUT}/extra_chunk{c}.npz", B=np.array(B, dtype=object)); print("chunk", c, "done")
