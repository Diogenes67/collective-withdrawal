"""
exp_v39_eicu_seed_preds.py. Prediction cache for the extra eICU training seeds (review item M6: the ten-seed check on eICU).
For each seed in SEEDS and each consortium, the pooled model (results/model_cache_eicu_es/orig_s{s}.pt), the exact retrain
without the consortium and the retrain without its stay-matched random set (results/v7_eicu_consortium_es/ckpt/exact_{key}_s{s}.pt
and exactr_{key}_s{s}.pt, written by exp_v7_eicu_consortium_v72.py with SEEDS=3,...,9) are scored on the whole held-out set, in
the layout of the v13.1 cache (P[(key, seed, arm)]), so that exp_v29_seeds_equivalence.py can run the centralised eICU
contrasts on ten seeds (ESEEDS=0,...,9). The federation arms exist for seeds 0 to 2 only and are not extended.
Data pipeline, split, consortium definitions and model exactly as v7.2. Resumable: a seed and consortium already in the cache
is skipped; the cache is rewritten atomically after each consortium.

Env: ROOT, SEEDS (default 3,4,5,6,7,8,9), DEVICE, QUICK=1 (seed 3, two consortia, separate output). Reads derived/eicu_cohort.parquet,
results/model_cache_eicu_es, results/v7_eicu_consortium_es/ckpt. Writes results/v39_eicu_seed_core/{preds_eicu.npz, runs/*.json, log.txt}.
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); V72 = f"{ROOT}/results/v7_eicu_consortium_es"; ORIG = f"{ROOT}/results/model_cache_eicu_es"
OUT = f"{ROOT}/results/v39_eicu_seed_core" + ("_quick" if QUICK else ""); os.makedirs(f"{OUT}/runs", exist_ok=True)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "3" if QUICK else "3,4,5,6,7,8,9").split(",")]
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; quick={QUICK}; seeds {SEEDS}; out {OUT}")

# ---------------------------------------------------------------- data, identical to v7.2
df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
for c in Xdf.columns[Xdf.isna().any()]:
    Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32); hosp_id = df.hospitalid.values
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
test_idx = np.where(is_test)[0]; y_te = y[test_idx]
hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); groups = []
for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
    for v in sorted(hattr[col].dropna().unique(), key=str):
        hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
        if f_tr.sum() < 800 or f_te.sum() < 100: continue
        groups.append(dict(key=f"{fam}_{str(v).replace(' ', '_').replace('/', '_')}", f_te=f_te))
if QUICK: groups = groups[:2]
log(f"{len(groups)} consortia; {len(test_idx)} held-out stays")

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def load(path):
    if not os.path.exists(path): raise SystemExit(f"missing checkpoint {path}; run exp_v7_eicu_consortium_v72.py with SEEDS first")
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def predict(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(torch.tensor(Xa[i:i + 65536], device=DEVICE))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")

cache = f"{OUT}/preds_eicu.npz"
P = np.load(cache, allow_pickle=True)["P"].item() if os.path.exists(cache) else {}
def save():
    np.savez_compressed(cache + ".tmp.npz", P=np.array(P, dtype=object)); os.replace(cache + ".tmp.npz", cache)
for s in SEEDS:
    p_orig = None
    for g in groups:
        key = g["key"]
        if all((key, s, a) in P for a in ["original", "exact_group", "exact_random"]): continue
        if p_orig is None: p_orig = predict(load(f"{ORIG}/orig_s{s}.pt"), X[test_idx])
        tg = time.time()
        P[(key, s, "original")] = p_orig
        P[(key, s, "exact_group")] = predict(load(f"{V72}/ckpt/exact_{key}_s{s}.pt"), X[test_idx])
        P[(key, s, "exact_random")] = predict(load(f"{V72}/ckpt/exactr_{key}_s{s}.pt"), X[test_idx])
        pos = np.where(g["f_te"][test_idx])[0]; rec = dict(consortium=key, seed=s, n_test_stays=int(len(pos)), auroc={a: auc(y_te[pos], P[(key, s, a)][pos]) for a in ["original", "exact_group", "exact_random"]},
                                                        full_test_auroc={a: auc(y_te, P[(key, s, a)]) for a in ["original", "exact_group", "exact_random"]}, date=time.strftime("%Y-%m-%d %H:%M:%S"))
        ref = f"{V72}/runs/{key.split('_', 1)[0]}__{key.split('_', 1)[1]}__s{s}.json"
        if os.path.exists(ref):
            rj = json.load(open(ref)); rec["v72_auroc"] = {a: rj["models"][a]["auroc"] for a in ["original", "exact_group", "exact_random"]}
            gap = max(abs(rec["auroc"][a] - rec["v72_auroc"][a]) for a in rec["v72_auroc"])
            if gap > 1e-4: log(f"  WARNING {key} s{s}: AUROC differs from the v7.2 run record by {gap:.2e}")
        json.dump(rec, open(f"{OUT}/runs/{key}__s{s}.json", "w"), indent=1); save()
        log(f"{key} s{s}: original {rec['auroc']['original']:.4f} exact {rec['auroc']['exact_group']:.4f} random {rec['auroc']['exact_random']:.4f} ({time.time()-tg:.0f}s)")
log(f"done; cache holds {len({s for (_, s, _) in P})} seeds x {len({k for (k, _, _) in P})} consortia in {cache}")
