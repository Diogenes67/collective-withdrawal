"""
exp_v22_unlearning_flags.py. Clinical size of the approximate-unlearning cost: events unflagged at a matched flag rate for
fine-tuning, NegGrad+ and Fair Unlearning (both protection sets), MIMIC-IV, 25 clustered groups x 3 seeds. No training: every
model is loaded from the saved v5/v10 checkpoints (results/v5_partitions/ckpt: orig, exact, exactrp, ft, ftrp, ng, ngrp,
fucomm, fucommrp, furace, furacerp).

Each model is scored on the group's held-out admissions. Matched flag rate as v13b/v19: k = number of the group's held-out
admissions the original model flags at its system-wide top-decile threshold; every arm flags its own top k; events unflagged
per 1,000 admissions. Contrasts: each method applied to the group minus the same method applied to the patient-matched random
removal (the paired contrast reported for AUROC in v10), and each method minus the original model. AUROC is recomputed as a
check against v10 (NegGrad+ -0.036, Fair Unlearning community -0.040). Two-stage bootstrap (groups, then patients; 1,000 draws).

Env: ROOT, QUICK=1, DEVICE, NBOOT. Writes results/v22_unlearning_flags/{preds, summary.csv, intervals.json}.
"""
import os, json, time, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); CK = f"{ROOT}/results/v5_partitions/ckpt"; V5 = f"{ROOT}/results/v5_partitions"
OUT = f"{ROOT}/results/v22_unlearning_flags" + ("_quick" if QUICK else ""); os.makedirs(f"{OUT}/preds", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], min_patients=300, n_boot=int(os.environ.get("NBOOT", 50 if QUICK else 1000)), version="v22.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ---------------------------------------------------------------- data, split, pipeline and groups exactly as v5/v19/v21
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
feat_lab = [c for c in feat if c.startswith("lab_")]
y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test; test_idx = np.where(is_test)[0]
Xs = df[feat].astype(float)
for c in feat_lab: Xs[c + "_missing"] = Xs[c].isna().astype(float); Xs[c] = Xs[c].fillna(Xs[c].median())
Xs = Xs.values.astype(np.float32); mu0, sd0 = Xs[is_train].mean(0), Xs[is_train].std(0) + 1e-6; Xs = (Xs - mu0) / sd0
pat_mean = pd.DataFrame(Xs).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
cluster = assign.reindex(sid).values
groups = []
for c in range(25):
    sh = f"c{c:02d}"; mem = cluster == c; f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    ref = f"{V5}/runs/community_trainfit__{sh}__s0.json"
    if os.path.exists(ref):
        r = json.load(open(ref))
        if r["n_train_patients"] != n_tr_p or r["n_test_patients"] != n_te_p: raise SystemExit(f"{sh}: membership differs from v5")
    groups.append(dict(key=f"community_trainfit_{sh}", name=f"community_trainfit/{sh}", te=np.where(f_te)[0]))
log(f"{len(groups)} groups")

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def probs(path, rows):
    m = MLP(Xs.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(rows), 65536): out.append(torch.sigmoid(m(torch.tensor(Xs[rows[i:i + 65536]], device=DEVICE))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
# method -> (checkpoint prefix applied to the group, prefix applied to the matched random removal)
METHODS = {"exact": ("exact", "exactrp"), "finetune": ("ft", "ftrp"), "neggrad": ("ng", "ngrp"), "fu_community": ("fucomm", "fucommrp"), "fu_race": ("furace", "furacerp")}
thr = {}
for s in CFG["seeds"]:
    p_all = probs(f"{CK}/orig_s{s}.pt", test_idx); thr[s] = float(np.percentile(p_all, 90))
pos = {i: j for j, i in enumerate(test_idx)}
missing = set()
for g in groups:
    for s in CFG["seeds"]:
        f = f"{OUT}/preds/{g['key']}__s{s}.npz"
        if os.path.exists(f): continue
        te = g["te"]; p = {"original": probs(f"{CK}/orig_s{s}.pt", te)}
        for meth, (pg, pr) in METHODS.items():
            for arm, pre in [(meth, pg), (meth + "_random", pr)]:
                path = f"{CK}/{pre}_{g['key']}_s{s}.pt"
                if os.path.exists(path): p[arm] = probs(path, te)
                else: missing.add(pre)
        k_flag = int((p["original"] >= thr[s]).sum())
        np.savez_compressed(f, y=y[te], pid=sid[te], k_flag=k_flag, **p)
    log(f"{g['key']} scored")
if missing: log(f"missing checkpoint families (arms skipped): {sorted(missing)}")

# ---------------------------------------------------------------- summary and two-stage bootstrap
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); k_, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(k_, int(s_))] = {k: Z[k] for k in Z.files}
seeds = CFG["seeds"]; keys = sorted(set(k for k, _ in P if all((k, s_) in P for s_ in seeds)))
arms = [a for a in P[(keys[0], seeds[0])] if a not in ("y", "pid", "k_flag")]
meths = [m for m in METHODS if m in arms and m + "_random" in arms]
rows = []
for k in keys:
    for s in seeds:
        Z = P[(k, s)]
        for a in arms: rows.append(dict(group=k, seed=s, arm=a, auroc=fast_auc(Z["y"], Z[a]), unflagged=missed_at_k(Z["y"], Z[a], Z["k_flag"].item())))
S = pd.DataFrame(rows).groupby(["group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="group", columns="arm", values="auroc"); U = S.pivot_table(index="group", columns="arm", values="unflagged")
print("\nmean over groups, AUROC and events unflagged per 1,000 at the matched flag rate"); print(pd.concat([W.mean().rename("auroc"), U.mean().rename("unflagged")], axis=1).round(4).to_string())
def stats(k, idx):
    o = {}
    for a in arms:
        o[a] = (np.nanmean([fast_auc(P[(k, s)]["y"][idx], P[(k, s)][a][idx]) for s in seeds]),
                np.mean([missed_at_k(P[(k, s)]["y"][idx], P[(k, s)][a][idx], max(1, int(round(int(P[(k, s)]["k_flag"]) * len(idx) / len(P[(k, s)]["y"]))))) for s in seeds]))
    return o
def contrasts(st):
    d = {}
    for m in meths:
        d[f"AUROC | {m} group minus {m} random"] = np.mean([o[m][0] - o[m + "_random"][0] for o in st])
        d[f"unflagged per 1,000 | {m} group minus {m} random"] = np.mean([o[m][1] - o[m + "_random"][1] for o in st])
        d[f"unflagged per 1,000 | {m} group minus original"] = np.mean([o[m][1] - o["original"][1] for o in st])
    return d
pt = contrasts([stats(k, np.arange(len(P[(k, seeds[0])]["y"]))) for k in keys]); pat = {}
for k in keys:
    u, inv = np.unique(P[(k, seeds[0])]["pid"], return_inverse=True); pat[k] = [np.where(inv == i)[0] for i in range(len(u))]
r = np.random.RandomState(0); boots = {k: [] for k in pt}
for b in range(CFG["n_boot"]):
    st = [stats(k, np.concatenate([pat[k][j] for j in r.choice(len(pat[k]), len(pat[k]), True)])) for k in r.choice(keys, len(keys), True)]
    for kk, v in contrasts(st).items(): boots[kk].append(v)
    if (b + 1) % 200 == 0: log(f"  bootstrap draw {b + 1}")
res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(keys), n_boot=CFG["n_boot"]) for k in pt}
lw = {m: dict(auroc=int(sum(W.loc[k, m] < W.loc[k, m + "_random"] for k in keys)), unflagged=int(sum(U.loc[k, m] > U.loc[k, m + "_random"] for k in keys))) for m in meths}
print(f"\n== {len(keys)} clustered groups, two-stage bootstrap ({CFG['n_boot']} draws)")
for k, v in res.items(): print(f"  {k:70s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
print("groups worse than the matched random removal (AUROC lower / more unflagged):", lw)
with open(f"{OUT}/intervals.json", "w") as f: json.dump(dict(results=res, worse_in=lw, date=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=CFG), f, indent=1)
log(f"done, results in {OUT}")
