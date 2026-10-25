"""
exp_v30_seed_core.py. Additional training seeds for the core MIMIC-IV models: the pooled model, the exact retrain without each
clustered group, and the exact retrain without its patient-matched random set. Data pipeline, group construction, random sets
and training recipe are v5's, so the v16, v17, v19 and v24 chain and v29 can run with more seeds. Checkpoints go to
results/v5_partitions/ckpt under v5's names (orig_s{s}, exact_community_trainfit_c{cc}_s{s}, exactrp_community_trainfit_c{cc}_s{s});
an existing checkpoint is never retrained. For each group and seed a prediction cache (y, pid, original, base, random on the
group's held-out rows) and a JSON with AUROCs are written to results/v30_seed_core. The random set drawn for each group is
checked against the v5 run record (rows_random_pat and n_train_patients), so the control is the same patients v5 used.

Env: ROOT (default ~/mimic), SEEDS (default 3,4,5,6,7,8,9), DEVICE, QUICK=1 (two groups, two epochs, seed 3 only, and a
separate checkpoint folder so v5's cache is untouched).
"""
import os, json, time, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
SEEDS = [3] if QUICK else [int(x) for x in os.environ.get("SEEDS", "3,4,5,6,7,8,9").split(",")]
V5 = f"{ROOT}/results/v5_partitions"
OUT = f"{ROOT}/results/v30_seed_core" + ("_quick" if QUICK else "")
CKPT = f"{OUT}/ckpt" if QUICK else f"{V5}/ckpt"
for d in [f"{OUT}/runs", f"{OUT}/preds", CKPT]: os.makedirs(d, exist_ok=True)
CFG = dict(seeds=SEEDS, epochs=2 if QUICK else 12, lr=1e-3, batch=2048, k_communities=25, min_patients=300, version="v30.0", recipe="v5.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; quick={QUICK}; seeds {SEEDS}; checkpoints {CKPT}; out {OUT}")

# ---------------------------------------------------------------- data, split and scaling exactly as exp_v5_partitions.py
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
log(f"{len(df)} admissions, {len(pids)} patients, {X.shape[1]} features, train {is_train.sum()}, test {is_test.sum()}")

# ---------------------------------------------------------------- partition labels in v5's order, so each group's index (and random set) is v5's
def modal_label(col):
    cnt = df.groupby(["subject_id", col]).size().reset_index(name="n")
    cnt = cnt.sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id")
    return cnt.set_index("subject_id")[col]
labels = {}
race = modal_label("race_group")
for g in sorted(race.unique()): labels[("race", str(g))] = set(race.index[race == g])
if "english" in df.columns:
    eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
ins = modal_label("insurance")
if (ins == "Medicaid").any(): labels[("insurance", "medicaid")] = set(ins.index[ins == "Medicaid"])
ayg = modal_label("anchor_year_group")
for g in sorted(ayg.unique()): labels[("anchor_year", str(g).replace(" ", "").replace("-", "_"))] = set(ayg.index[ayg == g])
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
for c in range(CFG["k_communities"]): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
groups = []
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p = len(set(sid[f_tr])); n_te_p = len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p, n_te_p=n_te_p))
def patient_matched(g, r):
    tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]])))
    fp = r.choice(tr_p, min(g["n_tr_p"], len(tr_p)), False)
    return np.isin(sid, fp) & is_train
for i, g in enumerate(groups):
    if g["family"] == "community_trainfit": g["r_pat"] = patient_matched(g, np.random.RandomState(100 + i))
comm = [g for g in groups if g["family"] == "community_trainfit"]
for g in comm:
    ref = json.load(open(f"{V5}/runs/{g['family']}__{g['name']}__s0.json"))
    if ref["n_train_patients"] != g["n_tr_p"] or ref["n_test_patients"] != g["n_te_p"]: raise SystemExit(f"{g['name']}: membership differs from v5")
    if ref["rows_random_pat"] != int(g["r_pat"].sum()): raise SystemExit(f"{g['name']}: random set differs from v5 ({int(g['r_pat'].sum())} rows against {ref['rows_random_pat']})")
log(f"{len(groups)} groups in v5's order; 25 clustered groups and their random sets match the v5 records")
if QUICK: comm = comm[:2]

# ---------------------------------------------------------------- model and training recipe exactly as v5
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train(Xtr, ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(Xtr), to_t(ytr); n, B = len(Xtr), CFG["batch"]
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m
def cached(key, fn):
    path = f"{CKPT}/{key}.pt"
    if os.path.exists(path):
        m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m, False
    m = fn(); torch.save(m.state_dict(), path + ".tmp"); os.replace(path + ".tmp", path); return m, True
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    m.train(); return np.concatenate(out).astype(np.float64)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

# ---------------------------------------------------------------- runs
test_idx = np.where(is_test)[0]
for s in SEEDS:
    ts = time.time(); orig, new = cached(f"orig_s{s}", lambda: train(X[is_train], y[is_train], s))
    p_orig_full = predict(orig, X[is_test]); log(f"seed {s}: pooled model {'trained' if new else 'loaded'}, full-test AUROC {auc(y[is_test], p_orig_full):.4f} ({time.time()-ts:.0f}s)")
    for g in comm:
        path = f"{OUT}/runs/{g['family']}__{g['name']}__s{s}.json"
        if os.path.exists(path): continue
        tg = time.time(); key = f"{g['family']}_{g['name']}"
        r_tr_g, r_tr_p = is_train & ~g["f_tr"], is_train & ~g["r_pat"]
        base, nb = cached(f"exact_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s))
        rnd, nr = cached(f"exactrp_{key}_s{s}", lambda: train(X[r_tr_p], y[r_tr_p], s))
        te = np.where(g["f_te"])[0]; pos = np.searchsorted(test_idx, te)
        p = dict(original=p_orig_full[pos], base=predict(base, X[te]), random=predict(rnd, X[te]))
        np.savez_compressed(f"{OUT}/preds/{g['family']}__{g['name']}__s{s}.npz", y=y[te], pid=sid[te], **p)
        rec = dict(family=g["family"], group=g["name"], seed=s, n_train_patients=g["n_tr_p"], n_test_patients=g["n_te_p"], n_train_rows=int(g["f_tr"].sum()),
                   rows_random_pat=int(g["r_pat"].sum()), trained=dict(exact_group=nb, exact_random_pat=nr),
                   auroc={k: auc(y[te], v) for k, v in p.items()}, full_test_auroc=dict(original=auc(y[is_test], p_orig_full), exact_group=auc(y[is_test], predict(base, X[is_test])), exact_random_pat=auc(y[is_test], predict(rnd, X[is_test]))),
                   cfg=CFG, device=DEVICE, torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
        save_json(path, rec)
        log(f"  {g['name']} s{s}: group AUROC original {rec['auroc']['original']:.4f} exact {rec['auroc']['base']:.4f} random {rec['auroc']['random']:.4f} ({time.time()-tg:.0f}s)")
log(f"done, {len(SEEDS)} seeds x {len(comm)} groups; checkpoints in {CKPT}, caches in {OUT}/preds")
