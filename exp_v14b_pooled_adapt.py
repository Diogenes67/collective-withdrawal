"""exp_v14b: the deployable adaptation comparator (review point 4).
Every shared model gets the same community component, fitted on the community's TRAINING admissions (as in v11 and v14 Part B):
  pooled + component            the pooled model (which already contains those admissions) plus a residual component on them
  single base + component       exact retrain without the community plus the same component (confinement, single exclusion)
  common base + component       read from v14.1 Part B for the two five-community sets
Also the representation head for pooled and single base. All 25 communities x 3 seeds; no retraining (v5 checkpoints, v14.1 bases).
Writes results/v14b_pooled_adapt/{runs/*.json, summary.csv, paired.json}.
"""
import os, json, time, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
V5 = f"{ROOT}/results/v5_partitions"; V14 = f"{ROOT}/results/v14_commonbase_1" + ("_quick" if QUICK else "")
OUT = f"{ROOT}/results/v14b_pooled_adapt" + ("_quick" if QUICK else "")
os.makedirs(f"{OUT}/runs", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 200)),
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0, version="v14b.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")
# ---------- data, identical to exp_v5 / v11
COHORT = f"{ROOT}/derived/cohort_latest.parquet"
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(COHORT); feat = [c for c in df.columns if c not in META]; X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd; test_idx = np.where(is_test)[0]
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
groups = {}
for c in range(25):
    members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    ref = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
    if ref["n_train_patients"] != len(set(sid[f_tr])) or ref["n_test_patients"] != len(set(sid[f_te])): raise SystemExit(f"c{c:02d}: membership differs from v5")
    groups[f"c{c:02d}"] = dict(f_tr=f_tr, f_te=f_te, n_tr_p=len(set(sid[f_tr])))
log("25 communities match the v5 records")
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train(Xtr, ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed); m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(Xtr), to_t(ytr); n, B = len(Xtr), CFG["batch"]
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B): idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m
def load(path):
    if not os.path.exists(path): raise SystemExit(f"missing checkpoint {path}")
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def cached(key, fn):
    p = f"{OUT}/ckpt/{key}.pt"
    if os.path.exists(p): return load(p)
    m = fn(); torch.save(m.state_dict(), p + ".tmp"); os.replace(p + ".tmp", p); return m
def batched(fn, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(fn(to_t(Xa[i:i + 65536])).cpu().numpy())
    return np.concatenate(out)
def logits(m, Xa): return batched(lambda t: m(t), Xa).astype(np.float64)
def penult(m, Xa): return batched(lambda t: m.net[:4](t), Xa).astype(np.float32)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
class Platt:
    """Fallback adapter: logistic on the first column only (the shared logit); monotone, so it leaves the ranking unchanged."""
    def __init__(self, Z, yy): self.m = LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, :1], yy)
    def predict_proba(self, Z): return self.m.predict_proba(Z[:, :1])
def fit_adapter(Z, yy, pid, seed, offset=False):
    """Ridge logistic adapter with C chosen by patient-grouped CV log loss. With offset=True the first column is the shared logit:
    it is scaled by logit_scale so the L2 penalty barely touches its coefficient, and a logit-only (Platt) candidate competes in the CV,
    so the adapted model cannot be chosen when it does worse in CV than leaving the shared ranking alone."""
    Z = Z.copy()
    if offset: Z[:, 0] *= CFG["logit_scale"]
    n_p = len(np.unique(pid)); grid = CFG["adapter_C_grid"]; chosen, cv = "platt" if offset else grid[len(grid) // 2], None
    if n_p >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pid)); cv = {}
        cands = (["platt"] if offset else []) + list(grid)
        for C in cands:
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = Platt(Z[tr], yy[tr]) if C == "platt" else LogisticRegression(C=C, max_iter=3000).fit(Z[tr], yy[tr])
                ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[C] = float(np.mean(ll))
        if cv: chosen = min(cv, key=cv.get)
    m = Platt(Z, yy) if chosen == "platt" else LogisticRegression(C=chosen, max_iter=3000).fit(Z, yy)
    class Wrapped:
        def predict_proba(self, Zn):
            Zn = Zn.copy()
            if offset: Zn[:, 0] *= CFG["logit_scale"]
            return m.predict_proba(Zn)
    return Wrapped(), chosen
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def boot_auc(yy, pp, pid):
    r = np.random.RandomState(1); ups = np.unique(pid); gi = {p: np.where(pid == p)[0] for p in ups}; vals = []
    for _ in range(CFG["n_boot"]): i = np.concatenate([gi[p] for p in r.choice(ups, len(ups), True)]); vals.append(auc(yy[i], pp[i]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
def paired(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 2: return dict(d=float(d.mean()) if len(d) else float("nan"), ci=[float("nan")] * 2, frac_worse=float("nan"), n=int(len(d)))
    r = np.random.RandomState(0); bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
    return dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), n=int(len(d)))

originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt") for s in CFG["seeds"]}
for n in sorted(groups):
    g = groups[n]; te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{n}__s{s}.json"
        if os.path.exists(path): continue
        tg = time.time(); orig = originals[s]; single = load(f"{V5}/ckpt/exact_community_trainfit_{n}_s{s}.pt"); arms, Cs = {}, {}
        for lab, m in [("pooled", orig), ("single_base", single)]:
            arms[lab] = sigmoid(logits(m, X[te]))
            ad, C = fit_adapter(np.column_stack([logits(m, X[tr]), X[tr]]), y[tr], sid[tr], s, offset=True); arms[lab + "_resid"] = ad.predict_proba(np.column_stack([logits(m, X[te]), X[te]]))[:, 1]; Cs[lab + "_resid"] = C
            ad, C = fit_adapter(penult(m, X[tr]), y[tr], sid[tr], s); arms[lab + "_rep"] = ad.predict_proba(penult(m, X[te]))[:, 1]; Cs[lab + "_rep"] = C
        res = {a: dict(auroc=auc(yy, p), auroc_ci=boot_auc(yy, p, pid), ece=ece(yy, p)) for a, p in arms.items()}
        save_json(path, dict(community=n, seed=s, n_train_patients=int(g["n_tr_p"]), n_test_patients=int(len(np.unique(pid))), adapter_C=Cs, arms=res, cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg))
        log(f"{n} s{s}: pooled {res['pooled']['auroc']:.4f} +comp {res['pooled_resid']['auroc']:.4f} | single {res['single_base']['auroc']:.4f} +comp {res['single_base_resid']['auroc']:.4f} ({time.time()-tg:.0f}s)")

recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(community=r["community"], seed=r["seed"], arm=a, auroc=m["auroc"], ece=m["ece"]) for r in recs for a, m in r["arms"].items()]
S = pd.DataFrame(rows).groupby(["community", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="community", columns="arm", values="auroc"); E = S.pivot_table(index="community", columns="arm", values="ece"); out = {}
print(f"\n== Every shared model plus the same component fitted on the community's training admissions, n={len(W)} communities, seed means")
print(W[["pooled", "pooled_resid", "pooled_rep", "single_base", "single_base_resid", "single_base_rep"]].round(4).to_string())
for a, b, lab in [("pooled_resid", "pooled", "pooled+component minus pooled (what the component adds to the pooled model)"),
                  ("single_base_resid", "single_base", "base+component minus base"),
                  ("single_base_resid", "pooled_resid", "base+component minus pooled+component (price of exclusion, both with components)"),
                  ("single_base_rep", "pooled_rep", "same with the representation head"),
                  ("single_base", "pooled", "base minus pooled (unadapted)"),
                  ("single_base_resid", "pooled", "base+component minus pooled (as in v11)")]:
    pa, pe = paired(W[a] - W[b]), paired(E[a] - E[b]); out[f"{a}_vs_{b}"] = dict(auroc=pa, ece=pe)
    print(f"  {lab:88s} dAUROC {pa['d']:+.4f} [{pa['ci'][0]:+.4f},{pa['ci'][1]:+.4f}] worse {pa['frac_worse']:.0%}  dECE {pe['d']:+.4f}")
sets = json.load(open(f"{V14}/sets_B.json")) if os.path.exists(f"{V14}/sets_B.json") else {}
for sname, members in sets.items():
    rb = [json.load(open(f)) for f in sorted(glob.glob(f"{V14}/runs_B/{sname}__s*.json"))]
    if not rb: continue
    cb = pd.DataFrame([dict(community=n, seed=r["seed"], common_base_resid=r["communities"][n]["common_base_resid"]["auroc"], common_base=r["communities"][n]["common_base"]["auroc"]) for r in rb for n in members]).groupby("community").mean(numeric_only=True)
    T = pd.concat([W.loc[members, ["pooled", "pooled_resid", "single_base_resid"]], cb[["common_base", "common_base_resid"]]], axis=1)
    print(f"\n== set '{sname}' (v14.1 Part B common base joined)"); print(T.round(4).to_string())
    for a, b, lab in [("common_base_resid", "pooled_resid", "common base+component minus pooled+component"), ("single_base_resid", "pooled_resid", "single base+component minus pooled+component")]:
        pa = paired(T[a] - T[b]); out[f"{sname}|{a}_vs_{b}"] = pa
        print(f"    {lab:70s} {pa['d']:+.4f} [{pa['ci'][0]:+.4f},{pa['ci'][1]:+.4f}] worse {pa['frac_worse']:.0%} (n={pa['n']})")
save_json(f"{OUT}/paired.json", out); log(f"done, results in {OUT}")
