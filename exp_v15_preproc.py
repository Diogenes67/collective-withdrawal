"""
exp_v15_preproc.py. Does the shared model's preprocessing carry the community's influence? Sensitivity check on MIMIC-IV.

In every run so far the feature schema was fitted on the whole cohort: the 100 diagnosis-code indicators were the 100
most frequent 3-character codes over all admissions, imputation medians (where there are any) were computed over all
rows, and scaling used all training rows including the withdrawing community. Under confinement the weights exclude
the community but these three steps do not. This script refits all three on the retained training admissions only
(community excluded; test rows excluded), rebuilds the feature matrix, retrains the shared model with the v5 recipe,
fits the community's residual component on its training rows, and compares community held-out AUROC with the standard
pipeline (v5 exact retrain, v11 component) on the same patients. Five most isolated communities, three seeds.

Env: ROOT (default ~/mimic), QUICK=1, DEVICE, COMMUNITIES (comma list to override the five). Reads derived/cohort_latest.parquet,
hosp/diagnoses_icd.csv.gz, results/v5_partitions/{ckpt,runs,results.csv}. Writes results/v15_preproc/{runs/*.json, ckpt/, summary.csv, log}.
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
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); V5 = f"{ROOT}/results/v5_partitions"
OUT = f"{ROOT}/results/v15_preproc" + ("_quick" if QUICK else "")
for d in ["runs", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_top=100, n_communities=2 if QUICK else 5,
           adapter_C_grid=[0.01, 0.1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, version="v15.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
feat_dx = [c for c in feat if c.startswith("dx_")]; feat_other = [c for c in feat if not c.startswith("dx_")]; feat_lab = [c for c in feat if c.startswith("lab_")]
y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values; hadm = df.hadm_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test

def build_X(fit_rows, dx_long):
    """Feature matrix with vocabulary, imputation and scaling fitted on fit_rows only. Returns float32 array and the vocabulary."""
    top = dx_long[dx_long.hadm_id.isin(set(hadm[fit_rows]))].code3.value_counts().head(CFG["n_top"]).index
    wide = dx_long[dx_long.code3.isin(top)].assign(v=1).pivot_table(index="hadm_id", columns="code3", values="v", aggfunc="max", fill_value=0)
    wide = wide.reindex(columns=sorted(top), fill_value=0).reindex(hadm, fill_value=0)
    Xo = df[feat_other].astype(float).copy()
    for c in feat_lab:
        med = Xo.loc[fit_rows, c].median(); Xo[c + "_missing"] = Xo[c].isna().astype(float); Xo[c] = Xo[c].fillna(med)
    Xa = np.concatenate([Xo.values, wide.values.astype(float)], axis=1).astype(np.float32)
    mu, sd = Xa[fit_rows].mean(0), Xa[fit_rows].std(0) + 1e-6
    return (Xa - mu) / sd, list(top)
# standard pipeline, exactly as v5 (vocabulary = the parquet's dx_ columns; scaling on all training rows)
Xs = df[feat].astype(float)
for c in feat_lab: Xs[c + "_missing"] = Xs[c].isna().astype(float); Xs[c] = Xs[c].fillna(Xs[c].median())
Xs = Xs.values.astype(np.float32); mu, sd = Xs[is_train].mean(0), Xs[is_train].std(0) + 1e-6; Xs = (Xs - mu) / sd
pat_mean = pd.DataFrame(Xs).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
groups = {}
for c in range(25):
    members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    ref = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
    if ref["n_train_patients"] != len(set(sid[f_tr])) or ref["n_test_patients"] != len(set(sid[f_te])): raise SystemExit(f"c{c:02d}: membership differs from v5")
    groups[f"c{c:02d}"] = dict(f_tr=f_tr, f_te=f_te)
log("25 communities match the v5 records")
if os.environ.get("COMMUNITIES"): chosen = os.environ["COMMUNITIES"].split(",")
else:
    R = pd.read_csv(f"{V5}/results.csv"); R = R[(R.family == "community_trainfit") & (R.model == "original")]
    chosen = list(R.groupby("group").isolation_ratio.mean().sort_values(ascending=False).index[: CFG["n_communities"]])
log(f"communities: {chosen}")
log("loading diagnoses")
dx = pd.read_csv(f"{ROOT}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}, usecols=["hadm_id", "icd_code", "icd_version"])
dx = dx[dx.hadm_id.isin(set(hadm))]; dx["code3"] = dx.icd_version.astype(str) + "_" + dx.icd_code.str[:3]; dx = dx[["hadm_id", "code3"]]
global_top = [c[3:] for c in feat_dx]
log(f"{len(dx)} diagnosis rows; global vocabulary {len(global_top)} codes")

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
def load(path, d):
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def cached(key, d, fn):
    p = f"{OUT}/ckpt/{key}.pt"
    if os.path.exists(p): return load(p, d)
    m = fn(); torch.save(m.state_dict(), p + ".tmp"); os.replace(p + ".tmp", p); return m
def logits(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(m(to_t(Xa[i:i + 65536])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def fit_adapter(Z, yy, pid, seed):
    n_p = len(np.unique(pid)); grid = CFG["adapter_C_grid"]; chosen_C, cv = grid[len(grid) // 2], None
    if n_p >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pid)); cv = {}
        for C in grid:
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], yy[tr]); ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[C] = float(np.mean(ll))
        if cv: chosen_C = min(cv, key=cv.get)
    return LogisticRegression(C=chosen_C, max_iter=2000).fit(Z, yy)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)

for n in chosen:
    g = groups[n]; te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; fit_rows = is_train & ~g["f_tr"]; yy, pid = y[te], sid[te]
    Xr, vocab = build_X(fit_rows, dx); overlap = len(set(vocab) & set(global_top))
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{n}__s{s}.json"
        if os.path.exists(path): continue
        tg = time.time()
        std_base = load(f"{V5}/ckpt/exact_community_trainfit_{n}_s{s}.pt", Xs.shape[1]); strict_base = cached(f"strict_{n}_s{s}", Xr.shape[1], lambda: train(Xr[fit_rows], y[fit_rows], s))
        p_std, p_strict = sigmoid(logits(std_base, Xs[te])), sigmoid(logits(strict_base, Xr[te]))
        ad = fit_adapter(np.column_stack([logits(std_base, Xs[tr]), Xs[tr]]), y[tr], sid[tr], s); p_std_res = ad.predict_proba(np.column_stack([logits(std_base, Xs[te]), Xs[te]]))[:, 1]
        ad = fit_adapter(np.column_stack([logits(strict_base, Xr[tr]), Xr[tr]]), y[tr], sid[tr], s); p_strict_res = ad.predict_proba(np.column_stack([logits(strict_base, Xr[te]), Xr[te]]))[:, 1]
        others_te = np.where(is_test & ~g["f_te"])[0]
        rec = dict(community=n, seed=s, vocabulary_overlap_with_global=overlap, n_vocab=len(vocab), n_train_rows_fit=int(fit_rows.sum()),
                   arms=dict(standard_base=dict(auroc=auc(yy, p_std), ece=ece(yy, p_std), others_auroc=auc(y[others_te], sigmoid(logits(std_base, Xs[others_te])))),
                             strict_base=dict(auroc=auc(yy, p_strict), ece=ece(yy, p_strict), others_auroc=auc(y[others_te], sigmoid(logits(strict_base, Xr[others_te])))),
                             standard_base_resid=dict(auroc=auc(yy, p_std_res), ece=ece(yy, p_std_res)), strict_base_resid=dict(auroc=auc(yy, p_strict_res), ece=ece(yy, p_strict_res))),
                   cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
        with open(path + ".tmp", "w") as f: json.dump(rec, f, indent=1, default=float)
        os.replace(path + ".tmp", path)
        a = rec["arms"]; log(f"{n} s{s}: vocab overlap {overlap}/{len(vocab)}; base standard {a['standard_base']['auroc']:.4f} strict {a['strict_base']['auroc']:.4f}; "
                             f"+component standard {a['standard_base_resid']['auroc']:.4f} strict {a['strict_base_resid']['auroc']:.4f}; others {a['standard_base']['others_auroc']:.4f} vs {a['strict_base']['others_auroc']:.4f} ({time.time()-tg:.0f}s)")

recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(community=r["community"], seed=r["seed"], vocab_overlap=r["vocabulary_overlap_with_global"], arm=a, **{k: v for k, v in m.items()}) for r in recs for a, m in r["arms"].items()]
S = pd.DataFrame(rows).groupby(["community", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="community", columns="arm", values="auroc"); E = S.pivot_table(index="community", columns="arm", values="ece")
print("\nheld-out AUROC on the community, seed means: preprocessing fitted on all data (standard) vs on retained training rows only (strict)")
print(pd.concat([W[["standard_base", "strict_base", "standard_base_resid", "strict_base_resid"]].round(4), S[S.arm == "standard_base"].set_index("community")["vocab_overlap"].astype(int)], axis=1).to_string())
d1 = (W["strict_base"] - W["standard_base"]); d2 = (W["strict_base_resid"] - W["standard_base_resid"])
print(f"\nstrict minus standard, base: mean {d1.mean():+.4f} (range {d1.min():+.4f} to {d1.max():+.4f}); with component: mean {d2.mean():+.4f} (range {d2.min():+.4f} to {d2.max():+.4f})")
print(f"ECE strict minus standard, base: {(E['strict_base'] - E['standard_base']).mean():+.4f}; with component: {(E['strict_base_resid'] - E['standard_base_resid']).mean():+.4f}")
O = S.pivot_table(index="community", columns="arm", values="others_auroc"); print(f"others' AUROC, strict minus standard base: {(O['strict_base'] - O['standard_base']).mean():+.4f}")
log(f"done, results in {OUT}")
