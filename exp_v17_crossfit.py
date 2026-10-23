"""
exp_v17_crossfit.py. The fair pooled-plus-component comparator (review point 2), MIMIC-IV clustered groups.

In v14b the residual component fitted to the pooled model used the pooled model's logits on admissions it was trained on, an
in-sample residual, while the base's logits on the group's training admissions are out-of-sample by construction. Here the
pooled model's component is cross-fitted: the group's training patients are split into two patient-level halves; for each half
a pooled model is retrained on all training admissions except that half (everyone else plus the other half of the group), and
its logits on the held-out half are out-of-fold. The residual component is fitted on these out-of-fold logits and the features,
then applied, as in stacking, to the full pooled model's logits on the group's held-out patients. The base side is the v5 exact
retrain with the same component (its logits on the group are already out-of-sample). Same preprocessing as the pooled model
(whole-training), so the comparison isolates the in-sample asymmetry. The representation head is not cross-fitted because
penultimate representations of separately trained models are not aligned.

Arms on each group's held-out patients: pooled; pooled + in-sample component (v14b); pooled + cross-fitted component; base;
base + component. Two-stage bootstrap (groups, then patients; 1,000 draws) from cached predictions.

Env: ROOT (default ~/mimic), QUICK=1, DEVICE, ONLY_GROUPS, NBOOT. Reads derived/cohort_latest.parquet, results/v5_partitions/{ckpt,runs}.
Writes results/v17_crossfit/{runs/*.json, preds/*.npz, ckpt/, summary.csv, intervals.json}.
"""
import os, json, time, glob, hashlib
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
OUT = f"{ROOT}/results/v17_crossfit" + ("_quick" if QUICK else "")
for d in ["runs", "preds", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_top=100, min_patients=300,
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           n_boot=int(os.environ.get("NBOOT", 50 if QUICK else 1000)), n_folds=2, version="v17.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ---------------------------------------------------------------- data, split and groups exactly as v5/v11
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
feat_dx = [c for c in feat if c.startswith("dx_")]; feat_other = [c for c in feat if not c.startswith("dx_")]; feat_lab = [c for c in feat if c.startswith("lab_")]
y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values; hadm = df.hadm_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test; test_idx = np.where(is_test)[0]
# standard (whole-training) pipeline: used for the pooled model and for group construction only
Xs = df[feat].astype(float)
for c in feat_lab: Xs[c + "_missing"] = Xs[c].isna().astype(float); Xs[c] = Xs[c].fillna(Xs[c].median())
Xs = Xs.values.astype(np.float32); mu0, sd0 = Xs[is_train].mean(0), Xs[is_train].std(0) + 1e-6; Xs = (Xs - mu0) / sd0
def modal_label(col):
    cnt = df.groupby(["subject_id", col]).size().reset_index(name="n").sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id")
    return cnt.set_index("subject_id")[col]
labels = {}
pat_mean = pd.DataFrame(Xs).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
for c in range(25): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
groups = []
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    groups.append(dict(family=fam, name=name, key=f"{fam}/{name}", f_tr=f_tr, f_te=f_te, mem=mem, n_tr_p=n_tr_p, n_te_p=n_te_p))
order = {"community_trainfit": 0, "language": 1, "race": 2}; groups.sort(key=lambda g: (order[g["family"]], g["name"]))
if QUICK: groups = [g for g in groups if g["key"] in ("community_trainfit/c00", "community_trainfit/c03")]
only = os.environ.get("ONLY_GROUPS")
if only: groups = [g for g in groups if g["key"] in only.split(",")]
for g in groups:
    ref = json.load(open(f"{V5}/runs/{g['family']}__{g['name']}__s0.json"))
    if ref["n_train_patients"] != g["n_tr_p"] or ref["n_test_patients"] != g["n_te_p"]: raise SystemExit(f"{g['key']}: membership differs from v5")
log(f"{len(groups)} groups match the v5 records")
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
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read())
    return h.hexdigest()
def batched(fn, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(fn(to_t(Xa[i:i + 65536])).cpu().numpy())
    return np.concatenate(out)
def logits(m, Xa): return batched(lambda t: m(t), Xa).astype(np.float64)
def penult(m, Xa): return batched(lambda t: m.net[:4](t), Xa).astype(np.float32)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
class Platt:
    def __init__(self, Z, yy): self.m = LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, :1], yy)
    def predict_proba(self, Z): return self.m.predict_proba(Z[:, :1])
def fit_adapter(Z, yy, pid, offset=False):
    """v14.1 adapter: logit column scaled so its coefficient is effectively unpenalised; logit-only recalibration competes in CV."""
    Z = Z.copy()
    if offset: Z[:, 0] *= CFG["logit_scale"]
    grid = CFG["adapter_C_grid"]; chosen = "platt" if offset else grid[len(grid) // 2]
    if len(np.unique(pid)) >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pid)); cv = {}
        for C in (["platt"] if offset else []) + list(grid):
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = Platt(Z[tr], yy[tr]) if C == "platt" else LogisticRegression(C=C, max_iter=3000).fit(Z[tr], yy[tr])
                ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[str(C)] = float(np.mean(ll))
        if cv: k = min(cv, key=cv.get); chosen = "platt" if k == "platt" else float(k)
    m = Platt(Z, yy) if chosen == "platt" else LogisticRegression(C=chosen, max_iter=3000).fit(Z, yy)
    def predict(Zn):
        Zn = Zn.copy()
        if offset: Zn[:, 0] *= CFG["logit_scale"]
        return m.predict_proba(Zn)[:, 1]
    return predict, chosen
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- runs
X = Xs
originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt", X.shape[1]) for s in CFG["seeds"]}
thr_sys = {s: float(np.percentile(sigmoid(logits(originals[s], X[is_test])), 90)) for s in CFG["seeds"]}
ARMS = ["pooled", "pooled_resid_insample", "pooled_resid_crossfit", "base", "base_resid"]
for gi, g in enumerate(groups):
    tag = g["key"].replace("/", "__"); te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{tag}__s{s}.json"
        if os.path.exists(path): continue
        ts = time.time(); orig = originals[s]; base = load(f"{V5}/ckpt/exact_{tag.split('__')[0]}_{g['name']}_s{s}.pt", X.shape[1])
        r = np.random.RandomState(3000 + gi + 31 * s); ups = np.unique(sid[tr]); r.shuffle(ups); folds = np.array_split(ups, CFG["n_folds"])
        z_oof = np.zeros(len(tr))
        for k, fp in enumerate(folds):
            hold = np.isin(sid[tr], fp); drop_rows = np.zeros(len(X), bool); drop_rows[tr[hold]] = True
            fit_rows = is_train & ~drop_rows
            ck = f"{OUT}/ckpt/pooled_minus_{tag}_f{k}_s{s}.pt"
            if os.path.exists(ck): m = load(ck, X.shape[1])
            else: m = train(X[fit_rows], y[fit_rows], s); torch.save(m.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
            assert not np.isin(sid[fit_rows], fp).any()
            z_oof[hold] = logits(m, X[tr[hold]])
        z_full_tr = logits(orig, X[tr]); z_te = logits(orig, X[te]); zb_tr = logits(base, X[tr]); zb_te = logits(base, X[te])
        p = {"pooled": sigmoid(z_te), "base": sigmoid(zb_te)}
        f, C_in = fit_adapter(np.column_stack([z_full_tr, X[tr]]), y[tr], sid[tr], offset=True); p["pooled_resid_insample"] = f(np.column_stack([z_te, X[te]]))
        f, C_cf = fit_adapter(np.column_stack([z_oof, X[tr]]), y[tr], sid[tr], offset=True); p["pooled_resid_crossfit"] = f(np.column_stack([z_te, X[te]]))
        f, C_b = fit_adapter(np.column_stack([zb_tr, X[tr]]), y[tr], sid[tr], offset=True); p["base_resid"] = f(np.column_stack([zb_te, X[te]]))
        k_flag = int((p["pooled"] >= thr_sys[s]).sum())
        np.savez_compressed(f"{OUT}/preds/{tag}__s{s}.npz", y=yy, pid=pid, k_flag=k_flag, **p)
        rec = dict(group=g["key"], seed=s, n_train_patients=g["n_tr_p"], n_test_patients=g["n_te_p"], k_flag=k_flag,
                   arms={a: dict(auroc=auc(yy, p[a]), ece=ece(yy, p[a])) for a in ARMS},
                   logit_gap=dict(mean_insample_minus_oof=float(np.mean(z_full_tr - z_oof)), corr_insample_oof=float(np.corrcoef(z_full_tr, z_oof)[0, 1]),
                                  auroc_insample_on_train=auc(y[tr], z_full_tr), auroc_oof_on_train=auc(y[tr], z_oof)),
                   adapter_C=dict(insample=C_in, crossfit=C_cf, base=C_b), cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
        save_json(path, rec)
        a = rec["arms"]; log(f"{g['key']} s{s}: pooled {a['pooled']['auroc']:.4f} +in-sample {a['pooled_resid_insample']['auroc']:.4f} +cross-fitted {a['pooled_resid_crossfit']['auroc']:.4f} | "
                             f"base {a['base']['auroc']:.4f} +component {a['base_resid']['auroc']:.4f} | train AUROC in-sample {rec['logit_gap']['auroc_insample_on_train']:.3f} oof {rec['logit_gap']['auroc_oof_on_train']:.3f} ({time.time()-ts:.0f}s)")

# ---------------------------------------------------------------- summary from disk
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs for a, m in r["arms"].items()]
S = pd.DataFrame(rows).groupby(["group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="group", columns="arm", values="auroc")
print("\nAUROC on each group's held-out patients, seed means"); print(W[ARMS].round(4).to_string())
G = pd.DataFrame([dict(group=r["group"], **r["logit_gap"]) for r in recs]).groupby("group").mean()
print(f"\npooled model on the group's training admissions: AUROC in-sample {G.auroc_insample_on_train.mean():.4f} vs out-of-fold {G.auroc_oof_on_train.mean():.4f}; "
      f"mean logit gap {G.mean_insample_minus_oof.mean():+.3f}; correlation {G.corr_insample_oof.mean():.3f}")
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); g, s = os.path.basename(f)[:-4].rsplit("__s", 1); P[(g.replace("__", "/"), int(s))] = {k: Z[k] for k in Z.files}
seeds = sorted(set(s for _, s in P)); names = sorted(set(g for g, _ in P if all((g, s_) in P for s_ in seeds)))
pairs = {"base + component minus pooled + cross-fitted component": ("base_resid", "pooled_resid_crossfit"),
         "base + component minus pooled + in-sample component": ("base_resid", "pooled_resid_insample"),
         "pooled + cross-fitted component minus pooled": ("pooled_resid_crossfit", "pooled"),
         "pooled + in-sample component minus pooled": ("pooled_resid_insample", "pooled"),
         "base + component minus pooled": ("base_resid", "pooled"), "base minus pooled": ("base", "pooled")}
def stats(g, idx):
    o = {}
    for a in ARMS:
        o[a] = (np.nanmean([fast_auc(P[(g, s)]["y"][idx], P[(g, s)][a][idx]) for s in seeds]),
                np.mean([missed_at_k(P[(g, s)]["y"][idx], P[(g, s)][a][idx], max(1, int(round(int(P[(g, s)]["k_flag"]) * len(idx) / len(P[(g, s)]["y"]))))) for s in seeds]))
    return o
def contrasts(st):
    d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
    d["events unflagged per 1,000 at matched flag rate, base + component minus pooled + cross-fitted component"] = np.mean([o["base_resid"][1] - o["pooled_resid_crossfit"][1] for o in st])
    return d
if names:
    pt = contrasts([stats(g, np.arange(len(P[(g, seeds[0])]["y"]))) for g in names])
    pat_idx = {}
    for g in names:
        u, inv = np.unique(P[(g, seeds[0])]["pid"], return_inverse=True); pat_idx[g] = (u, [np.where(inv == i)[0] for i in range(len(u))])
    r = np.random.RandomState(0); boots = {k: [] for k in pt}
    for b in range(CFG["n_boot"]):
        st = []
        for g in r.choice(names, len(names), True):
            u, gidx = pat_idx[g]; idx = np.concatenate([gidx[j] for j in r.choice(len(u), len(u), True)]); st.append(stats(g, idx))
        for k, v in contrasts(st).items(): boots[k].append(v)
        if (b + 1) % 200 == 0: log(f"  bootstrap draw {b + 1}")
    res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(names), n_boot=CFG["n_boot"]) for k in pt}
    lw = {k: int(sum((W.loc[g, a] - W.loc[g, b]) < 0 for g in names)) for k, (a, b) in pairs.items()}
    print(f"\n== {len(names)} clustered groups, two-stage bootstrap ({CFG['n_boot']} draws)")
    for k, v in res.items(): print(f"  {k:90s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(names)}" if k in lw else ""))
    save_json(f"{OUT}/intervals.json", dict(results=res, lower_in=lw, date=time.strftime("%Y-%m-%d %H:%M:%S")))
log(f"done, results in {OUT}")
