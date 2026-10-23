"""
exp_v24_symmetric_mimic.py. A symmetric adapted pooled comparator (co-author question 1): the pooled model receives the same
choice among three component types as the confined system, chosen by validation, MIMIC-IV, 25 clustered groups x 3 seeds.

The pooled model's own predictions for its training patients are in-sample, so every pooled component is cross-fitted with the
v17 fold models (pooled model retrained without each of two patient-level halves of the group's training patients; cached in
results/v17_crossfit/ckpt). Residual: the v17 component (one adapter on out-of-fold logits for all of the group's training
patients, applied with the full pooled model's logits). Representation head and fine-tuned copy: fitted on half k with the fold-k
model, which never saw half k, applied to the held-out patients with the fold-k model, and averaged over the two folds.
Selection: within each half, 20 % of patients are held back; each type is fitted on the other 80 % with the fold model and scored
by log loss on the held-back patients, summed over both halves; the type with the lowest log loss is the pooled comparator's
choice. Held-out (test) patients play no part. Confined arms are read from v19 on the same patients (strict base with the
validation-chosen component). Two-stage bootstrap (groups, then patients; 1,000 draws).

Env: ROOT, QUICK=1, DEVICE, ONLY_GROUPS, NBOOT. Writes results/v24_symmetric_mimic/{runs, preds, summary.csv, intervals.json}.
"""
import os, json, time, glob, hashlib, copy, gc
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
OUT = f"{ROOT}/results/v24_symmetric_mimic" + ("_quick" if QUICK else ""); V17 = f"{ROOT}/results/v17_crossfit" + ("_quick" if QUICK else ""); V19 = f"{ROOT}/results/v19_protocol" + ("_quick" if QUICK else "")
for d in ["runs", "preds", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_top=100, min_patients=300,
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           n_boot=int(os.environ.get("NBOOT", 50 if QUICK else 1000)), n_folds=2, full_lr=1e-4, full_batch=256, full_max_epochs=2 if QUICK else 30, full_patience=5, full_val_frac=0.2, select_val_frac=0.2, version="v24.0")
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



def finetune_copy(base, rows, seed):
    r = np.random.RandomState(seed); u = np.unique(sid[rows]); r.shuffle(u); val_p = set(u[: max(1, int(CFG["full_val_frac"] * len(u)))])
    va = np.isin(sid[rows], list(val_p)); tr_ = ~va
    Xtr, ytr, Xva, yva = X[rows][tr_], y[rows][tr_], X[rows][va], y[rows][va]
    torch.manual_seed(seed); m = copy.deepcopy(base).to(DEVICE); m.train(); opt = torch.optim.Adam(m.parameters(), lr=CFG["full_lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt, Xv, yv = to_t(Xtr), to_t(ytr), to_t(Xva), to_t(yva); n, B = len(Xtr), CFG["full_batch"]
    def vl():
        m.eval()
        with torch.no_grad(): v = float(bce(m(Xv), yv))
        m.train(); return v
    best, best_state, bad = vl(), copy.deepcopy(m.state_dict()), 0
    for ep in range(1, CFG["full_max_epochs"] + 1):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B): idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        v = vl()
        if v < best - 1e-5: best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= CFG["full_patience"]: break
    m.load_state_dict(best_state); m.eval(); return m
def fit_type(kind, base, rows, seed):
    """Component of the given type fitted on rows with base; returns predict(rows) -> probabilities, applied with the same base."""
    if kind == "residual":
        f, _ = fit_adapter(np.column_stack([logits(base, X[rows]), X[rows]]), y[rows], sid[rows], offset=True)
        return lambda rr: f(np.column_stack([logits(base, X[rr]), X[rr]]))
    if kind == "head":
        f, _ = fit_adapter(penult(base, X[rows]), y[rows], sid[rows]); return lambda rr: f(penult(base, X[rr]))
    m = finetune_copy(base, rows, seed); return lambda rr: sigmoid(logits(m, X[rr]))
KINDS = ["residual", "head", "finetuned"]

# ---------------------------------------------------------------- runs
X = Xs
ARMS = ["pooled", "pooled_cf_residual", "pooled_cf_head", "pooled_cf_finetuned", "pooled_cf_chosen", "confined_chosen"]
for gi, g in enumerate(groups):
    tag = g["key"].replace("/", "__"); te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{tag}__s{s}.json"
        if os.path.exists(path): continue
        ts = time.time()
        Z17 = np.load(f"{V17}/preds/{tag}__s{s}.npz"); Z19 = np.load(f"{V19}/preds/{tag}__s{s}.npz")
        assert np.array_equal(Z17["pid"], pid) and np.array_equal(Z19["pid"], pid), "held-out order differs from v17/v19"
        best = None
        for gi_c in sorted({gi, 0}):
            r = np.random.RandomState(3000 + gi_c + 31 * s); ups = np.unique(sid[tr]); r.shuffle(ups); fl = np.array_split(ups, CFG["n_folds"]); gap = []
            for k, fp in enumerate(fl):
                m = load(f"{V17}/ckpt/pooled_minus_{tag}_f{k}_s{s}.pt", X.shape[1]); hold = np.isin(sid[tr], fp)
                gap.append(auc(y[tr[~hold]], logits(m, X[tr[~hold]])) - auc(y[tr[hold]], logits(m, X[tr[hold]])))
            if best is None or np.mean(gap) > best[1]: best = (gi_c, float(np.mean(gap)), fl)
        gi_fold, fold_gap, folds = best
        sel_pred = {k: [] for k in KINDS}; sel_y = []; te_pred = {k: [] for k in ["head", "finetuned"]}
        for k, fp in enumerate(folds):
            base_k = load(f"{V17}/ckpt/pooled_minus_{tag}_f{k}_s{s}.pt", X.shape[1])
            hold = tr[np.isin(sid[tr], fp)]
            rs = np.random.RandomState(5000 + gi + 31 * s + 7 * k); hp = np.unique(sid[hold]); rs.shuffle(hp); sp = set(hp[: max(2, int(CFG["select_val_frac"] * len(hp)))])
            sel = hold[np.isin(sid[hold], list(sp))]; fit = hold[~np.isin(sid[hold], list(sp))]; sel_y.append(y[sel])
            for kd in KINDS: sel_pred[kd].append(fit_type(kd, base_k, fit, s)(sel))
            for kd in ["head", "finetuned"]: te_pred[kd].append(fit_type(kd, base_k, hold, s)(te))
        ys = np.concatenate(sel_y); val_ll = {kd: float(log_loss(ys, np.clip(np.concatenate(sel_pred[kd]), 1e-6, 1 - 1e-6), labels=[0, 1])) for kd in KINDS}
        chosen = min(val_ll, key=val_ll.get)
        p = {"pooled": Z17["pooled"], "pooled_cf_residual": Z17["pooled_resid_crossfit"], "pooled_cf_head": np.mean(te_pred["head"], 0),
             "pooled_cf_finetuned": np.mean(te_pred["finetuned"], 0), "confined_chosen": Z19["chosen"]}
        p["pooled_cf_chosen"] = p[f"pooled_cf_{chosen}"]
        np.savez_compressed(f"{OUT}/preds/{tag}__s{s}.npz", y=yy, pid=pid, k_flag=int(Z19["k_flag"]), **p)
        rec = dict(group=g["key"], seed=s, selection=dict(val_logloss=val_ll, chosen=chosen), fold_seed_index=gi_fold, fold_gap=fold_gap, arms={a: dict(auroc=auc(yy, p[a]), ece=ece(yy, p[a])) for a in ARMS},
                   cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
        save_json(path, rec); a = rec["arms"]
        gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()
        log(f"{g['key']} s{s}: pooled chose {chosen} | pooled {a['pooled']['auroc']:.4f} cf-res {a['pooled_cf_residual']['auroc']:.4f} cf-head {a['pooled_cf_head']['auroc']:.4f} "
            f"cf-ft {a['pooled_cf_finetuned']['auroc']:.4f} | confined {a['confined_chosen']['auroc']:.4f} ({time.time()-ts:.0f}s)")

# ---------------------------------------------------------------- summary and two-stage bootstrap
if os.environ.get("SUMMARY", "1") == "0": raise SystemExit(0)
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
S = pd.DataFrame([dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs for a, m in r["arms"].items()]).groupby(["group", "arm"]).mean(numeric_only=True).reset_index()
S.to_csv(f"{OUT}/summary.csv", index=False); W = S.pivot_table(index="group", columns="arm", values="auroc")
print("\nAUROC on each group's held-out patients, seed means"); print(W[ARMS].round(4).to_string())
print("pooled comparator choices:", pd.Series([r["selection"]["chosen"] for r in recs]).value_counts().to_dict())
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); rk = np.empty(len(pv)); rk[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (rk[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); g_, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(g_.replace("__", "/"), int(s_))] = {k: Z[k] for k in Z.files}
seeds = sorted(set(s_ for _, s_ in P)); names = sorted(set(g_ for g_, _ in P if all((g_, s_) in P for s_ in seeds)))
pairs = {"confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)": ("confined_chosen", "pooled_cf_chosen"),
         "confined (chosen) minus pooled with cross-fitted residual (v17/v19 comparator)": ("confined_chosen", "pooled_cf_residual"),
         "pooled with chosen cross-fitted component minus pooled": ("pooled_cf_chosen", "pooled"),
         "pooled + cross-fitted residual minus pooled": ("pooled_cf_residual", "pooled"), "pooled + cross-fitted head minus pooled": ("pooled_cf_head", "pooled"),
         "pooled + cross-fitted fine-tuned copy minus pooled": ("pooled_cf_finetuned", "pooled")}
def stats(g_, idx):
    o = {}
    for a in ARMS:
        o[a] = (np.nanmean([fast_auc(P[(g_, s_)]["y"][idx], P[(g_, s_)][a][idx]) for s_ in seeds]),
                np.mean([missed_at_k(P[(g_, s_)]["y"][idx], P[(g_, s_)][a][idx], max(1, int(round(int(P[(g_, s_)]["k_flag"]) * len(idx) / len(P[(g_, s_)]["y"]))))) for s_ in seeds]))
    return o
def contrasts(st):
    d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
    d["events unflagged per 1,000 at matched flag rate, confined minus symmetric comparator"] = np.mean([o["confined_chosen"][1] - o["pooled_cf_chosen"][1] for o in st])
    return d
ISO = ["community_trainfit/c02", "community_trainfit/c05", "community_trainfit/c12", "community_trainfit/c15", "community_trainfit/c20"]
def run_boot(subset, label):
    pt = contrasts([stats(g_, np.arange(len(P[(g_, seeds[0])]["y"]))) for g_ in subset]); pat = {}
    for g_ in subset:
        u, inv = np.unique(P[(g_, seeds[0])]["pid"], return_inverse=True); pat[g_] = [np.where(inv == i)[0] for i in range(len(u))]
    r = np.random.RandomState(0); boots = {k: [] for k in pt}
    for b in range(CFG["n_boot"]):
        st = [stats(g_, np.concatenate([pat[g_][j] for j in r.choice(len(pat[g_]), len(pat[g_]), True)])) for g_ in r.choice(subset, len(subset), True)]
        for k, v in contrasts(st).items(): boots[k].append(v)
    res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(subset), n_boot=CFG["n_boot"]) for k in pt}
    lw = {k: int(sum((W.loc[g_, a] - W.loc[g_, b]) < 0 for g_ in subset)) for k, (a, b) in pairs.items()}
    print(f"\n== {label}: {len(subset)} groups, two-stage bootstrap ({CFG['n_boot']} draws)")
    for k, v in res.items(): print(f"  {k:95s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(subset)}" if k in lw else ""))
    return dict(results=res, lower_in=lw)
if names:
    out = {"all": run_boot(names, "all clustered groups")}
    iso = [g_ for g_ in names if g_ in ISO]
    if len(iso) >= 2: out["most_isolated"] = run_boot(iso, "five most isolated groups")
    save_json(f"{OUT}/intervals.json", dict(**out, choices=pd.Series([r["selection"]["chosen"] for r in recs]).value_counts().to_dict(), date=time.strftime("%Y-%m-%d %H:%M:%S")))
log(f"done, results in {OUT}")
