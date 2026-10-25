"""
exp_v32_pooled_full_adapt.py. The full pooled-adaptation comparator (review of v7, item 2). The pooled model, which was trained
with the group's data, is given a component fitted on the group's whole training partition: the residual component on its
logits, the representation head on its penultimate layer, or a fine-tuned copy, each fitted exactly as the confined system's
component is (v19 on MIMIC-IV, v20 on eICU) but with the pooled model as the base. The type is chosen by log loss on a fifth of
the group's training patients held back, then refitted on all of them; evaluation is on the group's held-out patients, which no
model saw. This is the adaptation a deployer could run on the pooled model itself; its fitting and selection data are
in-sample for the pooled model, which is why the main analysis cross-fits (Note 5). The cached v24 (MIMIC-IV) and v25 (eICU)
predictions of the adapted pooled comparator and the confined system are carried into the same bootstrap.

Env: ROOT (default ~/mimic), PART=mimic|eicu|both, SEEDS (MIMIC-IV seeds, default 0,1,2; eICU always 0,1,2), DEVICE, QUICK=1
(two groups per dataset, seed 0, two fine-tuning epochs, 30 bootstrap draws), NBOOT. Resumable: finished runs are skipped.
Writes results/v32_pooled_full_adapt/{runs, preds, summary_mimic.csv, summary_eicu.csv, intervals.json, summary.txt, log.txt}.
"""
import os, json, time, glob, copy, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; PART = os.environ.get("PART", "both")
SEEDS = [0] if QUICK else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]; ESEEDS = [0] if QUICK else [0, 1, 2]
OUT = f"{R}/v32_pooled_full_adapt" + ("_quick" if QUICK else "")
for d in ["runs", "preds"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           full_lr=1e-4, full_batch=256, full_max_epochs=2 if QUICK else 30, full_patience=5, full_val_frac=0.2, select_val_frac=0.2,
           n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 1000)), version="v32.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; quick={QUICK}; part {PART}; MIMIC seeds {SEEDS}; eICU seeds {ESEEDS}; out {OUT}")

# ---------------------------------------------------------------- model, components and selection, as v19 (MIMIC-IV) and v20 (eICU)
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def load(path, d):
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); m.eval(); return m
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
    return predict
def finetune_copy(base, X, y, pid, rows, seed):
    """Fine-tune a copy of the base on rows, early-stopped on a fifth of their patients (v19 and v20 recipe). Returns the model and epochs kept."""
    r = np.random.RandomState(seed); ups = np.unique(pid[rows]); r.shuffle(ups); val_p = set(ups[: max(1, int(CFG["full_val_frac"] * len(ups)))])
    va = np.isin(pid[rows], list(val_p)); tr_ = ~va
    Xtr, ytr, Xva, yva = X[rows][tr_], y[rows][tr_], X[rows][va], y[rows][va]
    torch.manual_seed(seed); m = copy.deepcopy(base).to(DEVICE); m.train(); opt = torch.optim.Adam(m.parameters(), lr=CFG["full_lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt, Xv, yv = to_t(Xtr), to_t(ytr), to_t(Xva), to_t(yva); n, B = len(Xtr), CFG["full_batch"]
    def vl():
        m.eval()
        with torch.no_grad(): v = float(bce(m(Xv), yv))
        m.train(); return v
    best, best_state, bad, kept = vl(), copy.deepcopy(m.state_dict()), 0, 0
    for ep in range(1, CFG["full_max_epochs"] + 1):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B): idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        v = vl()
        if v < best - 1e-5: best, best_state, bad, kept = v, copy.deepcopy(m.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= CFG["full_patience"]: break
    m.load_state_dict(best_state); m.eval(); return m, kept
KINDS = ["residual", "head", "finetuned"]
def fit_component(kind, base, X, y, pid, rows, seed):
    """Returns (predict(rows) -> probabilities, detail) for the component type fitted on rows with the given base."""
    if kind == "residual":
        f = fit_adapter(np.column_stack([logits(base, X[rows]), X[rows]]), y[rows], pid[rows], offset=True)
        return (lambda rr: f(np.column_stack([logits(base, X[rr]), X[rr]]))), None
    if kind == "head":
        f = fit_adapter(penult(base, X[rows]), y[rows], pid[rows]); return (lambda rr: f(penult(base, X[rr]))), None
    m, kept = finetune_copy(base, X, y, pid, rows, seed); return (lambda rr: sigmoid(logits(m, X[rr]))), kept
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def ece(yy, pp, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (pp >= edges[i]) & (pp < edges[i + 1] if i < bins - 1 else pp <= edges[i + 1])
        if m.any(): e += m.mean() * abs(yy[m].mean() - pp[m].mean())
    return float(e)
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
FULL = ["full_residual", "full_head", "full_finetuned", "full_chosen"]

def adapt(label, gi, s, pooled, X, y, pid, tr, te, carried, k_flag, path, extra):
    """One group and seed: select the type on a held-back fifth of the group's training patients, refit on all, predict held-out."""
    ts = time.time()
    r = np.random.RandomState(4000 + gi + 31 * s); ups = np.unique(pid[tr]); r.shuffle(ups); sel_p = set(ups[: max(2, int(CFG["select_val_frac"] * len(ups)))])
    sel = tr[np.isin(pid[tr], list(sel_p))]; fit = tr[~np.isin(pid[tr], list(sel_p))]
    val_ll, val_auc = {}, {}
    for k in KINDS:
        f, _ = fit_component(k, pooled, X, y, pid, fit, s); pr = f(sel)
        val_ll[k] = float(log_loss(y[sel], np.clip(pr, 1e-6, 1 - 1e-6), labels=[0, 1])); val_auc[k] = auc(y[sel], pr)
    p_sel = sigmoid(logits(pooled, X[sel])); pooled_ll = float(log_loss(y[sel], np.clip(p_sel, 1e-6, 1 - 1e-6), labels=[0, 1]))
    chosen = min(val_ll, key=val_ll.get)
    p = dict(carried); kept = {}
    for k in KINDS:
        f, d = fit_component(k, pooled, X, y, pid, tr, s); p[f"full_{k}"] = f(te)
        if d is not None: kept[k] = d
    p["full_chosen"] = p[f"full_{chosen}"]
    yy = y[te]; dmax = float(np.abs(p["pooled"] - sigmoid(logits(pooled, X[te]))).max())
    if dmax > 1e-4: log(f"  WARNING {label} s{s}: cached pooled predictions differ from the pooled model by up to {dmax:.2e}")
    np.savez_compressed(f"{OUT}/preds/{path}.npz", y=yy, pid=pid[te], k_flag=k_flag, **extra, **p)
    arms = {a: dict(auroc=auc(yy, p[a]), ece=ece(yy, p[a])) for a in p}
    rec = dict(group=label, seed=s, n_train_patients=int(len(ups)), n_train_rows=int(len(tr)), n_test_rows=int(len(te)),
               selection=dict(n_fit_patients=int(len(ups) - len(sel_p)), n_select_patients=len(sel_p), val_logloss=val_ll, val_auroc=val_auc, pooled_val_logloss=pooled_ll,
                              pooled_val_auroc=auc(y[sel], p_sel), chosen=chosen), finetune_epochs_kept=kept,
               in_sample=dict(pooled_auroc_train_rows=auc(y[tr], sigmoid(logits(pooled, X[tr]))), pooled_auroc_held_out=arms["pooled"]["auroc"]),
               arms=arms, cfg=CFG, device=DEVICE, torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
    save_json(f"{OUT}/runs/{path}.json", rec)
    log(f"  {label} s{s}: chose {chosen} (val logloss " + ", ".join(f"{k} {v:.4f}" for k, v in val_ll.items()) + f"; pooled {pooled_ll:.4f}; fine-tune epochs kept {kept.get('finetuned')}) | held-out AUROC pooled {arms['pooled']['auroc']:.4f} "
        + " ".join(f"{a} {arms[a]['auroc']:.4f}" for a in FULL) + f" | adapted cf {arms['pooled_cf_chosen']['auroc']:.4f} confined {arms['confined_chosen']['auroc']:.4f} ({time.time()-ts:.0f}s)")

# ---------------------------------------------------------------- MIMIC-IV
def run_mimic():
    V5 = f"{R}/v5_partitions"; V24 = f"{R}/v24_symmetric_mimic"
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
    X = df[feat].astype(float)
    for c in [c for c in feat if c.startswith("lab_")]: X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
    X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    groups = []
    for c in range(25):
        members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
        if len(set(sid[f_tr])) < 300 or len(set(sid[f_te])) < 50: continue
        groups.append(dict(name=f"c{c:02d}", f_tr=f_tr, f_te=f_te))
    if QUICK: groups = groups[:2]
    log(f"MIMIC-IV: {len(df)} admissions, {X.shape[1]} features, {len(groups)} clustered groups")
    pooled = {s: load(f"{V5}/ckpt/orig_s{s}.pt", X.shape[1]) for s in SEEDS}
    for gi, g in enumerate(groups):
        tag = f"community_trainfit__{g['name']}"; key = f"community_trainfit/{g['name']}"; tr = np.where(g["f_tr"])[0]; te = np.where(g["f_te"])[0]
        for s in SEEDS:
            if os.path.exists(f"{OUT}/runs/{tag}__s{s}.json"): continue
            Z = np.load(f"{V24}/preds/{tag}__s{s}.npz"); assert np.array_equal(Z["pid"], sid[te]) and np.array_equal(Z["y"], y[te]), "v24 held-out order differs"
            carried = {k: Z[k] for k in ["pooled", "pooled_cf_residual", "pooled_cf_head", "pooled_cf_finetuned", "pooled_cf_chosen", "confined_chosen"]}
            adapt(key, gi, s, pooled[s], X, y, sid, tr, te, carried, int(Z["k_flag"]), f"{tag}__s{s}", {})

# ---------------------------------------------------------------- eICU
def run_eicu():
    ORIG = f"{R}/model_cache_eicu_es"; V25 = f"{R}/v25_symmetric_eicu"
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
    META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
    feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
    for c in Xdf.columns[Xdf.isna().any()]: Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32); pid = np.asarray(df.uniquepid.values, dtype=object); hosp_id = df.hospitalid.values
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    names = np.unique(pid); code = np.searchsorted(names, pid)  # integer patient codes with the strings' sort order (as v31)
    hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); groups = []
    for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
        for v in sorted(hattr[col].dropna().unique(), key=str):
            hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
            if f_tr.sum() < 800 or f_te.sum() < 100: continue
            groups.append(dict(key=f"{fam}_{str(v).replace(' ', '_').replace('/', '_')}", f_tr=f_tr, f_te=f_te))
    if QUICK: groups = groups[:2]
    log(f"eICU: {len(df)} stays, {X.shape[1]} features, {len(groups)} consortia")
    pooled = {s: load(f"{ORIG}/orig_s{s}.pt", X.shape[1]) for s in ESEEDS}
    for gi, g in enumerate(groups):
        tr = np.where(g["f_tr"])[0]; te = np.where(g["f_te"])[0]
        for s in ESEEDS:
            if os.path.exists(f"{OUT}/runs/{g['key']}__s{s}.json"): continue
            Z = np.load(f"{V25}/preds/{g['key']}__s{s}.npz"); assert np.array_equal(Z["y"], y[te]) and np.array_equal(Z["hosp"], hosp_id[te]), "v25 held-out order differs"
            carried = {k: Z[k] for k in ["pooled", "pooled_cf_residual", "pooled_cf_head", "pooled_cf_finetuned", "pooled_cf_chosen", "confined_chosen", "confined_strict_chosen"] if k in Z.files}
            adapt(g["key"], gi, s, pooled[s], X, y, code, tr, te, carried, int(Z["k_flag"]), f"{g['key']}__s{s}", dict(hosp=Z["hosp"], landmark=Z["landmark"]))

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()

# ---------------------------------------------------------------- summaries and bootstraps
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); rk = np.empty(len(pv)); rk[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (rk[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
def load_preds(pattern):
    P = {}
    for f in glob.glob(pattern):
        Z = np.load(f, allow_pickle=True); g_, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(g_, int(s_))] = {k: Z[k] for k in Z.files}
    return P
def pairs_for(arms):
    d = {"full comparator (chosen) minus pooled": ("full_chosen", "pooled"), "pooled + full residual (in-sample) minus pooled": ("full_residual", "pooled"),
         "pooled + full head minus pooled": ("full_head", "pooled"), "pooled + full fine-tuned copy minus pooled": ("full_finetuned", "pooled"),
         "full comparator minus adapted (cross-fitted) pooled comparator": ("full_chosen", "pooled_cf_chosen"),
         "confined (chosen) minus full comparator": ("confined_chosen", "full_chosen"), "confined (chosen) minus pooled": ("confined_chosen", "pooled"),
         "confined (chosen) minus pooled + full fine-tuned copy (best full arm in hindsight)": ("confined_chosen", "full_finetuned"),
         "adapted (cross-fitted) pooled comparator minus pooled + full fine-tuned copy": ("pooled_cf_chosen", "full_finetuned")}
    if "confined_strict_chosen" in arms:
        d["confined strict (chosen) minus full comparator"] = ("confined_strict_chosen", "full_chosen"); d["confined strict (chosen) minus pooled + full fine-tuned copy"] = ("confined_strict_chosen", "full_finetuned")
    return d
S_LINES = [f"v32 full pooled-adaptation comparator ({time.strftime('%Y-%m-%d %H:%M')})"]; OUTJ = {}
def report(label, res, lw, n):
    S_LINES.append(f"== {label} ({n} groups, {CFG['n_boot']} draws)")
    for k, v in res.items(): S_LINES.append(f"  {k:88s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {n}" if k in lw else ""))

recs_m = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/community_trainfit__*.json"))]
if recs_m:
    S = pd.DataFrame([dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs_m for a, m in r["arms"].items()]).groupby(["group", "arm"]).mean(numeric_only=True).reset_index()
    S.to_csv(f"{OUT}/summary_mimic.csv", index=False); W = S.pivot_table(index="group", columns="arm", values="auroc")
    ch = pd.Series([r["selection"]["chosen"][0].upper() for r in sorted(recs_m, key=lambda r: (r["group"], r["seed"]))], index=[r["group"] for r in sorted(recs_m, key=lambda r: (r["group"], r["seed"]))]).groupby(level=0).agg("".join)
    ep = pd.Series([r["finetune_epochs_kept"].get("finetuned", 0) for r in recs_m], index=[r["group"] for r in recs_m]).groupby(level=0).mean()
    S_LINES.append("MIMIC-IV, AUROC on each group's held-out patients, seed means"); S_LINES.append(pd.concat([W[["pooled"] + FULL + ["pooled_cf_chosen", "confined_chosen"]].round(4), ch.rename("type"), ep.rename("ft epochs")], axis=1).to_string())
    S_LINES.append("choices: " + str(pd.Series([r["selection"]["chosen"] for r in recs_m]).value_counts().to_dict()) + f"; fine-tuning kept 0 epochs in {sum(r['finetune_epochs_kept'].get('finetuned', 0) == 0 for r in recs_m)} of {len(recs_m)} runs")
    ins = np.mean([r["in_sample"]["pooled_auroc_train_rows"] for r in recs_m]); S_LINES.append(f"pooled model AUROC on the group's training rows (in-sample) {ins:.4f}, on held-out {np.mean([r['in_sample']['pooled_auroc_held_out'] for r in recs_m]):.4f}")
    P = load_preds(f"{OUT}/preds/community_trainfit__*.npz"); seeds = sorted(set(s_ for _, s_ in P)); names = sorted(set(g_ for g_, _ in P if all((g_, s_) in P for s_ in seeds)))
    ARMS = sorted(set(k for Z in P.values() for k in Z) - {"y", "pid", "k_flag"}); pairs = pairs_for(ARMS)
    def stats(g_, idx):
        o = {}
        for a in ARMS:
            o[a] = (np.nanmean([fast_auc(P[(g_, s_)]["y"][idx], P[(g_, s_)][a][idx]) for s_ in seeds]),
                    np.mean([missed_at_k(P[(g_, s_)]["y"][idx], P[(g_, s_)][a][idx], max(1, int(round(int(P[(g_, s_)]["k_flag"]) * len(idx) / len(P[(g_, s_)]["y"]))))) for s_ in seeds]))
        return o
    def contrasts(st):
        d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
        d["events unflagged per 1,000 at matched flag rate, confined minus full comparator"] = np.mean([o["confined_chosen"][1] - o["full_chosen"][1] for o in st])
        d["events unflagged per 1,000 at matched flag rate, full comparator minus pooled"] = np.mean([o["full_chosen"][1] - o["pooled"][1] for o in st])
        d["events unflagged per 1,000 at matched flag rate, confined minus pooled + full fine-tuned copy"] = np.mean([o["confined_chosen"][1] - o["full_finetuned"][1] for o in st]); return d
    ISO = ["community_trainfit__c02", "community_trainfit__c05", "community_trainfit__c12", "community_trainfit__c15", "community_trainfit__c20"]
    def run_boot(subset, label):
        pt = contrasts([stats(g_, np.arange(len(P[(g_, seeds[0])]["y"]))) for g_ in subset]); pat = {}
        for g_ in subset:
            u, inv = np.unique(P[(g_, seeds[0])]["pid"], return_inverse=True); pat[g_] = [np.where(inv == i)[0] for i in range(len(u))]
        r = np.random.RandomState(0); boots = {k: [] for k in pt}
        for b in range(CFG["n_boot"]):
            st = [stats(g_, np.concatenate([pat[g_][j] for j in r.choice(len(pat[g_]), len(pat[g_]), True)])) for g_ in r.choice(subset, len(subset), True)]
            for k, v in contrasts(st).items(): boots[k].append(v)
            if (b + 1) % 200 == 0: log(f"  MIMIC-IV {label}: draw {b + 1}")
        res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(subset), n_boot=CFG["n_boot"]) for k in pt}
        lw = {k: int(sum((W.loc[g_.replace("__", "/"), a] - W.loc[g_.replace("__", "/"), b]) < 0 for g_ in subset)) for k, (a, b) in pairs.items()}
        report(f"MIMIC-IV {label}, two-stage bootstrap", res, lw, len(subset)); return dict(results=res, lower_in=lw)
    if names:
        OUTJ["mimic_all"] = run_boot(names, "all clustered groups"); iso = [g_ for g_ in names if g_ in ISO]
        if len(iso) >= 2: OUTJ["mimic_most_isolated"] = run_boot(iso, "five most isolated groups")

recs_e = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json")) if not os.path.basename(f).startswith("community_trainfit")]
if recs_e:
    S = pd.DataFrame([dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs_e for a, m in r["arms"].items()]).groupby(["group", "arm"]).mean(numeric_only=True).reset_index()
    S.to_csv(f"{OUT}/summary_eicu.csv", index=False); W = S.pivot_table(index="group", columns="arm", values="auroc")
    ch = pd.Series([r["selection"]["chosen"][0].upper() for r in sorted(recs_e, key=lambda r: (r["group"], r["seed"]))], index=[r["group"] for r in sorted(recs_e, key=lambda r: (r["group"], r["seed"]))]).groupby(level=0).agg("".join)
    ep = pd.Series([r["finetune_epochs_kept"].get("finetuned", 0) for r in recs_e], index=[r["group"] for r in recs_e]).groupby(level=0).mean()
    cols = ["pooled"] + FULL + ["pooled_cf_chosen", "confined_chosen"] + (["confined_strict_chosen"] if "confined_strict_chosen" in W.columns else [])
    S_LINES.append("eICU, AUROC on each consortium's held-out stays, seed means"); S_LINES.append(pd.concat([W[cols].round(4), ch.rename("type"), ep.rename("ft epochs")], axis=1).to_string())
    S_LINES.append("choices: " + str(pd.Series([r["selection"]["chosen"] for r in recs_e]).value_counts().to_dict()) + f"; fine-tuning kept 0 epochs in {sum(r['finetune_epochs_kept'].get('finetuned', 0) == 0 for r in recs_e)} of {len(recs_e)} runs")
    S_LINES.append(f"pooled model AUROC on the consortium's training rows (in-sample) {np.mean([r['in_sample']['pooled_auroc_train_rows'] for r in recs_e]):.4f}, on held-out {np.mean([r['in_sample']['pooled_auroc_held_out'] for r in recs_e]):.4f}")
    P = {k: v for k, v in load_preds(f"{OUT}/preds/*.npz").items() if not k[0].startswith("community_trainfit")}
    seeds = sorted(set(s_ for _, s_ in P)); keys = sorted(set(k_ for k_, _ in P if all((k_, s_) in P for s_ in seeds)))
    ARMS = sorted(set(k for Z in P.values() for k in Z) - {"y", "pid", "k_flag", "hosp", "landmark"}); pairs = pairs_for(ARMS)
    def run_boot_e(pop, label):
        hosp_all = np.unique(np.concatenate([P[(k, seeds[0])]["hosp"] for k in keys]))
        def stats(keep_h, weights):
            out = {}
            for k in keys:
                Z0 = P[(k, seeds[0])]
                if weights is not None: idx = np.concatenate([np.where(Z0["hosp"] == h)[0] for h in keep_h for _ in range(weights[h])])
                else: idx = np.where(np.isin(Z0["hosp"], keep_h))[0]
                if pop == "landmark": idx = idx[Z0["landmark"][idx].astype(bool)]
                if len(idx) < 50 or len(np.unique(Z0["y"][idx])) < 2: continue
                o = {}
                for a in ARMS:
                    o[a] = (np.nanmean([fast_auc(P[(k, s_)]["y"][idx], P[(k, s_)][a][idx]) for s_ in seeds]),
                            np.mean([missed_at_k(P[(k, s_)]["y"][idx], P[(k, s_)][a][idx], max(1, int(round(int(P[(k, s_)]["k_flag"]) * len(idx) / len(Z0["y"]))))) for s_ in seeds]))
                out[k] = o
            return out
        def contrasts(st):
            d = {k: np.mean([o[a][0] - o[b][0] for o in st.values()]) for k, (a, b) in pairs.items()}
            d["events unflagged per 1,000 at matched flag rate, confined minus full comparator"] = np.mean([o["confined_chosen"][1] - o["full_chosen"][1] for o in st.values()])
            d["events unflagged per 1,000 at matched flag rate, full comparator minus pooled"] = np.mean([o["full_chosen"][1] - o["pooled"][1] for o in st.values()])
            d["events unflagged per 1,000 at matched flag rate, confined minus pooled + full fine-tuned copy"] = np.mean([o["confined_chosen"][1] - o["full_finetuned"][1] for o in st.values()]); return d
        pt = contrasts(stats(hosp_all, None)); r = np.random.RandomState(0); boots = {k: [] for k in pt}
        for b in range(CFG["n_boot"]):
            drawn = r.choice(hosp_all, len(hosp_all), True); w = dict(zip(*np.unique(drawn, return_counts=True)))
            for k, v in contrasts(stats(np.array(list(w)), w)).items(): boots[k].append(v)
            if (b + 1) % 200 == 0: log(f"  eICU {pop}: draw {b + 1}")
        res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_boot=CFG["n_boot"]) for k in pt}
        lw = {k: int(sum((W.loc[c, a] - W.loc[c, b]) < 0 for c in keys)) for k, (a, b) in pairs.items()}
        report(f"eICU {label}, hospital-cluster bootstrap", res, lw, len(keys)); return dict(results=res, lower_in=lw)
    if keys: OUTJ["eicu_all"] = run_boot_e("all", "all held-out stays"); OUTJ["eicu_landmark"] = run_boot_e("landmark", "stays in the unit at 24 h")
save_json(f"{OUT}/intervals.json", dict(**OUTJ, cfg=CFG, date=time.strftime("%Y-%m-%d %H:%M:%S")))
open(f"{OUT}/summary.txt", "w").write("\n".join(S_LINES) + "\n"); print("\n" + "\n".join(S_LINES)); log(f"done, results in {OUT}")
