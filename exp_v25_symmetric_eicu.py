"""
exp_v25_symmetric_eicu.py. Symmetric adapted pooled comparator on the centralised eICU model (co-author question 1), 10 consortia
x 3 seeds: the pooled model receives the same choice among residual, representation-head and fine-tuned components as the
confined system, chosen by validation, every pooled component cross-fitted with the v20 fold models (pooled model retrained
without each half of the consortium's training patients; cached in results/v20_eicu_protocol/ckpt). Residual: v20's component on
out-of-fold logits. Head and fine-tuned copy: fitted on half k with the fold-k model and averaged over folds. Selection: 20 % of
patients in each half held back, log loss summed over halves. Confined arms are read from v20 (original-pipeline base) and, when
present, v23 (strict pipeline). Hospital-cluster bootstrap (1,000 draws) over all held-out stays and the 24-hour landmark.

Env: ROOT, QUICK=1, DEVICE, SEED, IDX, SUMMARY=0, NBOOT. Writes results/v25_symmetric_eicu/{runs, preds, summary.csv, intervals.json}.
"""
import os, json, time, glob, copy, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); V72 = f"{ROOT}/results/v7_eicu_consortium_es"; ORIG = f"{ROOT}/results/model_cache_eicu_es"
V20 = f"{ROOT}/results/v20_eicu_protocol" + ("_quick" if QUICK else ""); V23 = f"{ROOT}/results/v23_eicu_strict" + ("_quick" if QUICK else ""); OUT = f"{ROOT}/results/v25_symmetric_eicu" + ("_quick" if QUICK else "")
for d in ["runs", "preds", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else ([int(os.environ["SEED"])] if os.environ.get("SEED") else [0, 1, 2]), epochs=2 if QUICK else 30, patience=3, val_frac=0.1, lr=1e-3, batch=2048, min_stays=800,
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           full_lr=1e-4, full_batch=256, full_max_epochs=2 if QUICK else 30, full_patience=5, full_val_frac=0.2, select_val_frac=0.2, n_folds=2,
           n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 1000)), version="v25.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ---------------------------------------------------------------- data, identical to v7.2
df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
for c in Xdf.columns[Xdf.isna().any()]: Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32); pid = df.uniquepid.values; hosp_id = df.hospitalid.values; stay = df.patientunitstayid.values
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
is_val = df.uniquepid.isin(set(tr_p[: int(CFG["val_frac"] * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val; test_idx = np.where(is_test)[0]
hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); groups = []
for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
    for v in sorted(hattr[col].dropna().unique(), key=str):
        hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
        if f_tr.sum() < CFG["min_stays"] or f_te.sum() < 100: continue
        groups.append(dict(family=fam, name=str(v).replace(" ", "_").replace("/", "_"), hs=hs, mem=mem, f_tr=f_tr, f_te=f_te, n_tr_p=int(len(np.unique(pid[f_tr])))))
if QUICK: groups = groups[:2]
only = os.environ.get("ONLY")
if only: groups = [g for g in groups if f"{g['family']}/{g['name']}" in only.split(",")]
log(f"{len(groups)} consortia")
pt_off = pd.read_csv(f"{ROOT}/eicu/patient.csv.gz", usecols=["patientunitstayid", "unitdischargeoffset"]).set_index("patientunitstayid").unitdischargeoffset
landmark = (pt_off.reindex(stay).values >= 1440)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train_es(Xtr, ytr, Xv, yv, seed):
    torch.manual_seed(seed); np.random.seed(seed); m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt, Xvt, yvt = to_t(Xtr), to_t(ytr), to_t(Xv), to_t(yv); n, B = len(Xtr), CFG["batch"]; best, best_state, bad = float("inf"), None, 0
    for ep in range(1, CFG["epochs"] + 1):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B): idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = float(bce(m(Xvt), yvt))
        m.train()
        if v < best - 1e-5: best, best_state, bad = v, {k: t.detach().clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    m.load_state_dict(best_state); m.eval(); return m
def load(path):
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
    def __init__(self, Z, yy): self.m = LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, :1], yy)
    def predict_proba(self, Z): return self.m.predict_proba(Z[:, :1])
def fit_adapter(Z, yy, pp, offset=False):
    Z = Z.copy()
    if offset: Z[:, 0] *= CFG["logit_scale"]
    grid = CFG["adapter_C_grid"]; chosen = "platt" if offset else grid[len(grid) // 2]
    if len(np.unique(pp)) >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pp)); cv = {}
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
def finetune_copy(base, rows, seed):
    r = np.random.RandomState(seed); u = np.unique(pid[rows]); r.shuffle(u); val_p = set(u[: max(1, int(CFG["full_val_frac"] * len(u)))])
    va = np.isin(pid[rows], list(val_p)); tr_ = ~va; Xtr, ytr, Xva, yva = X[rows][tr_], y[rows][tr_], X[rows][va], y[rows][va]
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
def fit_component(kind, base, rows, seed):
    if kind == "residual":
        f = fit_adapter(np.column_stack([logits(base, X[rows]), X[rows]]), y[rows], pid[rows], offset=True); return lambda rr: f(np.column_stack([logits(base, X[rr]), X[rr]]))
    if kind == "head":
        f = fit_adapter(penult(base, X[rows]), y[rows], pid[rows]); return lambda rr: f(penult(base, X[rr]))
    m = finetune_copy(base, rows, seed); return lambda rr: sigmoid(logits(m, X[rr]))
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
KINDS = ["residual", "head", "finetuned"]


TRUE_IDX = {f"{g['family']}_{g['name']}": i for i, g in enumerate(groups)}
if os.environ.get("IDX") is not None: groups = [groups[int(os.environ["IDX"])]]
def resolve_folds(key, tr, s):
    """v20 seeded the fold split with the consortium's index in the list it ran over: its true index in the single-process part of
    the run and 0 in the one-consortium-per-process part. Both candidates are tried; the right one is the split on which each fold
    model predicts its own held-out half worse than the half it was trained on."""
    best = None
    for gi_c in sorted({TRUE_IDX[key], 0}):
        r = np.random.RandomState(3000 + gi_c + 31 * s); u = np.unique(pid[tr]); r.shuffle(u); folds = np.array_split(u, CFG["n_folds"]); gap = []
        for k, fp in enumerate(folds):
            m = load(f"{V20}/ckpt/pooled_minus_{key}_f{k}_s{s}.pt"); hold = np.isin(pid[tr], fp)
            gap.append(auc(y[tr[~hold]], logits(m, X[tr[~hold]])) - auc(y[tr[hold]], logits(m, X[tr[hold]])))
        g_ = float(np.mean(gap)); log(f"  {key} s{s}: fold seed index {gi_c}, in-sample minus out-of-fold AUROC {g_:+.4f}")
        if best is None or g_ > best[1]: best = (gi_c, g_, folds)
    return best
def fit_type(kind, base, rows, seed):
    if kind == "residual":
        f = fit_adapter(np.column_stack([logits(base, X[rows]), X[rows]]), y[rows], pid[rows], offset=True); return lambda rr: f(np.column_stack([logits(base, X[rr]), X[rr]]))
    if kind == "head":
        f = fit_adapter(penult(base, X[rows]), y[rows], pid[rows]); return lambda rr: f(penult(base, X[rr]))
    m = finetune_copy(base, rows, seed); return lambda rr: sigmoid(logits(m, X[rr]))
ARMS = ["pooled", "pooled_cf_residual", "pooled_cf_head", "pooled_cf_finetuned", "pooled_cf_chosen", "confined_chosen", "confined_strict_chosen"]
# ---------------------------------------------------------------- runs
for gj, g in enumerate(groups):
    key = f"{g['family']}_{g['name']}"; gi = TRUE_IDX[key]; te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy = y[te]
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{key}__s{s}.json"
        if os.path.exists(path): continue
        ts = time.time(); Z20 = np.load(f"{V20}/preds/{key}__s{s}.npz"); assert np.array_equal(Z20["y"], yy), "v20 held-out order differs"
        f23 = f"{V23}/preds/{key}__s{s}.npz"; Z23 = np.load(f23) if os.path.exists(f23) else None
        gi_fold, fold_gap, folds = resolve_folds(key, tr, s)
        sel_pred = {k: [] for k in KINDS}; sel_y = []; te_pred = {k: [] for k in ["head", "finetuned"]}
        for k, fp in enumerate(folds):
            ck = f"{V20}/ckpt/pooled_minus_{key}_f{k}_s{s}.pt"; assert os.path.exists(ck), f"missing fold model {ck}"
            base_k = load(ck); hold = tr[np.isin(pid[tr], fp)]
            rs = np.random.RandomState(5000 + gi + 31 * s + 7 * k); hp = np.unique(pid[hold]); rs.shuffle(hp); sp = set(hp[: max(2, int(CFG["select_val_frac"] * len(hp)))])
            sel = hold[np.isin(pid[hold], list(sp))]; fit = hold[~np.isin(pid[hold], list(sp))]; sel_y.append(y[sel])
            for kd in KINDS: sel_pred[kd].append(fit_type(kd, base_k, fit, s)(sel))
            for kd in ["head", "finetuned"]: te_pred[kd].append(fit_type(kd, base_k, hold, s)(te))
            del base_k; gc.collect()
            if DEVICE == "mps": torch.mps.empty_cache()
        ys = np.concatenate(sel_y); val_ll = {kd: float(log_loss(ys, np.clip(np.concatenate(sel_pred[kd]), 1e-6, 1 - 1e-6), labels=[0, 1])) for kd in KINDS}
        chosen = min(val_ll, key=val_ll.get)
        p = {"pooled": Z20["pooled"], "pooled_cf_residual": Z20["pooled_cf"], "pooled_cf_head": np.mean(te_pred["head"], 0), "pooled_cf_finetuned": np.mean(te_pred["finetuned"], 0),
             "confined_chosen": Z20["chosen"], "confined_strict_chosen": (Z23["chosen"] if Z23 is not None else Z20["chosen"])}
        p["pooled_cf_chosen"] = p[f"pooled_cf_{chosen}"]
        np.savez_compressed(f"{OUT}/preds/{key}__s{s}.npz", y=yy, hosp=Z20["hosp"], landmark=Z20["landmark"], k_flag=int(Z20["k_flag"]), **p)
        rec = dict(consortium=key, seed=s, selection=dict(val_logloss=val_ll, chosen=chosen), strict_available=Z23 is not None, fold_seed_index=gi_fold, fold_gap=fold_gap,
                   arms={a: dict(auroc=auc(yy, p[a])) for a in ARMS}, cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
        save_json(path, rec); a = rec["arms"]
        log(f"{key} s{s}: pooled chose {chosen} | pooled {a['pooled']['auroc']:.4f} cf-res {a['pooled_cf_residual']['auroc']:.4f} cf-head {a['pooled_cf_head']['auroc']:.4f} "
            f"cf-ft {a['pooled_cf_finetuned']['auroc']:.4f} | confined {a['confined_chosen']['auroc']:.4f} strict {a['confined_strict_chosen']['auroc']:.4f} ({time.time()-ts:.0f}s)")

# ---------------------------------------------------------------- summary and hospital-cluster bootstrap from disk
if os.environ.get("SUMMARY", "1") == "0": log("runs only; summary skipped"); raise SystemExit(0)
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
S = pd.DataFrame([dict(consortium=r["consortium"], seed=r["seed"], arm=a, **m) for r in recs for a, m in r["arms"].items()]).groupby(["consortium", "arm"]).mean(numeric_only=True).reset_index()
S.to_csv(f"{OUT}/summary.csv", index=False); W = S.pivot_table(index="consortium", columns="arm", values="auroc")
ch = pd.Series({r["consortium"]: r["selection"]["chosen"] for r in recs}).groupby(level=0).agg(lambda x: ",".join(x))
print("\nAUROC on each consortium's held-out stays, seed means"); print(pd.concat([W[ARMS].round(4), ch.rename("chosen per seed")], axis=1).to_string())
print("choices:", pd.Series([r["selection"]["chosen"] for r in recs]).value_counts().to_dict())
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); rk = np.empty(len(pv)); rk[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (rk[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); k_, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(k_, int(s_))] = {k: Z[k] for k in Z.files}
seeds = sorted(set(s_ for _, s_ in P)); keys = sorted(set(k_ for k_, _ in P if all((k_, s_) in P for s_ in seeds)))
pairs = {"confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)": ("confined_chosen", "pooled_cf_chosen"),
         "confined strict (chosen) minus symmetric comparator": ("confined_strict_chosen", "pooled_cf_chosen"),
         "confined (chosen) minus pooled with cross-fitted residual (v20 comparator)": ("confined_chosen", "pooled_cf_residual"),
         "pooled with chosen cross-fitted component minus pooled": ("pooled_cf_chosen", "pooled"), "pooled + cross-fitted residual minus pooled": ("pooled_cf_residual", "pooled"),
         "pooled + cross-fitted head minus pooled": ("pooled_cf_head", "pooled"), "pooled + cross-fitted fine-tuned copy minus pooled": ("pooled_cf_finetuned", "pooled")}
def run_boot(pop, label):
    hosp_all = np.unique(np.concatenate([P[(k, seeds[0])]["hosp"] for k in keys]))
    def stats(keep_h, weights):
        out = {}
        for k in keys:
            Z0 = P[(k, seeds[0])]; m = np.isin(Z0["hosp"], keep_h) & (Z0["landmark"] if pop == "landmark" else True)
            if weights is not None: idx = np.concatenate([np.where(Z0["hosp"] == h)[0] for h in keep_h for _ in range(weights[h])]); idx = idx[(Z0["landmark"][idx] if pop == "landmark" else np.ones(len(idx), bool))]
            else: idx = np.where(m)[0]
            if len(idx) < 50 or len(np.unique(Z0["y"][idx])) < 2: continue
            o = {}
            for a in ARMS:
                o[a] = (np.nanmean([fast_auc(P[(k, s_)]["y"][idx], P[(k, s_)][a][idx]) for s_ in seeds]),
                        np.mean([missed_at_k(P[(k, s_)]["y"][idx], P[(k, s_)][a][idx], max(1, int(round(int(P[(k, s_)]["k_flag"]) * len(idx) / len(Z0["y"]))))) for s_ in seeds]))
            out[k] = o
        return out
    def contrasts(st):
        d = {k: np.mean([o[a][0] - o[b][0] for o in st.values()]) for k, (a, b) in pairs.items()}
        d["events unflagged per 1,000 at matched flag rate, confined minus symmetric comparator"] = np.mean([o["confined_chosen"][1] - o["pooled_cf_chosen"][1] for o in st.values()]); return d
    pt = contrasts(stats(hosp_all, None)); r = np.random.RandomState(0); boots = {k: [] for k in pt}
    for b in range(CFG["n_boot"]):
        drawn = r.choice(hosp_all, len(hosp_all), True); w = dict(zip(*np.unique(drawn, return_counts=True)))
        for k, v in contrasts(stats(np.array(list(w)), w)).items(): boots[k].append(v)
        if (b + 1) % 200 == 0: log(f"  {pop}: draw {b + 1}")
    res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_boot=CFG["n_boot"]) for k in pt}
    lw = {k: int(sum((W.loc[c, a] - W.loc[c, b]) < 0 for c in keys)) for k, (a, b) in pairs.items()}
    print(f"\n== {label}: {len(keys)} consortia, hospital-cluster bootstrap ({CFG['n_boot']} draws)")
    for k, v in res.items(): print(f"  {k:95s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(keys)}" if k in lw else ""))
    return dict(results=res, lower_in=lw)
if keys:
    out = {"all": run_boot("all", "all held-out stays"), "landmark": run_boot("landmark", "stays in the ICU at 24 h")}
    save_json(f"{OUT}/intervals.json", dict(**out, date=time.strftime("%Y-%m-%d %H:%M:%S")))
log(f"done, results in {OUT}")
