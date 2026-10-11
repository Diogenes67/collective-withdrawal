"""
exp_v7_eicu_consortium.py. Consortium withdrawal on eICU-CRD v2.0: all hospitals in a region, all hospitals of a
teaching status, all hospitals in a bed-size band, each withdrawn as one governed partition with authority and
structural correlation together. Same models, remedies and clinical metrics as exp_v5_partitions, evaluated on the
consortium's own held-out patients, against a stay-matched random control drawn from the other hospitals.

Env: ROOT (default ~/mimic), QUICK=1. Reads ROOT/derived/eicu_cohort.parquet and the originals in
ROOT/results/model_cache_eicu (orig_s{seed}); writes ROOT/results/v7_eicu_consortium[_quick]/{runs,ckpt,...}.
"""
import os, json, time, glob, zipfile, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
COHORT = f"{ROOT}/derived/eicu_cohort.parquet"
ORIG_CACHE = f"{ROOT}/results/model_cache_eicu"
OUT = f"{ROOT}/results/v7_eicu_consortium" + ("_quick" if QUICK else "")
RUNS, CKPT = f"{OUT}/runs", f"{OUT}/ckpt"; os.makedirs(RUNS, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, unlearn_epochs=1 if QUICK else 2, lr=1e-3, unlearn_lr=1e-4,
           neggrad_alpha=0.05, neggrad_cap=0.6, batch=2048, n_boot=30 if QUICK else 200, min_stays=800, fixed_cut=0.1,
           nb_thresholds=[0.05, 0.1, 0.15, 0.2, 0.3], version="v7.0")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ---------- data, identical to exp_eicu
df = pd.read_parquet(COHORT)
META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
feat = [c for c in df.columns if c not in META]
Xdf = df[feat].astype(float)
for c in Xdf.columns[Xdf.isna().any()]:
    Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32)
pid = df.uniquepid.values; hosp_id = df.hospitalid.values
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups)
test_p = set(ups[: int(0.2 * len(ups))]); is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
log(f"{len(df)} stays, {X.shape[1]} features, mortality {y.mean():.3f}, train {is_train.sum()}, test {is_test.sum()}")

# ---------- consortium partitions (hospital-level attributes)
hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first()
labels = {}
for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
    vals = hattr[col].dropna().unique()
    for v in sorted(vals, key=str):
        hs = set(hattr.index[hattr[col] == v]); labels[(fam, str(v).replace(" ", "_").replace("/", "_"))] = hs
groups = []
for (fam, name), hs in labels.items():
    mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    if f_tr.sum() < CFG["min_stays"] or f_te.sum() < 100: log(f"skip {fam}/{name}: {f_tr.sum()} training stays"); continue
    groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_hosp=len(hs), n_tr=int(f_tr.sum()), n_te=int(f_te.sum())))
if QUICK: groups = groups[:2]
log(f"{len(groups)} consortia: " + ", ".join(f"{g['family']}/{g['name']}({g['n_hosp']} hospitals, {g['n_tr']} stays)" for g in groups))
for i, g in enumerate(groups):
    r = np.random.RandomState(300 + i); other = np.where(is_train & ~g["f_tr"])[0]
    pick = r.choice(other, min(g["n_tr"], len(other)), False); ctrl = np.zeros(len(df), bool); ctrl[pick] = True; g["r_ctrl"] = ctrl

# ---------- model (as exp_eicu)
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train(Xtr, ytr, seed, epochs, lr, init=None, forget=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]).to(DEVICE)
    if init is not None: m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=lr); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(Xtr), to_t(ytr); n, B = len(Xtr), CFG["batch"]
    if forget is not None: Xf, yf = to_t(forget[0]), to_t(forget[1])
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx])
            if forget is not None:
                j = torch.randint(0, len(Xf), (min(B, len(Xf)),), device=DEVICE)
                loss = loss - CFG["neggrad_alpha"] * torch.clamp(bce(m(Xf[j]), yf[j]), max=CFG["neggrad_cap"])
            opt.zero_grad(); loss.backward(); opt.step()
    return m
def load_from(folder, key):
    path = f"{folder}/{key}.pt"
    if not os.path.exists(path): return None
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m
def cached(key, fn):
    m = load_from(CKPT, key)
    if m is None:
        m = fn(); torch.save(m.state_dict(), f"{CKPT}/{key}.pt.tmp"); os.replace(f"{CKPT}/{key}.pt.tmp", f"{CKPT}/{key}.pt")
    return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    m.train(); return np.concatenate(out).astype(np.float64)

# ---------- metrics (as exp_v5)
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def cal_slope_intercept(yy, pp):
    pc = np.clip(pp, 1e-6, 1 - 1e-6); lg = np.log(pc / (1 - pc)).reshape(-1, 1)
    if len(np.unique(yy)) < 2: return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg, yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def threshold_metrics(yy, pp, thr):
    flag = pp >= thr; tp = float((flag & (yy == 1)).sum())
    return dict(flag_rate=float(flag.mean()), sensitivity=float(tp / max((yy == 1).sum(), 1)), ppv=float(tp / max(flag.sum(), 1))), flag
def net_benefit(yy, pp, pt):
    flag = pp >= pt; n = len(yy); tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum()
    return float(tp / n - fp / n * pt / (1 - pt))
def boot_patient(fn, yy, pp, pp_id):
    r = np.random.RandomState(1); ups_ = np.unique(pp_id); gi = {p: np.where(pp_id == p)[0] for p in ups_}; vals = []
    for _ in range(CFG["n_boot"]):
        i = np.concatenate([gi[p] for p in r.choice(ups_, len(ups_), True)]); vals.append(fn(yy[i], pp[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
def evaluate(m, g, p_orig_group, thr_orig):
    te = np.where(g["f_te"])[0]; yy, pp_id, Xg = y[te], pid[te], X[te]
    p_full = predict(m, X[is_test]); p = predict(m, Xg); thr10 = float(np.percentile(p_full, 90))
    m10, flag10 = threshold_metrics(yy, p, thr10); mfix, _ = threshold_metrics(yy, p, CFG["fixed_cut"])
    slope, intercept = cal_slope_intercept(yy, p); alo, ahi = boot_patient(auc, yy, p, pp_id)
    out = dict(auroc=auc(yy, p), auroc_lo=alo, auroc_hi=ahi, ece=ece(yy, p), brier=float(np.mean((p - yy) ** 2)), cal_slope=slope,
               cal_intercept=intercept, mean_pred=float(p.mean()), observed=float(yy.mean()), full_test_auroc=auc(y[is_test], p_full),
               thr_top_decile=thr10, top_decile=m10, fixed_cut=mfix, net_benefit={str(pt): net_benefit(yy, p, pt) for pt in CFG["nb_thresholds"]})
    if p_orig_group is not None:
        _, flag_o = threshold_metrics(yy, p_orig_group, thr_orig)
        out["reclassified_top_decile"] = float((flag10 != flag_o).mean()); out["mean_abs_change"] = float(np.abs(p - p_orig_group).mean())
    return out
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

originals = {}
for s in CFG["seeds"]:
    m = load_from(ORIG_CACHE, f"orig_s{s}") if not QUICK else None
    originals[s] = m if m is not None else cached(f"orig_s{s}", lambda s=s: train(X[is_train], y[is_train], s, CFG["epochs"], CFG["lr"]))
log("original models ready")

for g in groups:
    for s in CFG["seeds"]:
        path = f"{RUNS}/{g['family']}__{g['name']}__s{s}.json"
        if os.path.exists(path):
            try: json.load(open(path)); log(f"skip {g['family']}/{g['name']} seed {s} (done)"); continue
            except Exception: pass
        tg = time.time(); key = f"{g['family']}_{g['name']}"; st = originals[s].state_dict()
        r_tr_g, r_tr_c = is_train & ~g["f_tr"], is_train & ~g["r_ctrl"]
        models = {"original": originals[s],
                  "exact_group": cached(f"exact_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s, CFG["epochs"], CFG["lr"])),
                  "exact_random": cached(f"exactr_{key}_s{s}", lambda: train(X[r_tr_c], y[r_tr_c], s, CFG["epochs"], CFG["lr"])),
                  "finetune_group": cached(f"ft_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st)),
                  "neggrad_group": cached(f"ng_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st, forget=(X[g["f_tr"]], y[g["f_tr"]]))),
                  "finetune_random": cached(f"ftr_{key}_s{s}", lambda: train(X[r_tr_c], y[r_tr_c], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st)),
                  "neggrad_random": cached(f"ngr_{key}_s{s}", lambda: train(X[r_tr_c], y[r_tr_c], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st, forget=(X[g["r_ctrl"]], y[g["r_ctrl"]])))}
        te = np.where(g["f_te"])[0]; p_orig = predict(originals[s], X[te]); thr_orig = float(np.percentile(predict(originals[s], X[is_test]), 90))
        res = {name: evaluate(m, g, None if name == "original" else p_orig, thr_orig) for name, m in models.items()}
        rec = dict(family=g["family"], group=g["name"], seed=s, n_hospitals=g["n_hosp"], n_train_rows=g["n_tr"], n_test_rows=g["n_te"],
                   n_test_patients=int(len(set(pid[te]))), frac_training=float(g["n_tr"] / is_train.sum()),
                   mortality_group_train=float(y[g["f_tr"]].mean()), mortality_control_train=float(y[g["r_ctrl"]].mean()),
                   models=res, cfg=CFG, device=DEVICE, torch=torch.__version__, python=platform.python_version(),
                   date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
        save_json(path, rec)
        log(f"{g['family']}/{g['name']} seed {s}: exact dAUROC {res['exact_group']['auroc']-res['original']['auroc']:+.4f} vs random "
            f"{res['exact_random']['auroc']-res['original']['auroc']:+.4f} ({time.time()-tg:.0f}s)")

def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs"); return
    rows = []
    for r in recs:
        a0, e0 = r["models"]["original"]["auroc"], r["models"]["original"]["ece"]
        for name, m in r["models"].items():
            rows.append(dict(family=r["family"], group=r["group"], seed=r["seed"], model=name, n_hospitals=r["n_hospitals"], n_train_rows=r["n_train_rows"],
                             frac_training=r["frac_training"], auroc=m["auroc"], d_auroc=m["auroc"] - a0, ece=m["ece"], d_ece=m["ece"] - e0, brier=m["brier"],
                             cal_slope=m["cal_slope"], cal_intercept=m["cal_intercept"], full_test_auroc=m["full_test_auroc"],
                             sens_top_decile=m["top_decile"]["sensitivity"], ppv_top_decile=m["top_decile"]["ppv"], sens_fixed=m["fixed_cut"]["sensitivity"],
                             ppv_fixed=m["fixed_cut"]["ppv"], **{"nb_0.1": m["net_benefit"]["0.1"], "nb_0.2": m["net_benefit"]["0.2"]},
                             reclassified_top_decile=m.get("reclassified_top_decile", 0.0)))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results.csv", index=False)
    S = R.groupby(["family", "group", "model"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 220)
    W = S.pivot_table(index=["family", "group"], columns="model", values="d_auroc"); WE = S.pivot_table(index=["family", "group"], columns="model", values="d_ece")
    cols = [c for c in ["exact_group", "exact_random", "finetune_group", "neggrad_group", "finetune_random", "neggrad_random"] if c in W.columns]
    info = S[S.model == "original"].set_index(["family", "group"])[["n_hospitals", "n_train_rows", "frac_training"]]
    print("\nchange in AUROC on the consortium's own held-out patients, seed means")
    print(pd.concat([info.round(3), W[cols].round(4)], axis=1).to_string())
    print("\nchange in ECE"); print(WE[cols].round(4).to_string())
    out = {}
    d = (W["exact_group"] - W["exact_random"]).dropna(); r = np.random.RandomState(0)
    bs = [r.choice(d.values, len(d), True).mean() for _ in range(5000)]
    out["all|exact_group_vs_exact_random"] = dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), n=int(len(d)))
    print(f"\npaired over all consortia: dAUROC {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] worse {100*(d<0).mean():.0f}% (n={len(d)})")
    with open(f"{OUT}/paired.json", "w") as f: json.dump(out, f, indent=1)
    cl = S[S.model.isin(["original", "exact_group", "exact_random"])].pivot_table(index=["family", "group"], columns="model", values=["sens_top_decile", "ppv_top_decile", "cal_slope"])
    print("\nclinical metrics, original vs exact removal (top-decile threshold)"); print(cl.round(3).to_string())
def export():
    z = f"{OUT}/v7_eicu_consortium_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv", f"{OUT}/paired.json"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
