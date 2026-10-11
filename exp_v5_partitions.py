"""
exp_v5_partitions.py. Collective withdrawal of REAL partitions of MIMIC-IV, with clinical metrics.

Partitions (patient-level, from metadata held out of the feature set):
  race_group (asian, black, hispanic, native, other, unknown, white), non-English language, Medicaid insurance,
  anchor_year_group, and optionally the 25 k-means pseudo-communities refitted on TRAINING patients only.
For each group: original model, exact retrain without the group, exact retrain without a patient-matched random
set and without an admission-matched random set, fine-tune and bounded NegGrad+ for the group and for the
patient-matched random set. Everything is evaluated on the group's own held-out patients: AUROC, ECE, Brier,
calibration slope and intercept, sensitivity and PPV at the model's top-decile threshold and at a fixed cut-off,
net benefit, reclassification against the original model, plus full-test-set AUROC. One JSON per (group, seed),
written atomically; the script resumes by skipping finished JSONs.

Env: ROOT (default /content/drive/MyDrive/mimic), QUICK=1 for a smoke run, COMMUNITIES=1 to include the refitted
pseudo-communities (full mode only). Outputs: ROOT/results/v5_partitions[_quick]/{runs,ckpt,summary.csv,paired.json,zip}.
"""
import os, sys, json, time, glob, zipfile, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", "/content/drive/MyDrive/mimic")
COHORT = os.environ.get("COHORT", f"{ROOT}/derived/cohort_latest.parquet")
OUT = f"{ROOT}/results/v5_partitions" + ("_quick" if QUICK else "")
RUNS, CKPT = f"{OUT}/runs", f"{OUT}/ckpt"
os.makedirs(RUNS, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, unlearn_epochs=1 if QUICK else 2,
           lr=1e-3, unlearn_lr=1e-4, neggrad_alpha=0.05, neggrad_cap=0.6, batch=2048,
           n_boot=30 if QUICK else 200, k_communities=25, min_patients=300,
           communities=(os.environ.get("COMMUNITIES", "0") == "1") and not QUICK,
           fixed_cut=0.3, nb_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4], version="v5.0")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ============================================================== data (identical split and scaling to exp_v2)
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group",
        "admission_type", "admission_location", "discharge_location", "insurance",
        "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(COHORT)
feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32)
y = df["readmit30"].values.astype(np.float32)
sid = df.subject_id.values
rng = np.random.RandomState(0)
pids = df.subject_id.unique(); rng.shuffle(pids)
test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6
X = (X - mu) / sd
rows_per_patient = pd.Series(sid[is_train]).value_counts()
log(f"{len(df)} admissions, {len(pids)} patients, {X.shape[1]} features, train {is_train.sum()}, test {is_test.sum()}")

# ---------- patient-level partition labels (modal value over a patient's admissions)
def modal_label(col):
    cnt = df.groupby(["subject_id", col]).size().reset_index(name="n")
    cnt = cnt.sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id")
    return cnt.set_index("subject_id")[col]
labels = {}
race = modal_label("race_group")
for g in sorted(race.unique()):
    labels[("race", str(g))] = set(race.index[race == g])
if "english" in df.columns:
    eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
ins = modal_label("insurance")
if (ins == "Medicaid").any(): labels[("insurance", "medicaid")] = set(ins.index[ins == "Medicaid"])
ayg = modal_label("anchor_year_group")
for g in sorted(ayg.unique()):
    labels[("anchor_year", str(g).replace(" ", "").replace("-", "_"))] = set(ayg.index[ayg == g])

if CFG["communities"]:
    pat_mean = pd.DataFrame(X).groupby(sid).mean()
    train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
    assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    for c in range(CFG["k_communities"]):
        labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])

groups = []
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype))
    f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p = len(set(sid[f_tr])); n_te_p = len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50:
        log(f"skip {fam}/{name}: {n_tr_p} training patients, {n_te_p} held-out patients"); continue
    groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p, n_te_p=n_te_p))
if QUICK: groups = [g for g in groups if g["family"] in ("race", "language")][:2]
log(f"{len(groups)} groups: " + ", ".join(f"{g['family']}/{g['name']}({g['n_tr_p']})" for g in groups))

# ---------- matched random controls, drawn once per group with a fixed seed
def patient_matched(g, r):
    tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]])))
    fp = r.choice(tr_p, min(g["n_tr_p"], len(tr_p)), False)
    return np.isin(sid, fp) & is_train
def admission_matched(g, r):
    target = int(g["f_tr"].sum())
    tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]]))); r.shuffle(tr_p)
    chosen, tot = [], 0
    for p in tr_p:
        if tot >= target: break
        chosen.append(p); tot += int(rows_per_patient[p])
    return np.isin(sid, np.array(chosen)) & is_train
for i, g in enumerate(groups):
    r = np.random.RandomState(100 + i)
    g["r_pat"] = patient_matched(g, r); g["r_adm"] = admission_matched(g, r)

# ---------- isolation of each group in feature space (for the dose-response analysis)
pat_mean_all = pd.DataFrame(X[is_train]).groupby(sid[is_train]).mean()
def isolation(g):
    r = np.random.RandomState(3)
    gp = np.array(sorted(set(sid[g["f_tr"]]))); rest = np.array(sorted(set(pat_mean_all.index) - set(gp)))
    gs = r.choice(gp, min(2000, len(gp)), False); rs = r.choice(rest, min(50000, len(rest)), False)
    G, Rr = pat_mean_all.loc[gs].values, pat_mean_all.loc[rs].values
    d_in = NearestNeighbors(n_neighbors=2).fit(G).kneighbors(G)[0][:, 1]
    d_out = NearestNeighbors(n_neighbors=1).fit(Rr).kneighbors(G)[0][:, 0]
    cent_gap = float(np.linalg.norm(G.mean(0) - Rr.mean(0)))
    return dict(nn_dist_within=float(d_in.mean()), nn_dist_to_retained=float(d_out.mean()),
                isolation_ratio=float(d_out.mean() / max(d_in.mean(), 1e-9)), centroid_gap=cent_gap,
                frac_nearest_outside=float((d_out < d_in).mean()))

# ============================================================== model
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
def cached(key, fn):
    path = f"{CKPT}/{key}.pt"
    if os.path.exists(path):
        m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m
    m = fn(); torch.save(m.state_dict(), path + ".tmp"); os.replace(path + ".tmp", path); return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    m.train(); return np.concatenate(out).astype(np.float64)

# ============================================================== metrics
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def cal_slope_intercept(yy, pp):
    lg = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    if len(np.unique(yy)) < 2: return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg, yy)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])
def threshold_metrics(yy, pp, thr):
    flag = pp >= thr; tp = float((flag & (yy == 1)).sum()); fp = float((flag & (yy == 0)).sum()); n = len(yy)
    sens = tp / max((yy == 1).sum(), 1); ppv = tp / max(flag.sum(), 1)
    return dict(flag_rate=float(flag.mean()), sensitivity=float(sens), ppv=float(ppv), tp=tp, fp=fp, n=n), flag
def net_benefit(yy, pp, pt):
    flag = pp >= pt; n = len(yy); tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum()
    return float(tp / n - fp / n * pt / (1 - pt))
def boot_patient(fn, yy, pp, pid):
    r = np.random.RandomState(1); ups = np.unique(pid); groups_ = {p: np.where(pid == p)[0] for p in ups}; vals = []
    for _ in range(CFG["n_boot"]):
        i = np.concatenate([groups_[p] for p in r.choice(ups, len(ups), True)]); vals.append(fn(yy[i], pp[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

def evaluate(m, g, s, p_orig_group, thr_orig):
    te = np.where(g["f_te"])[0]; yy, pid, Xg = y[te], sid[te], X[te]
    p_full = predict(m, X[is_test]); p = predict(m, Xg)
    thr10 = float(np.percentile(p_full, 90))
    m10, flag10 = threshold_metrics(yy, p, thr10)
    mfix, flagfix = threshold_metrics(yy, p, CFG["fixed_cut"])
    slope, intercept = cal_slope_intercept(yy, p)
    alo, ahi = boot_patient(auc, yy, p, pid)
    out = dict(auroc=auc(yy, p), auroc_lo=alo, auroc_hi=ahi, ece=ece(yy, p), brier=float(np.mean((p - yy) ** 2)),
               cal_slope=slope, cal_intercept=intercept, mean_pred=float(p.mean()), observed=float(yy.mean()),
               full_test_auroc=auc(y[is_test], p_full), thr_top_decile=thr10,
               top_decile=m10, fixed_cut=mfix,
               net_benefit={str(pt): net_benefit(yy, p, pt) for pt in CFG["nb_thresholds"]})
    if p_orig_group is not None:
        _, flag_o = threshold_metrics(yy, p_orig_group, thr_orig)
        out["reclassified_top_decile"] = float((flag10 != flag_o).mean())
        out["mean_abs_change"] = float(np.abs(p - p_orig_group).mean())
    return out

# ============================================================== run
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
originals = {}
for s in CFG["seeds"]:
    originals[s] = cached(f"orig_s{s}", lambda s=s: train(X[is_train], y[is_train], s, CFG["epochs"], CFG["lr"]))
log("original models ready")

for g in groups:
    iso = None
    for s in CFG["seeds"]:
        path = f"{RUNS}/{g['family']}__{g['name']}__s{s}.json"
        if os.path.exists(path):
            try:
                json.load(open(path)); log(f"skip {g['family']}/{g['name']} seed {s} (done)"); continue
            except Exception: pass
        if iso is None: iso = isolation(g)
        tg = time.time(); key = f"{g['family']}_{g['name']}"
        st = originals[s].state_dict()
        r_tr_g, r_tr_p, r_tr_a = is_train & ~g["f_tr"], is_train & ~g["r_pat"], is_train & ~g["r_adm"]
        models = {"original": originals[s]}
        models["exact_group"] = cached(f"exact_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s, CFG["epochs"], CFG["lr"]))
        models["exact_random_pat"] = cached(f"exactrp_{key}_s{s}", lambda: train(X[r_tr_p], y[r_tr_p], s, CFG["epochs"], CFG["lr"]))
        models["exact_random_adm"] = cached(f"exactra_{key}_s{s}", lambda: train(X[r_tr_a], y[r_tr_a], s, CFG["epochs"], CFG["lr"]))
        models["finetune_group"] = cached(f"ft_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st))
        models["neggrad_group"] = cached(f"ng_{key}_s{s}", lambda: train(X[r_tr_g], y[r_tr_g], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st, forget=(X[g["f_tr"]], y[g["f_tr"]])))
        models["finetune_random_pat"] = cached(f"ftrp_{key}_s{s}", lambda: train(X[r_tr_p], y[r_tr_p], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st))
        models["neggrad_random_pat"] = cached(f"ngrp_{key}_s{s}", lambda: train(X[r_tr_p], y[r_tr_p], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st, forget=(X[g["r_pat"]], y[g["r_pat"]])))
        te = np.where(g["f_te"])[0]
        p_orig = predict(originals[s], X[te]); thr_orig = float(np.percentile(predict(originals[s], X[is_test]), 90))
        res = {name: evaluate(m, g, s, None if name == "original" else p_orig, thr_orig) for name, m in models.items()}
        rec = dict(family=g["family"], group=g["name"], seed=s, n_train_patients=g["n_tr_p"], n_train_rows=int(g["f_tr"].sum()),
                   n_test_patients=g["n_te_p"], n_test_rows=int(g["f_te"].sum()),
                   prevalence_group_train=float(y[g["f_tr"]].mean()), prevalence_random_pat_train=float(y[g["r_pat"]].mean()),
                   rows_random_pat=int(g["r_pat"].sum()), rows_random_adm=int(g["r_adm"].sum()),
                   isolation=iso, models=res, cfg=CFG, device=DEVICE, gpu=(torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE),
                   torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"),
                   seconds=time.time() - tg)
        save_json(path, rec)
        log(f"{g['family']}/{g['name']} seed {s}: exact dAUROC {res['exact_group']['auroc']-res['original']['auroc']:+.4f} "
            f"vs random {res['exact_random_pat']['auroc']-res['original']['auroc']:+.4f} ({time.time()-tg:.0f}s)")

# ============================================================== summary from disk
def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs yet"); return None
    rows = []
    for r in recs:
        a0, e0 = r["models"]["original"]["auroc"], r["models"]["original"]["ece"]
        for name, m in r["models"].items():
            rows.append(dict(family=r["family"], group=r["group"], seed=r["seed"], model=name,
                             n_train_patients=r["n_train_patients"], n_train_rows=r["n_train_rows"], n_test_patients=r["n_test_patients"],
                             auroc=m["auroc"], d_auroc=m["auroc"] - a0, ece=m["ece"], d_ece=m["ece"] - e0, brier=m["brier"],
                             cal_slope=m["cal_slope"], cal_intercept=m["cal_intercept"], full_test_auroc=m["full_test_auroc"],
                             sens_top_decile=m["top_decile"]["sensitivity"], ppv_top_decile=m["top_decile"]["ppv"],
                             flag_rate_top_decile=m["top_decile"]["flag_rate"], sens_fixed=m["fixed_cut"]["sensitivity"],
                             ppv_fixed=m["fixed_cut"]["ppv"], **{"nb_0.2": m["net_benefit"]["0.2"], "nb_0.3": m["net_benefit"]["0.3"]},
                             reclassified_top_decile=m.get("reclassified_top_decile", 0.0), mean_abs_change=m.get("mean_abs_change", 0.0),
                             isolation_ratio=r["isolation"]["isolation_ratio"], centroid_gap=r["isolation"]["centroid_gap"],
                             frac_nearest_outside=r["isolation"]["frac_nearest_outside"]))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results.csv", index=False)
    S = R.groupby(["family", "group", "model"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
    W = S.pivot_table(index=["family", "group"], columns="model", values="d_auroc"); WE = S.pivot_table(index=["family", "group"], columns="model", values="d_ece")
    pd.set_option("display.width", 220)
    print("\nchange in AUROC on the group's own held-out patients, seed means")
    cols = [c for c in ["exact_group", "exact_random_pat", "exact_random_adm", "finetune_group", "neggrad_group", "finetune_random_pat", "neggrad_random_pat"] if c in W.columns]
    tab = W[cols].round(4); tab.insert(0, "n_train_pat", S[S.model == "original"].set_index(["family", "group"])["n_train_patients"].astype(int))
    print(tab.to_string())
    print("\nchange in ECE"); print(WE[cols].round(4).to_string())
    out = {}
    def paired(a, b):
        d = (a - b).dropna().values
        if len(d) < 2: return dict(d=float(d.mean()) if len(d) else float("nan"), ci=[float("nan")] * 2, frac_worse=float("nan"), n=int(len(d)))
        r = np.random.RandomState(0); bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
        return dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), n=int(len(d)))
    print("\npaired contrasts, group removal minus matched random removal (over groups)")
    for fam in W.index.get_level_values(0).unique():
        w = W.loc[fam]; we = WE.loc[fam]
        for a, b in [("exact_group", "exact_random_pat"), ("exact_group", "exact_random_adm"), ("finetune_group", "finetune_random_pat"), ("neggrad_group", "neggrad_random_pat")]:
            if a in w.columns and b in w.columns:
                pa, pe = paired(w[a], w[b]), paired(we[a], we[b]); out[f"{fam}|{a}_vs_{b}"] = dict(auroc=pa, ece=pe)
                print(f"  {fam:20s} {a:15s} vs {b:18s} n={pa['n']:2d} dAUROC {pa['d']:+.4f} [{pa['ci'][0]:+.4f},{pa['ci'][1]:+.4f}] worse {100*pa['frac_worse']:.0f}%  dECE {pe['d']:+.4f}")
    with open(f"{OUT}/paired.json", "w") as f: json.dump(out, f, indent=1)
    cl = S[S.model.isin(["original", "exact_group"])].pivot_table(index=["family", "group"], columns="model", values=["sens_top_decile", "ppv_top_decile", "cal_slope", "nb_0.2"])
    print("\nclinical metrics on the group's held-out patients, original vs exact removal of the group (top-decile threshold)")
    print(cl.round(3).to_string())
    return S
def export():
    z = f"{OUT}/v5_partitions_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv", f"{OUT}/paired.json"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    cwd_copy = os.path.join(os.getcwd(), os.path.basename(z))
    if os.path.abspath(cwd_copy) != os.path.abspath(z):
        import shutil; shutil.copy(z, cwd_copy)
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB) and {cwd_copy}")

if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
