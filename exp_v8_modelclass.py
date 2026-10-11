"""
exp_v8_modelclass.py. Does the main contrast depend on the MLP? Repeats exact retraining without a refitted
pseudo-community, without a patient-matched random set and without an admission-matched random set, with
gradient-boosted trees (sklearn HistGradientBoostingClassifier) and logistic regression, evaluated on the
community's own held-out patients (AUROC, ECE, Brier). One seed. Same data preparation, split and clustering
as exp_v5_partitions. Env: ROOT (default ~/mimic), QUICK=1. Output ROOT/results/v8_modelclass[_quick]/.
"""
import os, json, time, glob, zipfile, platform
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
COHORT = os.environ.get("COHORT", f"{ROOT}/derived/cohort_latest.parquet")
OUT = f"{ROOT}/results/v8_modelclass" + ("_quick" if QUICK else "")
RUNS = f"{OUT}/runs"; os.makedirs(RUNS, exist_ok=True)
CFG = dict(seed=0, k_communities=25, min_patients=300, gbdt=dict(max_iter=60 if QUICK else 300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0),
           lr=dict(C=1.0, max_iter=300), version="v8.0")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"quick={QUICK}; out {OUT}")

META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location",
        "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(COHORT)
feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids)
test_p = set(pids[: int(0.2 * len(pids))]); is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
rows_per_patient = pd.Series(sid[is_train]).value_counts()
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index); comm = assign.loc[sid].values
groups = []
for c in range(CFG["k_communities"]):
    f_tr, f_te = is_train & (comm == c), is_test & (comm == c); n_tr_p = len(set(sid[f_tr]))
    if n_tr_p >= CFG["min_patients"] and len(set(sid[f_te])) >= 50: groups.append(dict(c=c, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p))
if QUICK: groups = groups[:2]
for i, g in enumerate(groups):
    r = np.random.RandomState(500 + i); tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]])))
    g["r_pat"] = np.isin(sid, r.choice(tr_p, g["n_tr_p"], False)) & is_train
    r.shuffle(tr_p); chosen, tot, target = [], 0, int(g["f_tr"].sum())
    for p in tr_p:
        if tot >= target: break
        chosen.append(p); tot += int(rows_per_patient[p])
    g["r_adm"] = np.isin(sid, np.array(chosen)) & is_train
log(f"{len(df)} admissions; {len(groups)} communities")

def fit(kind, Xtr, ytr):
    if kind == "gbdt":
        m = HistGradientBoostingClassifier(random_state=CFG["seed"], early_stopping=False, **CFG["gbdt"])
    else:
        m = LogisticRegression(**CFG["lr"])
    return m.fit(Xtr, ytr)
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def evaluate(m, te):
    p = m.predict_proba(X[te])[:, 1]; pf = m.predict_proba(X[is_test])[:, 1]
    return dict(auroc=auc(y[te], p), ece=ece(y[te], p), brier=float(np.mean((p - y[te]) ** 2)), full_test_auroc=auc(y[is_test], pf))
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

originals = {}
for kind in ["gbdt", "logreg"]:
    tk = time.time(); originals[kind] = fit(kind, X[is_train], y[is_train])
    log(f"original {kind}: full-test AUROC {auc(y[is_test], originals[kind].predict_proba(X[is_test])[:, 1]):.4f} ({time.time()-tk:.0f}s)")
for g in groups:
    path = f"{RUNS}/c{g['c']:02d}.json"
    if os.path.exists(path):
        try: json.load(open(path)); log(f"skip c{g['c']:02d} (done)"); continue
        except Exception: pass
    tg = time.time(); te = np.where(g["f_te"])[0]; res = {}
    for kind in ["gbdt", "logreg"]:
        res[kind] = {"original": evaluate(originals[kind], te)}
        for name, mask in [("exact_group", g["f_tr"]), ("exact_random_pat", g["r_pat"]), ("exact_random_adm", g["r_adm"])]:
            keep = is_train & ~mask; res[kind][name] = evaluate(fit(kind, X[keep], y[keep]), te)
    rec = dict(community=g["c"], n_train_patients=g["n_tr_p"], n_train_rows=int(g["f_tr"].sum()), n_test_rows=int(len(te)),
               models=res, cfg=CFG, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
    save_json(path, rec)
    log(f"c{g['c']:02d}: gbdt exact {res['gbdt']['exact_group']['auroc']-res['gbdt']['original']['auroc']:+.4f} vs random "
        f"{res['gbdt']['exact_random_pat']['auroc']-res['gbdt']['original']['auroc']:+.4f} | logreg exact "
        f"{res['logreg']['exact_group']['auroc']-res['logreg']['original']['auroc']:+.4f} vs random "
        f"{res['logreg']['exact_random_pat']['auroc']-res['logreg']['original']['auroc']:+.4f} ({time.time()-tg:.0f}s)")

def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs"); return
    rows = []
    for rc in recs:
        for kind, res in rc["models"].items():
            a0, e0 = res["original"]["auroc"], res["original"]["ece"]
            for name, m in res.items():
                rows.append(dict(community=rc["community"], n_train_patients=rc["n_train_patients"], model_class=kind, model=name,
                                 auroc=m["auroc"], d_auroc=m["auroc"] - a0, ece=m["ece"], d_ece=m["ece"] - e0, brier=m["brier"], full_test_auroc=m["full_test_auroc"]))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results.csv", index=False); pd.set_option("display.width", 200)
    out = {}
    for kind in ["gbdt", "logreg"]:
        W = R[R.model_class == kind].pivot(index="community", columns="model", values="d_auroc")
        WE = R[R.model_class == kind].pivot(index="community", columns="model", values="d_ece")
        print(f"\n{kind}: mean change in AUROC on the community's held-out patients"); print(W.mean().round(4).to_string())
        for b in ["exact_random_pat", "exact_random_adm"]:
            d = (W["exact_group"] - W[b]).values; de = (WE["exact_group"] - WE[b]).values; r = np.random.RandomState(0)
            bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
            out[f"{kind}|exact_group_vs_{b}"] = dict(d_auroc=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                     frac_worse=float((d < 0).mean()), d_ece=float(de.mean()), n=int(len(d)))
            print(f"  paired vs {b}: dAUROC {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] worse {100*(d<0).mean():.0f}%  dECE {de.mean():+.4f} (n={len(d)})")
        print(f"  full-test AUROC original {R[(R.model_class==kind)&(R.model=='original')].full_test_auroc.mean():.4f}")
    with open(f"{OUT}/paired.json", "w") as f: json.dump(out, f, indent=1)
def export():
    z = f"{OUT}/v8_modelclass_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/paired.json"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
