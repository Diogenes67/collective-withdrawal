"""
exp_v13_intervals.py. Dependence-aware uncertainty and decision metrics for the paper, from cached checkpoints.

Why: the paired contrasts so far bootstrap groups as independent units. On eICU the ten consortia are three overlapping
partitions of the same hospitals and patients are nested in hospitals; on MIMIC admissions are nested in patients and
communities are the sampling unit. This script re-derives every headline interval with resampling that respects that:
  eICU   hospital-cluster bootstrap of the held-out set: resample test hospitals with replacement, recompute every
         consortium's held-out AUROC under each model from the resampled stays, take paired differences, average over
         seeds, then over consortia (overall and within each family). Overlap between consortia is preserved because the
         same hospital draw feeds all of them.
  MIMIC  two-stage bootstrap: resample the 25 communities with replacement, and within each drawn community resample its
         held-out patients (all admissions of a patient move together).
Models are loaded from the caches that produced the published runs (v5 for MIMIC; v7.2, model_cache_eicu_es and v12.2
for eICU). Group-held adapters (residual logistic on [logit, features]; the other-people control) are refit exactly as
in v11/v12.2 from the same training rows; the early-stopped fine-tuned copy is not refit here (its interval stays at the
group level from v11). Decision metrics per 1,000 held-out admissions/stays are bootstrapped the same way at a fixed
risk threshold and at the deployed system's top-decile threshold.

Env: ROOT (default ~/mimic), PART=mimic|eicu|both (default both), NBOOT (default 1000; QUICK=1 gives 20), DEVICE,
MEMBERSHIP_ONLY=1 (write membership_{mimic,eicu}.json, which v13b reads, and exit; the MIMIC membership is checked against v5).
Writes ROOT/results/v13_intervals/{preds_*.npz, intervals.json, intervals.csv, log}. Prediction arrays are cached so
the bootstrap can be rerun without touching a model.
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
PART = os.environ.get("PART", "both")
NBOOT = int(os.environ.get("NBOOT", 20 if QUICK else 1000))
OUT = f"{ROOT}/results/v13_intervals_1" + ("_quick" if QUICK else ""); os.makedirs(OUT, exist_ok=True)
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; part {PART}; n_boot {NBOOT}; out {OUT}")
CFG = dict(adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0, n_boot=NBOOT, seeds=[0, 1, 2],
           mimic_fixed_cut=0.3, eicu_fixed_cut=0.1, version="v13.1")

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def load(path, d):
    if not os.path.exists(path): raise SystemExit(f"missing checkpoint {path}")
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def logits(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(m(to_t(Xa[i:i + 65536])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
class Platt:
    """Logistic on the first column only (the shared logit); monotone, so it leaves the ranking unchanged."""
    def __init__(self, Z, yy): self.m = LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, :1], yy)
    def predict_proba(self, Z): return self.m.predict_proba(Z[:, :1])
def fit_logistic_adapter(Z, yy, pid, seed, offset=False):
    """Ridge logistic adapter, C chosen by patient-grouped CV log loss. offset=True: the first column is the shared logit, scaled by
    logit_scale so the L2 penalty barely touches its coefficient, and a logit-only (Platt) candidate competes in the CV, so the adapter
    cannot be chosen when leaving the shared ranking alone does better in CV. (v14.1 adapter; the earlier version penalised the logit.)"""
    Z = Z.copy()
    if offset: Z[:, 0] *= CFG["logit_scale"]
    n_p = len(np.unique(pid)); grid = CFG["adapter_C_grid"]; chosen, cv = "platt" if offset else grid[len(grid) // 2], None
    if n_p >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pid)); cv = {}
        for C in (["platt"] if offset else []) + list(grid):
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = Platt(Z[tr], yy[tr]) if C == "platt" else LogisticRegression(C=C, max_iter=3000).fit(Z[tr], yy[tr])
                ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[str(C)] = float(np.mean(ll))
        if cv: chosen = min(cv, key=cv.get); chosen = "platt" if chosen == "platt" else float(chosen)
    m = Platt(Z, yy) if chosen == "platt" else LogisticRegression(C=chosen, max_iter=3000).fit(Z, yy)
    class Wrapped:
        def predict_proba(self, Zn):
            Zn = Zn.copy()
            if offset: Zn[:, 0] *= CFG["logit_scale"]
            return m.predict_proba(Zn)
    return Wrapped()
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def missed_per_1000(yy, pp, thr):
    flag = pp >= thr; ev = (yy == 1).sum(); tp = (flag & (yy == 1)).sum(); return float((ev - tp) / len(yy) * 1000)
def flagged_per_1000(yy, pp, thr): return float((pp >= thr).mean() * 1000)
def pct(v): v = np.asarray(v, float); v = v[~np.isnan(v)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2

results = {}
def record(key, point, boots, extra=None):
    results[key] = dict(d=float(point), ci=pct(boots), n_boot=int(len(boots)), **(extra or {}))
    log(f"  {key:60s} {point:+.4f} [{results[key]['ci'][0]:+.4f}, {results[key]['ci'][1]:+.4f}]")

# =====================================================================================================  MIMIC
def run_mimic():
    log("MIMIC: loading cohort")
    COHORT = f"{ROOT}/derived/cohort_latest.parquet"; V5 = f"{ROOT}/results/v5_partitions"
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location",
            "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(COHORT); feat = [c for c in df.columns if c not in META]; X = df[feat].astype(float)
    for c in [c for c in feat if c.startswith("lab_")]:
        X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
    X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
    assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    test_idx = np.where(is_test)[0]
    groups = []
    for c in range(25):
        members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype))
        f_tr, f_te = is_train & mem, is_test & mem
        ref = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
        if ref["n_train_patients"] != len(set(sid[f_tr])) or ref["n_test_patients"] != len(set(sid[f_te])): raise SystemExit(f"community c{c:02d} membership differs from v5")
        groups.append(dict(name=f"c{c:02d}", f_tr=f_tr, f_te=f_te))
    log("MIMIC: 25 communities match the v5 records")
    json.dump({g["name"]: np.where(g["f_te"])[0].tolist() for g in groups}, open(f"{OUT}/membership_mimic.json", "w"))
    if os.environ.get("MEMBERSHIP_ONLY") == "1": log("membership written; exiting"); return
    cache = f"{OUT}/preds_mimic.npz"
    if os.path.exists(cache):
        Z = np.load(cache, allow_pickle=True); P = Z["P"].item(); log("MIMIC: predictions loaded from cache")
    else:
        P = {}   # P[(community, seed, arm)] = predictions on that community's held-out rows; arms: original, base, random, resid, resid_random, rep, platt_base
        for s in CFG["seeds"]:
            orig = load(f"{V5}/ckpt/orig_s{s}.pt", X.shape[1]); z_orig_full = logits(orig, X[is_test])
            for g in groups:
                te = np.where(g["f_te"])[0]; pos = np.searchsorted(test_idx, te); tr = np.where(g["f_tr"])[0]
                base = load(f"{V5}/ckpt/exact_community_trainfit_{g['name']}_s{s}.pt", X.shape[1])
                rnd = load(f"{V5}/ckpt/exactrp_community_trainfit_{g['name']}_s{s}.pt", X.shape[1])
                z_base_full = logits(base, X[is_test]); z_base_g = z_base_full[pos]; z_base_tr = logits(base, X[tr])
                P[(g["name"], s, "original")] = sigmoid(z_orig_full[pos]); P[(g["name"], s, "base")] = sigmoid(z_base_g)
                P[(g["name"], s, "random")] = sigmoid(logits(rnd, X[te]))
                P[(g["name"], s, "full_original")] = sigmoid(z_orig_full); P[(g["name"], s, "full_base")] = sigmoid(z_base_full)   # for system thresholds
                ad = fit_logistic_adapter(np.column_stack([z_base_tr, X[tr]]), y[tr], sid[tr], s, offset=True)
                P[(g["name"], s, "resid")] = ad.predict_proba(np.column_stack([z_base_g, X[te]]))[:, 1]
                with torch.no_grad():
                    h_tr = np.concatenate([base.net[:4](to_t(X[tr][i:i + 65536])).cpu().numpy() for i in range(0, len(tr), 65536)])
                    h_te = np.concatenate([base.net[:4](to_t(X[te][i:i + 65536])).cpu().numpy() for i in range(0, len(te), 65536)])
                ad = fit_logistic_adapter(h_tr, y[tr], sid[tr], s); P[(g["name"], s, "rep")] = ad.predict_proba(h_te)[:, 1]
                nong_te = test_idx[~np.isin(test_idx, te)]; r = np.random.RandomState(500 + int(g["name"][1:]) + 31 * s)
                ng_p = np.unique(sid[nong_te]); r.shuffle(ng_p); ctrl = nong_te[np.isin(sid[nong_te], ng_p[: min(len(set(sid[tr])), len(ng_p))])]
                ad = fit_logistic_adapter(np.column_stack([z_base_full[np.searchsorted(test_idx, ctrl)], X[ctrl]]), y[ctrl], sid[ctrl], s, offset=True)
                P[(g["name"], s, "resid_random")] = ad.predict_proba(np.column_stack([z_base_g, X[te]]))[:, 1]
                log(f"MIMIC {g['name']} seed {s}: original {auc(y[te], P[(g['name'], s, 'original')]):.4f} base {auc(y[te], P[(g['name'], s, 'base')]):.4f} resid {auc(y[te], P[(g['name'], s, 'resid')]):.4f}")
        np.savez_compressed(cache, P=np.array(P, dtype=object)); log(f"MIMIC: predictions cached at {cache}")
    # ---- bootstrap: communities, then patients within community
    names = [g["name"] for g in groups]; te_rows = {g["name"]: np.where(g["f_te"])[0] for g in groups}
    pat_index = {}
    for n in names:
        rows = te_rows[n]; pid = sid[rows]; ups = np.unique(pid); pat_index[n] = (ups, {p: np.where(pid == p)[0] for p in ups})
    thr_fixed = CFG["mimic_fixed_cut"]
    thr10 = {(n, s, a): float(np.percentile(P[(n, s, "full_original" if a == "original" else "full_base")], 90)) for n in names for s in CFG["seeds"] for a in ["original", "base"]}
    # composite system threshold for group arms: base everywhere else, arm on the group
    for n in names:
        for s in CFG["seeds"]:
            pos = np.searchsorted(test_idx, te_rows[n])
            for a in ["random", "resid", "rep", "resid_random"]:
                full = P[(n, s, "full_base")].copy(); full[pos] = P[(n, s, a)]; thr10[(n, s, a)] = float(np.percentile(full, 90))
            full = P[(n, s, "full_original")]; thr10[(n, s, "random")] = float(np.percentile(full, 90))   # random removal is a different shared model; approximate with the original's threshold scale
    arms = ["original", "base", "random", "resid", "rep", "resid_random"]
    def stats(n, idx):
        rows = te_rows[n][idx]; yy = y[rows]; out = {}
        for a in arms:
            vals_auc, vals_m10, vals_mfix, vals_f10 = [], [], [], []
            for s in CFG["seeds"]:
                p = P[(n, s, a)][idx]; vals_auc.append(auc(yy, p)); vals_m10.append(missed_per_1000(yy, p, thr10[(n, s, a)]))
                vals_mfix.append(missed_per_1000(yy, p, thr_fixed)); vals_f10.append(flagged_per_1000(yy, p, thr10[(n, s, a)]))
            out[a] = (np.nanmean(vals_auc), np.mean(vals_m10), np.mean(vals_mfix), np.mean(vals_f10))
        return out
    full_idx = {n: np.arange(len(te_rows[n])) for n in names}
    point = {n: stats(n, full_idx[n]) for n in names}
    def contrasts(S):
        d = {}
        d["base_vs_random (structured minus random withdrawal)"] = np.mean([S[n]["base"][0] - S[n]["random"][0] for n in S])
        d["base_vs_original"] = np.mean([S[n]["base"][0] - S[n]["original"][0] for n in S])
        d["resid_vs_original (sharing premium)"] = np.mean([S[n]["resid"][0] - S[n]["original"][0] for n in S])
        d["resid_vs_base (recovered)"] = np.mean([S[n]["resid"][0] - S[n]["base"][0] for n in S])
        d["rep_vs_base"] = np.mean([S[n]["rep"][0] - S[n]["base"][0] for n in S])
        d["resid_random_vs_base (control)"] = np.mean([S[n]["resid_random"][0] - S[n]["base"][0] for n in S])
        d["resid_vs_resid_random (group-specific gain)"] = np.mean([S[n]["resid"][0] - S[n]["resid_random"][0] for n in S])
        for a in ["base", "resid", "rep"]:
            d[f"missed_per_1000_top_decile {a} minus original"] = np.mean([S[n][a][1] - S[n]["original"][1] for n in S])
            d[f"missed_per_1000_fixed_cut {a} minus original"] = np.mean([S[n][a][2] - S[n]["original"][2] for n in S])
            d[f"flagged_per_1000_top_decile {a} minus original"] = np.mean([S[n][a][3] - S[n]["original"][3] for n in S])
        return d
    pt = contrasts(point); boots = {k: [] for k in pt}; r = np.random.RandomState(0)
    log(f"MIMIC: bootstrap {NBOOT} draws (communities, then patients within community)")
    for b in range(NBOOT):
        drawn = r.choice(names, len(names), True); S = {}
        for j, n in enumerate(drawn):
            ups, gi = pat_index[n]; idx = np.concatenate([gi[p] for p in r.choice(ups, len(ups), True)]); S[f"{n}#{j}"] = stats(n, idx)
        for k, v in contrasts(S).items(): boots[k].append(v)
        if (b + 1) % 100 == 0: log(f"  draw {b + 1}")
    for k in pt: record(f"MIMIC communities | {k}", pt[k], boots[k], dict(n_groups=len(names), unit="community, then patient"))
    # group-level-only bootstrap for comparison with the published intervals
    rg = np.random.RandomState(1); dvals = np.array([point[n]["base"][0] - point[n]["random"][0] for n in names])
    record("MIMIC communities | base_vs_random, group-level bootstrap only (as published)", dvals.mean(), [rg.choice(dvals, len(dvals), True).mean() for _ in range(5000)])

# =====================================================================================================  eICU
def run_eicu():
    log("eICU: loading cohort")
    COHORT = f"{ROOT}/derived/eicu_cohort.parquet"; V72 = f"{ROOT}/results/v7_eicu_consortium_es"; V122 = f"{ROOT}/results/v12_eicu_federated_es"
    ORIG = f"{ROOT}/results/model_cache_eicu_es"
    df = pd.read_parquet(COHORT)
    META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
    feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
    for c in Xdf.columns[Xdf.isna().any()]:
        Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32); pid = df.uniquepid.values; hosp = df.hospitalid.values
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    test_idx = np.where(is_test)[0]
    hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); groups = []
    for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
        for v in sorted(hattr[col].dropna().unique(), key=str):
            hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp, np.fromiter(hs, dtype=hosp.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
            if f_tr.sum() < 800 or f_te.sum() < 100: continue
            groups.append(dict(family=fam, name=str(v).replace(" ", "_").replace("/", "_"), hs=hs, f_tr=f_tr, f_te=f_te))
    log(f"eICU: {len(groups)} consortia")
    json.dump({f"{g['family']}_{g['name']}": sorted(int(h) for h in g["hs"]) for g in groups}, open(f"{OUT}/membership_eicu.json", "w"))
    if os.environ.get("MEMBERSHIP_ONLY") == "1": log("membership written; exiting"); return
    cache = f"{OUT}/preds_eicu.npz"
    if os.path.exists(cache):
        Z = np.load(cache, allow_pickle=True); P = Z["P"].item(); log("eICU: predictions loaded from cache")
    else:
        P = {}   # P[(key, seed, arm)] = predictions on the whole test set (composite for adapters)
        for s in CFG["seeds"]:
            orig = load(f"{ORIG}/orig_s{s}.pt", X.shape[1]); z_orig = logits(orig, X[is_test])
            flg = load(f"{V122}/ckpt/fl_global_s{s}.pt", X.shape[1]); p_flg = sigmoid(logits(flg, X[is_test]))
            for g in groups:
                key = f"{g['family']}_{g['name']}"; te = np.where(g["f_te"])[0]; pos = np.searchsorted(test_idx, te); tr = np.where(g["f_tr"])[0]
                P[(key, s, "original")] = sigmoid(z_orig)
                P[(key, s, "exact_group")] = sigmoid(logits(load(f"{V72}/ckpt/exact_{key}_s{s}.pt", X.shape[1]), X[is_test]))
                P[(key, s, "exact_random")] = sigmoid(logits(load(f"{V72}/ckpt/exactr_{key}_s{s}.pt", X.shape[1]), X[is_test]))
                P[(key, s, "fl_global")] = p_flg
                flw = load(f"{V122}/ckpt/fl_without_{key}_s{s}.pt", X.shape[1]); z_w = logits(flw, X[is_test]); P[(key, s, "fl_without")] = sigmoid(z_w)
                P[(key, s, "fl_random")] = sigmoid(logits(load(f"{V122}/ckpt/fl_random_{key}_s{s}.pt", X.shape[1]), X[is_test]))
                z_tr = logits(flw, X[tr]); ad = fit_logistic_adapter(np.column_stack([z_tr, X[tr]]), y[tr], pid[tr], s, offset=True)
                full = sigmoid(z_w).copy(); full[pos] = ad.predict_proba(np.column_stack([z_w[pos], X[te]]))[:, 1]; P[(key, s, "fl_resid")] = full
                nong_te = test_idx[~np.isin(test_idx, te)]; rc = np.random.RandomState(500 + groups.index(g) + 31 * s); ng_p = np.unique(pid[nong_te]); rc.shuffle(ng_p)
                ctrl = nong_te[np.isin(pid[nong_te], ng_p[: min(len(np.unique(pid[tr])), len(ng_p))])]
                ad = fit_logistic_adapter(np.column_stack([z_w[np.searchsorted(test_idx, ctrl)], X[ctrl]]), y[ctrl], pid[ctrl], s, offset=True)
                full = sigmoid(z_w).copy(); full[pos] = ad.predict_proba(np.column_stack([z_w[pos], X[te]]))[:, 1]; P[(key, s, "fl_resid_random")] = full
                log(f"eICU {key} seed {s}: exact_group {auc(y[te], P[(key, s, 'exact_group')][pos]):.4f} random {auc(y[te], P[(key, s, 'exact_random')][pos]):.4f} fl_resid {auc(y[te], full[pos]):.4f}")
        np.savez_compressed(cache, P=np.array(P, dtype=object)); log(f"eICU: predictions cached at {cache}")
    # ---- hospital-cluster bootstrap of the held-out set
    keys = [f"{g['family']}_{g['name']}" for g in groups]; fam_of = {f"{g['family']}_{g['name']}": g["family"] for g in groups}
    hosp_te = hosp[test_idx]; y_te = y[test_idx]; hospitals = np.unique(hosp_te); h_rows = {h: np.where(hosp_te == h)[0] for h in hospitals}
    cons_hosp = {f"{g['family']}_{g['name']}": set(g["hs"]) for g in groups}
    thr_fixed = CFG["eicu_fixed_cut"]; arms = ["original", "exact_group", "exact_random", "fl_global", "fl_without", "fl_random", "fl_resid", "fl_resid_random"]
    thr10 = {(k, s, a): float(np.percentile(P[(k, s, a)], 90)) for k in keys for s in CFG["seeds"] for a in arms}
    def stats(sample_rows, sample_hosp):
        out = {}
        for k in keys:
            m = np.isin(sample_hosp, list(cons_hosp[k])); rows = sample_rows[m]
            if len(rows) < 50 or len(np.unique(y_te[rows])) < 2: out[k] = None; continue
            yy = y_te[rows]; o = {}
            for a in arms:
                aucs, m10, mfix = [], [], []
                for s in CFG["seeds"]:
                    p = P[(k, s, a)][rows]; aucs.append(auc(yy, p)); m10.append(missed_per_1000(yy, p, thr10[(k, s, a)])); mfix.append(missed_per_1000(yy, p, thr_fixed))
                o[a] = (np.nanmean(aucs), np.mean(m10), np.mean(mfix))
            out[k] = o
        return out
    def contrasts(S):
        d = {}
        def mean_over(ks, f): v = [f(S[k]) for k in ks if S[k] is not None]; return np.mean(v) if v else np.nan
        pairs = {"exact_group_vs_exact_random (centralised penalty)": lambda o: o["exact_group"][0] - o["exact_random"][0],
                 "fl_without_vs_fl_random (federated penalty)": lambda o: o["fl_without"][0] - o["fl_random"][0],
                 "fl_resid_vs_fl_without (recovered)": lambda o: o["fl_resid"][0] - o["fl_without"][0],
                 "fl_resid_vs_fl_global (sharing premium)": lambda o: o["fl_resid"][0] - o["fl_global"][0],
                 "fl_resid_random_vs_fl_without (control)": lambda o: o["fl_resid_random"][0] - o["fl_without"][0],
                 "fl_resid_vs_fl_resid_random (consortium-specific gain)": lambda o: o["fl_resid"][0] - o["fl_resid_random"][0],
                 "missed_per_1000_top_decile exact_group minus original": lambda o: o["exact_group"][1] - o["original"][1],
                 "missed_per_1000_fixed_cut exact_group minus original": lambda o: o["exact_group"][2] - o["original"][2],
                 "missed_per_1000_top_decile fl_resid minus fl_global": lambda o: o["fl_resid"][1] - o["fl_global"][1],
                 "missed_per_1000_fixed_cut fl_resid minus fl_global": lambda o: o["fl_resid"][2] - o["fl_global"][2]}
        for name, f in pairs.items():
            d[f"all | {name}"] = mean_over(keys, f)
            for fam in ["region", "teaching", "bedsize"]: d[f"{fam} | {name}"] = mean_over([k for k in keys if fam_of[k] == fam], f)
        return d
    all_rows = np.arange(len(test_idx)); pt = contrasts(stats(all_rows, hosp_te)); boots = {k: [] for k in pt}; r = np.random.RandomState(0)
    log(f"eICU: bootstrap {NBOOT} draws over {len(hospitals)} held-out hospitals")
    for b in range(NBOOT):
        drawn = r.choice(hospitals, len(hospitals), True); rows = np.concatenate([h_rows[h] for h in drawn]); hs = np.concatenate([np.full(len(h_rows[h]), h) for h in drawn])
        for k, v in contrasts(stats(rows, hs)).items(): boots[k].append(v)
        if (b + 1) % 100 == 0: log(f"  draw {b + 1}")
    for k in pt: record(f"eICU | {k}", pt[k], boots[k], dict(unit="hospital cluster"))
    rg = np.random.RandomState(1); S0 = stats(all_rows, hosp_te); dvals = np.array([S0[k]["exact_group"][0] - S0[k]["exact_random"][0] for k in keys])
    record("eICU | all | exact_group_vs_exact_random, consortium-level bootstrap only (as published)", dvals.mean(), [rg.choice(dvals, len(dvals), True).mean() for _ in range(5000)])

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()
if os.environ.get("MEMBERSHIP_ONLY") == "1":
    log("membership only; intervals.json left untouched")
else:
    json.dump(dict(cfg=CFG, results=results, date=time.strftime("%Y-%m-%d %H:%M:%S"), device=DEVICE, torch=torch.__version__),
              open(f"{OUT}/intervals.json", "w"), indent=1)
    pd.DataFrame([dict(key=k, d=v["d"], lo=v["ci"][0], hi=v["ci"][1], n_boot=v["n_boot"]) for k, v in results.items()]).to_csv(f"{OUT}/intervals.csv", index=False)
    log(f"done, {len(results)} intervals in {OUT}/intervals.json")
