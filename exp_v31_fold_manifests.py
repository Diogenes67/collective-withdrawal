"""
exp_v31_fold_manifests.py. Patient-level manifests for every cross-fitting fold model, with a verification against the model.

The cross-fitting fold models (v17 on MIMIC-IV, v20 on eICU) were trained without one of two patient-level halves of a group's
training patients. The split is a deterministic function of a seed index, but the runs that produced the cached models used the
group's index in the list each process ran over, which was 0 for groups run one per process, so two candidate splits exist for
each cached model. v24 and v25 resolved the candidate by the in-sample minus out-of-fold AUROC gap. This script records the
resolution for every fold model: for both candidates it computes the model's AUROC on the half it would have been trained on
minus its AUROC on the half it would have excluded; a model fitted to one half fits that half better, so the true split shows a
positive gap and the wrong split a gap near zero. It writes the excluded patient identifiers of the resolved split for every
fold model (local manifest, never leaves the analysis machine), an aggregate manifest with both gaps and the checkpoint's
SHA-256, and checks the resolution against the index recorded in the v24 and v25 run records.

Env: ROOT (default ~/mimic), PART=mimic|eicu|both, SEEDS (MIMIC-IV seeds, default 0..9), DEVICE, QUICK=1.
Writes results/v31_fold_manifests/{manifest.csv, summary.txt, log.txt, folds/<dataset>/<model>.txt}.
"""
import os, json, time, csv, hashlib, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; PART = os.environ.get("PART", "both")
SEEDS = [0] if QUICK else [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]; ESEEDS = [0] if QUICK else [0, 1, 2]
OUT = f"{R}/v31_fold_manifests" + ("_quick" if QUICK else ""); os.makedirs(f"{OUT}/folds/mimic", exist_ok=True); os.makedirs(f"{OUT}/folds/eicu", exist_ok=True)
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; part {PART}; MIMIC seeds {SEEDS}; eICU seeds {ESEEDS}; out {OUT}")

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def load(path, d):
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); m.eval(); return m
def logits(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(m(torch.tensor(Xa[i:i + 65536], device=DEVICE)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read())
    return h.hexdigest()
rows = []

def resolve(dataset, key, tag, true_idx, s, ckpt_dir, tr, pid, X, y, recorded, names=None):
    """One group and seed: both candidate splits against both fold models; writes the resolved split's excluded patients.
    pid is an integer code per row; names (optional) maps a code back to the identifier written to the manifest."""
    cands = sorted({true_idx, 0}); gaps = {}
    z = {k: logits(load(f"{ckpt_dir}/pooled_minus_{tag}_f{k}_s{s}.pt", X.shape[1]), X[tr]) for k in range(2)}
    splits = {}
    for c in cands:
        r = np.random.RandomState(3000 + c + 31 * s); u = np.unique(pid[tr]); r.shuffle(u); folds = np.array_split(u, 2); splits[c] = folds
        gaps[c] = [auc(y[tr][~np.isin(pid[tr], fp)], z[k][~np.isin(pid[tr], fp)]) - auc(y[tr][np.isin(pid[tr], fp)], z[k][np.isin(pid[tr], fp)]) for k, fp in enumerate(folds)]
    chosen = max(cands, key=lambda c: float(np.mean(gaps[c]))); alt = [c for c in cands if c != chosen]; alt = alt[0] if alt else None
    m_c = float(np.mean(gaps[chosen])); m_a = float(np.mean(gaps[alt])) if alt is not None else float("nan")
    decisive = (alt is None) or (m_c - m_a >= 0.01)  # resolution is by the mean over the two folds, as in v24 and v25; a 0.01 AUROC margin is the check
    for k in range(2):
        fp = splits[chosen][k]; excluded = np.sort(fp) if names is None else np.sort(names[fp]); n_rows = int(np.isin(pid[tr], fp).sum())
        with open(f"{OUT}/folds/{dataset}/{tag}_s{s}_f{k}.txt", "w") as f: f.write("\n".join(str(p) for p in excluded) + "\n")
        g_c = gaps[chosen][k]; g_a = gaps[alt][k] if alt is not None else float("nan")
        rows.append(dict(dataset=dataset, group=key, seed=s, fold=k, seed_index=chosen, candidates=";".join(map(str, cands)), n_excluded_patients=int(len(fp)), n_excluded_rows=n_rows,
                         gap_resolved=g_c, gap_alternative=g_a, mean_gap_resolved=m_c, mean_gap_alternative=m_a, decisive=decisive,
                         recorded_index=recorded, agrees_with_record=(recorded is None) or (recorded == chosen), checkpoint_sha256=sha256(f"{ckpt_dir}/pooled_minus_{tag}_f{k}_s{s}.pt")))
    log(f"  {dataset} {key} s{s}: resolved index {chosen} (recorded {recorded}); mean gap resolved {m_c:+.4f}" + (f", alternative {m_a:+.4f}" if alt is not None else ", single candidate"))

def run_mimic():
    V17 = f"{R}/v17_crossfit"; V24 = f"{R}/v24_symmetric_mimic"
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]; feat_lab = [c for c in feat if c.startswith("lab_")]
    y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    Xs = df[feat].astype(float)
    for c in feat_lab: Xs[c + "_missing"] = Xs[c].isna().astype(float); Xs[c] = Xs[c].fillna(Xs[c].median())
    Xs = Xs.values.astype(np.float32); mu0, sd0 = Xs[is_train].mean(0), Xs[is_train].std(0) + 1e-6; Xs = (Xs - mu0) / sd0
    pat_mean = pd.DataFrame(Xs).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    groups = []
    for c in range(25):  # same list and order as v17 and v24, so the group index is theirs
        members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
        if len(set(sid[f_tr])) < 300 or len(set(sid[f_te])) < 50: continue
        groups.append(dict(name=f"c{c:02d}", f_tr=f_tr))
    if QUICK: groups = groups[:2]
    log(f"MIMIC: {len(groups)} groups")
    for gi, g in enumerate(groups):
        key = f"community_trainfit/{g['name']}"; tag = f"community_trainfit__{g['name']}"; tr = np.where(g["f_tr"])[0]
        for s in SEEDS:
            rec = f"{V24}/runs/{tag}__s{s}.json"; recorded = json.load(open(rec)).get("fold_seed_index") if os.path.exists(rec) else None
            resolve("mimic", key, tag, gi, s, f"{V17}/ckpt", tr, sid, Xs, y, recorded)

def run_eicu():
    V20 = f"{R}/v20_eicu_protocol"; V25 = f"{R}/v25_symmetric_eicu"
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
    META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
    feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
    for c in Xdf.columns[Xdf.isna().any()]: Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32); pid = df.uniquepid.values; hosp_id = df.hospitalid.values
    # uniquepid is a string; np.unique and np.isin on string arrays are slow, so rows carry an integer code with the same sort
    # order as the strings. np.unique(code[tr]) is then the same sequence as np.unique(pid[tr]), and the seeded shuffle gives v20's folds.
    pid = np.asarray(pid, dtype=object); names = np.unique(pid); code = np.searchsorted(names, pid); assert np.array_equal(names[code], pid)
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); groups = []
    for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
        for v in sorted(hattr[col].dropna().unique(), key=str):
            hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
            if f_tr.sum() < 800 or f_te.sum() < 100: continue
            groups.append(dict(key=f"{fam}_{str(v).replace(' ', '_').replace('/', '_')}", f_tr=f_tr))
    if QUICK: groups = groups[:2]
    log(f"eICU: {len(groups)} consortia")
    for gi, g in enumerate(groups):
        tr = np.where(g["f_tr"])[0]
        for s in ESEEDS:
            rec = f"{V25}/runs/{g['key']}__s{s}.json"; recorded = json.load(open(rec)).get("fold_seed_index") if os.path.exists(rec) else None
            resolve("eicu", g["key"], g["key"], gi, s, f"{V20}/ckpt", tr, code, X, y, recorded, names)

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()
with open(f"{OUT}/manifest.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
S = [f"v31 fold manifests ({time.strftime('%Y-%m-%d %H:%M')}): {len(rows)} fold models"]
for ds in ["mimic", "eicu"]:
    rr = [r for r in rows if r["dataset"] == ds]
    if not rr: continue
    two = [r for r in rr if ";" in r["candidates"]]; rec = [r for r in rr if r["recorded_index"] is not None]
    alt = (f"alternative gap {min(r['gap_alternative'] for r in two):+.4f} to {max(r['gap_alternative'] for r in two):+.4f}; " if two else "")
    S.append(f"  {ds}: {len(rr)} fold models ({len(two)} with two candidate splits); resolved gap {min(r['gap_resolved'] for r in rr):+.4f} to {max(r['gap_resolved'] for r in rr):+.4f}; " + alt +
             f"decisive {sum(r['decisive'] for r in rr)}/{len(rr)}; agree with the recorded index {sum(r['agrees_with_record'] for r in rec)}/{len(rec)}; "
             f"index 0 used in {sum(r['seed_index'] == 0 and ';' in r['candidates'] for r in rr)} of the two-candidate models")
    bad = [r for r in rr if r["fold"] == 0 and (not r["decisive"] or not r["agrees_with_record"])]
    for r in bad: S.append(f"    CHECK {r['group']} s{r['seed']}: mean gap resolved {r['mean_gap_resolved']:+.4f} alternative {r['mean_gap_alternative']:+.4f} index {r['seed_index']} recorded {r['recorded_index']}")
open(f"{OUT}/summary.txt", "w").write("\n".join(S) + "\n"); print("\n" + "\n".join(S)); log(f"done, manifests in {OUT}")
