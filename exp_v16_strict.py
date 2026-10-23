"""
exp_v16_strict.py. Contribution confinement with a strictly retained-only shared pipeline (MIMIC-IV).

Review point 1. In v11.1 the shared base excluded the group from weight fitting, but the diagnosis vocabulary, imputation and
feature scaling were fitted once on the whole cohort, so the group's records shaped the shared pipeline. Here, for each group
and seed, everything shared is refitted on the retained training admissions only (the group's patients and all held-out
patients excluded): the top-100 three-character diagnosis vocabulary, any imputation medians, and the scaling. The base is
retrained on those rows with the v5 recipe (fixed 12 epochs, no validation set, so model selection uses no data). The
group's residual component (v14.1 adapter), representation head and the control component (fitted on an equal number of
other patients' held-out records) are refitted on the strict base. The pooled model is the v5 original with whole-training
preprocessing, which is legitimate because it is the model that uses everyone.

Every run writes a provenance record: rows and patients used for preprocessing, base training, model selection, component
fitting and evaluation, with an assertion that no patient of the group is in the preprocessing or base-training rows, and
the base's SHA-256. Predictions on the group's held-out patients are cached locally for the two-stage bootstrap
(groups, then patients within groups; 1,000 draws), which the summary runs from disk.

Env: ROOT (default ~/mimic), QUICK=1, DEVICE, ONLY_GROUPS (comma list, e.g. community_trainfit/c05), NBOOT.
Reads derived/cohort_latest.parquet, hosp/diagnoses_icd.csv.gz, results/v5_partitions/{ckpt,runs}.
Writes results/v16_strict/{runs/*.json, preds/*.npz, ckpt/, summary.csv, intervals.json, provenance.csv}.
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
OUT = f"{ROOT}/results/v16_strict" + ("_quick" if QUICK else "")
for d in ["runs", "preds", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_top=100, min_patients=300,
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           n_boot=int(os.environ.get("NBOOT", 50 if QUICK else 1000)), version="v16.0")
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
race = modal_label("race_group"); labels[("race", "native")] = set(race.index[race == "native"])
eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
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
log("loading diagnoses")
dx = pd.read_csv(f"{ROOT}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}, usecols=["hadm_id", "icd_code", "icd_version"])
dx = dx[dx.hadm_id.isin(set(hadm))]; dx["code3"] = dx.icd_version.astype(str) + "_" + dx.icd_code.str[:3]; dx = dx[["hadm_id", "code3"]].drop_duplicates()
global_top = set(c[3:] for c in feat_dx)

def build_X(fit_rows):
    """Feature matrix whose vocabulary, imputation and scaling are fitted on fit_rows only. Constant columns in fit_rows keep
    their centring and are not rescaled (sd set to 1), so a feature the retained data never vary on cannot explode."""
    fit_hadm = set(hadm[fit_rows])
    top = dx[dx.hadm_id.isin(fit_hadm)].code3.value_counts().head(CFG["n_top"]).index
    wide = dx[dx.code3.isin(top)].assign(v=1).pivot_table(index="hadm_id", columns="code3", values="v", aggfunc="max", fill_value=0)
    wide = wide.reindex(columns=sorted(top), fill_value=0).reindex(hadm, fill_value=0)
    Xo = df[feat_other].astype(float).copy()
    for c in feat_lab:
        med = Xo.loc[fit_rows, c].median(); Xo[c + "_missing"] = Xo[c].isna().astype(float); Xo[c] = Xo[c].fillna(med)
    Xa = np.concatenate([Xo.values, wide.values.astype(float)], axis=1).astype(np.float32)
    mu, sd = Xa[fit_rows].mean(0), Xa[fit_rows].std(0); const = sd < 1e-8; sd = np.where(const, 1.0, sd)
    return ((Xa - mu) / sd).astype(np.float32), list(top), int(const.sum())

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
originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt", Xs.shape[1]) for s in CFG["seeds"]}
thr_sys = {s: float(np.percentile(sigmoid(logits(originals[s], Xs[is_test])), 90)) for s in CFG["seeds"]}
ARMS = ["pooled", "strict_base", "strict_resid", "strict_head", "strict_control"]
for gi, g in enumerate(groups):
    tag = g["key"].replace("/", "__"); te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
    fit_rows = is_train & ~g["mem"]
    if all(os.path.exists(f"{OUT}/runs/{tag}__s{s}.json") for s in CFG["seeds"]): continue
    tg = time.time(); Xr, vocab, n_const = build_X(fit_rows)
    assert not np.isin(sid[fit_rows], sid[g["f_tr"] | g["f_te"]]).any(), "group patient in the shared pipeline rows"
    assert not (fit_rows & is_test).any(), "held-out row in the shared pipeline rows"
    log(f"{g['key']}: strict pipeline on {int(fit_rows.sum())} retained rows; vocabulary overlap {len(set(vocab) & global_top)}/100; constant columns {n_const} ({time.time()-tg:.0f}s)")
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{tag}__s{s}.json"
        if os.path.exists(path): continue
        ts = time.time(); ck = f"{OUT}/ckpt/strict_{tag}_s{s}.pt"
        if os.path.exists(ck): base = load(ck, Xr.shape[1])
        else:
            base = train(Xr[fit_rows], y[fit_rows], s); torch.save(base.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        sha = sha256(ck)
        z_te = logits(base, Xr[te]); z_tr = logits(base, Xr[tr])
        p = {"pooled": sigmoid(logits(originals[s], Xs[te])), "strict_base": sigmoid(z_te)}
        f_res, C_res = fit_adapter(np.column_stack([z_tr, Xr[tr]]), y[tr], sid[tr], offset=True); p["strict_resid"] = f_res(np.column_stack([z_te, Xr[te]]))
        f_head, C_head = fit_adapter(penult(base, Xr[tr]), y[tr], sid[tr]); p["strict_head"] = f_head(penult(base, Xr[te]))
        nong = test_idx[~g["mem"][test_idx]]; rc = np.random.RandomState(500 + gi + 31 * s); ng_p = np.unique(sid[nong]); rc.shuffle(ng_p)
        ctrl = nong[np.isin(sid[nong], ng_p[: min(g["n_tr_p"], len(ng_p))])]
        f_ctl, C_ctl = fit_adapter(np.column_stack([logits(base, Xr[ctrl]), Xr[ctrl]]), y[ctrl], sid[ctrl], offset=True); p["strict_control"] = f_ctl(np.column_stack([z_te, Xr[te]]))
        assert sha256(ck) == sha, "base changed while fitting components"
        others = test_idx[~g["mem"][test_idx]]
        k_flag = int((p["pooled"] >= thr_sys[s]).sum())
        np.savez_compressed(f"{OUT}/preds/{tag}__s{s}.npz", y=yy, pid=pid, k_flag=k_flag, **p)
        rec = dict(group=g["key"], seed=s, n_train_patients=g["n_tr_p"], n_test_patients=g["n_te_p"], k_flag=k_flag,
                   arms={a: dict(auroc=auc(yy, p[a]), ece=ece(yy, p[a])) for a in ARMS},
                   others_auroc=dict(pooled=auc(y[others], sigmoid(logits(originals[s], Xs[others]))), strict_base=auc(y[others], sigmoid(logits(base, Xr[others])))),
                   adapter_C=dict(resid=C_res, head=C_head, control=C_ctl),
                   provenance=dict(preprocessing_rows=int(fit_rows.sum()), preprocessing_patients=int(len(np.unique(sid[fit_rows]))),
                                   group_patients_in_preprocessing=0, heldout_rows_in_preprocessing=0,
                                   base_training_rows=int(fit_rows.sum()), model_selection="fixed 12 epochs; no validation data",
                                   component_rows=int(len(tr)), component_patients=g["n_tr_p"], control_rows=int(len(ctrl)),
                                   evaluation_rows=int(len(te)), evaluation_patients=g["n_te_p"],
                                   vocabulary_overlap_with_whole_cohort=len(set(vocab) & global_top), constant_columns=n_const, base_sha256=sha),
                   cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
        save_json(path, rec)
        a = rec["arms"]; log(f"  s{s}: pooled {a['pooled']['auroc']:.4f} strict base {a['strict_base']['auroc']:.4f} +resid {a['strict_resid']['auroc']:.4f} "
                             f"+head {a['strict_head']['auroc']:.4f} +control {a['strict_control']['auroc']:.4f} ({time.time()-ts:.0f}s)")

# ---------------------------------------------------------------- summary from disk
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs for a, m in r["arms"].items()]
S = pd.DataFrame(rows).groupby(["group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
pd.DataFrame([dict(group=r["group"], seed=r["seed"], **r["provenance"]) for r in recs]).to_csv(f"{OUT}/provenance.csv", index=False)
W = S.pivot_table(index="group", columns="arm", values="auroc"); E = S.pivot_table(index="group", columns="arm", values="ece")
clu = [g for g in W.index if g.startswith("community_trainfit")]
print("\nAUROC on each group's held-out patients, seed means, strict retained-only shared pipeline")
print(W[ARMS].round(4).to_string())
# two-stage bootstrap over clustered groups, then patients
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); g, s = os.path.basename(f)[:-4].rsplit("__s", 1); P[(g.replace("__", "/"), int(s))] = {k: Z[k] for k in Z.files}
seeds = sorted(set(s for _, s in P)); names = [g for g in clu if all((g, s) in P for s in seeds)]
pairs = {"strict base minus pooled": ("strict_base", "pooled"), "strict base + residual minus pooled": ("strict_resid", "pooled"),
         "strict base + residual minus strict base": ("strict_resid", "strict_base"), "strict base + head minus strict base": ("strict_head", "strict_base"),
         "strict base + control minus strict base": ("strict_control", "strict_base"), "strict base + residual minus strict base + control (group-specific)": ("strict_resid", "strict_control")}
def stats(g, idx):
    o = {}
    for a in ARMS:
        o[a] = (np.nanmean([fast_auc(P[(g, s)]["y"][idx], P[(g, s)][a][idx]) for s in seeds]),
                np.mean([missed_at_k(P[(g, s)]["y"][idx], P[(g, s)][a][idx], max(1, int(round(int(P[(g, s)]["k_flag"]) * len(idx) / len(P[(g, s)]["y"]))))) for s in seeds]))
    return o
def contrasts(st):
    d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
    d["events unflagged per 1,000 at matched flag rate, strict base minus pooled"] = np.mean([o["strict_base"][1] - o["pooled"][1] for o in st])
    d["events unflagged per 1,000 at matched flag rate, strict base + residual minus pooled"] = np.mean([o["strict_resid"][1] - o["pooled"][1] for o in st])
    return d
if names:
    pt = contrasts([stats(g, np.arange(len(P[(g, seeds[0])]["y"]))) for g in names])
    pat_idx = {}
    for g in names:
        pidv = P[(g, seeds[0])]["pid"]; u, inv = np.unique(pidv, return_inverse=True); pat_idx[g] = (u, [np.where(inv == i)[0] for i in range(len(u))])
    r = np.random.RandomState(0); boots = {k: [] for k in pt}
    for b in range(CFG["n_boot"]):
        st = []
        for g in r.choice(names, len(names), True):
            u, gi_ = pat_idx[g]; idx = np.concatenate([gi_[j] for j in r.choice(len(u), len(u), True)]); st.append(stats(g, idx))
        for k, v in contrasts(st).items(): boots[k].append(v)
        if (b + 1) % 200 == 0: log(f"  bootstrap draw {b + 1}")
    res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(names), n_boot=CFG["n_boot"]) for k in pt}
    lw = {k: int(sum((W.loc[g, a] - W.loc[g, b]) < 0 for g in names)) for k, (a, b) in pairs.items()}
    print(f"\n== {len(names)} clustered groups, two-stage bootstrap ({CFG['n_boot']} draws)")
    for k, v in res.items(): print(f"  {k:85s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(names)}" if k in lw else ""))
    O = pd.DataFrame([dict(group=r_["group"], **r_["others_auroc"]) for r_ in recs]).groupby("group").mean()
    print(f"  others' AUROC, strict base minus pooled, mean over groups {(O.strict_base - O.pooled).mean():+.4f}")
    for g in [g for g in W.index if not g.startswith("community")]: print(f"  {g}: " + ", ".join(f"{a} {W.loc[g, a]:.4f}" for a in ARMS) + f"; ECE pooled {E.loc[g, 'pooled']:.4f} strict+resid {E.loc[g, 'strict_resid']:.4f}")
    save_json(f"{OUT}/intervals.json", dict(results=res, lower_in=lw, date=time.strftime("%Y-%m-%d %H:%M:%S")))
log(f"done, results in {OUT}")
