"""
exp_v33_strict_multigroup.py. Several groups withdrawn at once, on the retained-only pipeline with validation-chosen components
(review of v7, item 3). One common base is trained without a set of groups: the whole shared pipeline (diagnosis vocabulary,
imputation, scaling) and the base are refitted on the training rows outside every group in the set, as in exp_v16_strict.py
for one group. Each withdrawing group then gets its own component on that base, the type chosen by log loss on a held-back
fifth of its training patients and refitted on all of them, as in exp_v19_protocol.py. Sets: the 5, 10 and 15 most isolated
groups (isolation ratio from the v5 records) and 5, 10 and 15 random groups (the random five of exp_v14_commonbase.py, extended
with a seeded draw). Each group is compared, on its held-out patients, with the pooled model, its single-exclusion confined
system (v19), its adapted pooled comparator (v24) and its single-exclusion strict base; the common base is also scored on the
held-out patients outside the set against the pooled model. Provenance asserts that no row of any group in the set and no
held-out row entered the pipeline or the base; the base's SHA-256 is recorded.

Env: ROOT (default ~/mimic), SEEDS (default 0,1,2), SIZES (default 5,10,15), FAMILIES (default most_isolated,random), DEVICE,
QUICK=1 (sets of two groups, two epochs, 30 draws), NBOOT. Resumable: cached bases and finished runs are skipped.
Writes results/v33_strict_multigroup/{runs, preds, ckpt, summary.csv, intervals.json, summary.txt, log.txt}.
"""
import os, json, time, glob, copy, hashlib, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"
V5 = f"{R}/v5_partitions"; V14 = f"{R}/v14_commonbase"; V19 = f"{R}/v19_protocol"; V24 = f"{R}/v24_symmetric_mimic"
OUT = f"{R}/v33_strict_multigroup" + ("_quick" if QUICK else "")
for d in ["runs", "preds", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
SEEDS = [0] if QUICK else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
SIZES = [2] if QUICK else [int(x) for x in os.environ.get("SIZES", "5,10,15").split(",")]
FAMILIES = os.environ.get("FAMILIES", "most_isolated,random").split(",")
CFG = dict(seeds=SEEDS, sizes=SIZES, epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_top=100, min_patients=300,
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           full_lr=1e-4, full_batch=256, full_max_epochs=2 if QUICK else 30, full_patience=5, full_val_frac=0.2, select_val_frac=0.2,
           n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 1000)), version="v33.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; quick={QUICK}; seeds {SEEDS}; sizes {SIZES}; families {FAMILIES}; out {OUT}")

# ---------------------------------------------------------------- data, split and groups exactly as v5/v16
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
feat_dx = [c for c in feat if c.startswith("dx_")]; feat_other = [c for c in feat if not c.startswith("dx_")]; feat_lab = [c for c in feat if c.startswith("lab_")]
y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values; hadm = df.hadm_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test; test_idx = np.where(is_test)[0]
Xs = df[feat].astype(float)
for c in feat_lab: Xs[c + "_missing"] = Xs[c].isna().astype(float); Xs[c] = Xs[c].fillna(Xs[c].median())
Xs = Xs.values.astype(np.float32); mu0, sd0 = Xs[is_train].mean(0), Xs[is_train].std(0) + 1e-6; Xs = (Xs - mu0) / sd0
pat_mean = pd.DataFrame(Xs).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
groups = {}
for c in range(25):
    members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    ref = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
    if ref["n_train_patients"] != n_tr_p or ref["n_test_patients"] != n_te_p: raise SystemExit(f"c{c:02d}: membership differs from v5")
    groups[f"c{c:02d}"] = dict(name=f"c{c:02d}", gi=len(groups), f_tr=f_tr, f_te=f_te, mem=mem, n_tr_p=n_tr_p)
names = list(groups); log(f"{len(names)} clustered groups match the v5 records")
Rv5 = pd.read_csv(f"{V5}/results.csv"); Rv5 = Rv5[(Rv5.family == "community_trainfit") & (Rv5.model == "original")]
iso_rank = list(Rv5.groupby("group").isolation_ratio.mean().sort_values(ascending=False).index)
rand5 = json.load(open(f"{V14}/sets_B.json"))["random"] if os.path.exists(f"{V14}/sets_B.json") else ["c01", "c10", "c13", "c23", "c24"]
rr = np.random.RandomState(1); rest = [n for n in names if n not in rand5]; rr.shuffle(rest); rand_rank = list(rand5) + rest
SETS = {}
for k in SIZES:
    if "most_isolated" in FAMILIES: SETS[f"most_isolated_{k}"] = sorted(iso_rank[:k])
    if "random" in FAMILIES: SETS[f"random_{k}"] = sorted(rand_rank[:k])
json.dump(dict(sets=SETS, isolation_rank=iso_rank, random_rank=rand_rank), open(f"{OUT}/sets.json", "w"), indent=1)
log("sets: " + "; ".join(f"{k}: {','.join(v)}" for k, v in SETS.items()))
log("loading diagnoses")
dx = pd.read_csv(f"{ROOT}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}, usecols=["hadm_id", "icd_code", "icd_version"])
dx = dx[dx.hadm_id.isin(set(hadm))]; dx["code3"] = dx.icd_version.astype(str) + "_" + dx.icd_code.str[:3]; dx = dx[["hadm_id", "code3"]].drop_duplicates()
global_top = set(c[3:] for c in feat_dx)
def build_X(fit_rows):
    """Feature matrix whose vocabulary, imputation and scaling are fitted on fit_rows only (exp_v16_strict.py)."""
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

# ---------------------------------------------------------------- model, components and selection, as v16 and v19
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
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); m.eval(); return m
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
def finetune_copy(base, X, rows, seed):
    r = np.random.RandomState(seed); ups = np.unique(sid[rows]); r.shuffle(ups); val_p = set(ups[: max(1, int(CFG["full_val_frac"] * len(ups)))])
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
KINDS = ["residual", "head", "finetuned"]
def fit_component(kind, base, X, rows, seed):
    if kind == "residual":
        f = fit_adapter(np.column_stack([logits(base, X[rows]), X[rows]]), y[rows], sid[rows], offset=True)
        return lambda rr: f(np.column_stack([logits(base, X[rr]), X[rr]]))
    if kind == "head":
        f = fit_adapter(penult(base, X[rows]), y[rows], sid[rows]); return lambda rr: f(penult(base, X[rr]))
    m = finetune_copy(base, X, rows, seed); return lambda rr: sigmoid(logits(m, X[rr]))
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

# ---------------------------------------------------------------- runs
originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt", Xs.shape[1]) for s in SEEDS}
ARMS = ["pooled", "single_strict_base", "single_confined", "adapted_pooled", "common_base", "common_confined"]
for sname, members in SETS.items():
    if all(os.path.exists(f"{OUT}/runs/{sname}__s{s}.json") for s in SEEDS): continue
    in_set = np.zeros(len(df), bool)
    for n in members: in_set |= groups[n]["mem"]
    fit_rows = is_train & ~in_set; others = test_idx[~in_set[test_idx]]
    tg = time.time(); Xr, vocab, n_const = build_X(fit_rows)
    assert not np.isin(sid[fit_rows], sid[in_set]).any(), "patient of a withdrawing group in the shared pipeline rows"
    assert not (fit_rows & is_test).any(), "held-out row in the shared pipeline rows"
    log(f"{sname}: {len(members)} groups, {int(in_set[is_train].sum())} of their training rows withdrawn; strict pipeline on {int(fit_rows.sum())} retained rows "
        f"({100 * fit_rows.sum() / is_train.sum():.1f}% of training); vocabulary overlap {len(set(vocab) & global_top)}/100; constant columns {n_const} ({time.time()-tg:.0f}s)")
    for s in SEEDS:
        path = f"{OUT}/runs/{sname}__s{s}.json"
        if os.path.exists(path): continue
        ts = time.time(); ck = f"{OUT}/ckpt/common_{sname}_s{s}.pt"
        if os.path.exists(ck): base = load(ck, Xr.shape[1]); trained = False
        else:
            base = train(Xr[fit_rows], y[fit_rows], s); torch.save(base.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck); trained = True
        sha_before = sha256(ck)
        oth = dict(n_rows=int(len(others)), pooled=auc(y[others], sigmoid(logits(originals[s], Xs[others]))), common_base=auc(y[others], sigmoid(logits(base, Xr[others]))))
        log(f"  {sname} s{s}: base {'trained' if trained else 'loaded'}; others' AUROC pooled {oth['pooled']:.4f} common base {oth['common_base']:.4f} ({time.time()-ts:.0f}s)")
        per = {}
        for n in members:
            tn = time.time(); g = groups[n]; tag = f"community_trainfit__{n}"; te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
            Z19 = np.load(f"{V19}/preds/{tag}__s{s}.npz"); Z24 = np.load(f"{V24}/preds/{tag}__s{s}.npz")
            assert np.array_equal(Z19["pid"], pid) and np.array_equal(Z24["pid"], pid), "held-out order differs from v19/v24"
            r = np.random.RandomState(4000 + g["gi"] + 31 * s); ups = np.unique(sid[tr]); r.shuffle(ups); sel_p = set(ups[: max(2, int(CFG["select_val_frac"] * len(ups)))])
            sel = tr[np.isin(sid[tr], list(sel_p))]; fit = tr[~np.isin(sid[tr], list(sel_p))]
            val_ll = {}
            for k in KINDS:
                pr = fit_component(k, base, Xr, fit, s)(sel); val_ll[k] = float(log_loss(y[sel], np.clip(pr, 1e-6, 1 - 1e-6), labels=[0, 1]))
            chosen = min(val_ll, key=val_ll.get)
            p = dict(pooled=Z19["pooled"], single_strict_base=Z19["strict_base"], single_confined=Z19["chosen"], adapted_pooled=Z24["pooled_cf_chosen"],
                     common_base=sigmoid(logits(base, Xr[te])), common_confined=fit_component(chosen, base, Xr, tr, s)(te))
            np.savez_compressed(f"{OUT}/preds/{sname}__{tag}__s{s}.npz", y=yy, pid=pid, k_flag=int(Z19["k_flag"]), **p)
            per[n] = dict(n_train_patients=g["n_tr_p"], chosen=chosen, val_logloss=val_ll, single_chosen=str(json.load(open(f"{V19}/runs/{tag}__s{s}.json"))["selection"]["chosen"]) if os.path.exists(f"{V19}/runs/{tag}__s{s}.json") else None,
                          arms={a: dict(auroc=auc(yy, p[a]), ece=ece(yy, p[a])) for a in ARMS})
            a = per[n]["arms"]
            log(f"    {n}: chose {chosen} | pooled {a['pooled']['auroc']:.4f} single base {a['single_strict_base']['auroc']:.4f} single confined {a['single_confined']['auroc']:.4f} adapted {a['adapted_pooled']['auroc']:.4f} "
                f"common base {a['common_base']['auroc']:.4f} common confined {a['common_confined']['auroc']:.4f} ({time.time()-tn:.0f}s)")
        sha_after = sha256(ck)
        rec = dict(set=sname, members=members, seed=s, n_fit_rows=int(fit_rows.sum()), share_of_training=float(fit_rows.sum() / is_train.sum()), withdrawn_training_rows=int(in_set[is_train].sum()),
                   group_rows_in_pipeline=0, heldout_rows_in_pipeline=0, vocab_overlap=len(set(vocab) & global_top), constant_columns=n_const, base_sha256=sha_before, base_sha256_after=sha_after,
                   base_unchanged=sha_before == sha_after, others=oth, groups=per, cfg=CFG, device=DEVICE, torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
        save_json(path, rec)
if os.environ.get("SUMMARY", "1") == "0": log("SUMMARY=0; stopping before the summary stage"); raise SystemExit(0)

# ---------------------------------------------------------------- summary and two-stage bootstrap per set
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(set=r["set"], seed=r["seed"], group=n, arm=a, **m) for r in recs for n, per in r["groups"].items() for a, m in per["arms"].items()]
S = pd.DataFrame(rows).groupby(["set", "group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); rk = np.empty(len(pv)); rk[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (rk[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
pairs = {"common base + chosen component minus pooled": ("common_confined", "pooled"),
         "common base + chosen component minus single-exclusion confined system": ("common_confined", "single_confined"),
         "common base + chosen component minus adapted pooled comparator": ("common_confined", "adapted_pooled"),
         "common base minus single-exclusion strict base": ("common_base", "single_strict_base"),
         "common base minus pooled": ("common_base", "pooled"),
         "deletion of the component: common base + component minus common base": ("common_confined", "common_base")}
LINES = [f"v33 strict multi-group withdrawal ({time.strftime('%Y-%m-%d %H:%M')}); seeds {SEEDS}"]; OUTJ = {}
for sname, members in SETS.items():
    rs = [r for r in recs if r["set"] == sname]
    if len(rs) < len(SEEDS): LINES.append(f"== {sname}: {len(rs)} of {len(SEEDS)} seeds done; skipped"); continue
    seeds = sorted(r["seed"] for r in rs); P = {}
    for r in rs:
        for n in members:
            Z = np.load(f"{OUT}/preds/{sname}__community_trainfit__{n}__s{r['seed']}.npz"); P[(n, r["seed"])] = {k: Z[k] for k in Z.files}
    W = S[S.set == sname].pivot_table(index="group", columns="arm", values="auroc")
    def stats(n, idx):
        o = {}
        for a in ARMS:
            o[a] = (np.nanmean([fast_auc(P[(n, s_)]["y"][idx], P[(n, s_)][a][idx]) for s_ in seeds]),
                    np.mean([missed_at_k(P[(n, s_)]["y"][idx], P[(n, s_)][a][idx], max(1, int(round(int(P[(n, s_)]["k_flag"]) * len(idx) / len(P[(n, s_)]["y"]))))) for s_ in seeds]))
        return o
    def contrasts(st):
        d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
        d["events unflagged per 1,000 at matched flag rate, common base + component minus pooled"] = np.mean([o["common_confined"][1] - o["pooled"][1] for o in st])
        d["events unflagged per 1,000 at matched flag rate, common base + component minus single-exclusion confined"] = np.mean([o["common_confined"][1] - o["single_confined"][1] for o in st]); return d
    pt = contrasts([stats(n, np.arange(len(P[(n, seeds[0])]["y"]))) for n in members]); pat = {}
    for n in members:
        u, inv = np.unique(P[(n, seeds[0])]["pid"], return_inverse=True); pat[n] = [np.where(inv == i)[0] for i in range(len(u))]
    r_ = np.random.RandomState(0); boots = {k: [] for k in pt}
    for b in range(CFG["n_boot"]):
        st = [stats(n, np.concatenate([pat[n][j] for j in r_.choice(len(pat[n]), len(pat[n]), True)])) for n in r_.choice(members, len(members), True)]
        for k, v in contrasts(st).items(): boots[k].append(v)
    res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(members), n_boot=CFG["n_boot"]) for k in pt}
    lw = {k: int(sum((W.loc[n, a] - W.loc[n, b]) < 0 for n in members)) for k, (a, b) in pairs.items()}
    oth = dict(pooled=float(np.mean([r["others"]["pooled"] for r in rs])), common_base=float(np.mean([r["others"]["common_base"] for r in rs])), n_rows=rs[0]["others"]["n_rows"])
    ch = pd.Series([per["chosen"][0].upper() for r in sorted(rs, key=lambda r: r["seed"]) for n, per in r["groups"].items()], index=[n for r in sorted(rs, key=lambda r: r["seed"]) for n in r["groups"]]).groupby(level=0).agg("".join)
    LINES.append(f"== {sname}: {', '.join(members)}; retained {100 * rs[0]['share_of_training']:.1f}% of training rows; vocabulary overlap {rs[0]['vocab_overlap']}/100; base unchanged {all(r['base_unchanged'] for r in rs)}; "
                 f"others' AUROC ({oth['n_rows']} rows) pooled {oth['pooled']:.4f} common base {oth['common_base']:.4f}")
    LINES.append(pd.concat([W[ARMS].round(4), ch.rename("type")], axis=1).to_string())
    for k, v in res.items(): LINES.append(f"  {k:100s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(members)}" if k in lw else ""))
    OUTJ[sname] = dict(members=members, results=res, lower_in=lw, others=oth, choices=ch.to_dict(), share_of_training=rs[0]["share_of_training"])
save_json(f"{OUT}/intervals.json", dict(**OUTJ, cfg=CFG, date=time.strftime("%Y-%m-%d %H:%M:%S")))
open(f"{OUT}/summary.txt", "w").write("\n".join(LINES) + "\n"); print("\n" + "\n".join(LINES)); log(f"done, results in {OUT}")
