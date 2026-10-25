"""
exp_v19_protocol.py. The confinement protocol as a developer would run it, with the component chosen by validation (review 2,
the recommendation). MIMIC-IV, 25 clustered groups x 3 seeds.

For each group and seed: the strict retained-only base from v16 (preprocessing and weights fitted without the group). The
group's training patients are split 80/20 by patient; three component types (residual logistic on [base logit, features],
logistic head on the 128-d penultimate representation, fine-tuned copy of the base with early stopping) are fitted on the
80% and scored on the 20% by log loss; the type with the lowest validation log loss is chosen, refitted on all of the group's
training patients, and scored on the group's held-out patients. Nothing in the choice touches held-out data. Comparator: the
best pooled system from v17, the pooled model plus a cross-fitted residual component (same held-out patients; predictions
read from v17's cache). Two-stage bootstrap (groups, then patients; 1,000 draws).

Env: SEEDS (comma list of training seeds, default 0,1,2), SUMMARY=0 (stop before the summary and bootstrap stage), ROOT, QUICK=1, DEVICE, ONLY_GROUPS, NBOOT. Reads derived/cohort_latest.parquet, hosp/diagnoses_icd.csv.gz,
results/v5_partitions, results/v16_strict/ckpt, results/v17_crossfit/preds. Writes results/v19_protocol/{runs, preds, summary.csv, intervals.json}.
"""
import os, json, time, glob, hashlib, copy
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
OUT = f"{ROOT}/results/v19_protocol" + ("_quick" if QUICK else "")
V16 = f"{ROOT}/results/v16_strict" + ("_quick" if QUICK else ""); V17 = f"{ROOT}/results/v17_crossfit" + ("_quick" if QUICK else "")
for d in ["runs", "preds"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[int(x) for x in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else ([0] if QUICK else [0, 1, 2]), epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_top=100, min_patients=300,
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           n_boot=int(os.environ.get("NBOOT", 50 if QUICK else 1000)), full_lr=1e-4, full_batch=256, full_max_epochs=30, full_patience=5, full_val_frac=0.2, select_val_frac=0.2, version="v19.0")
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


# ---------------------------------------------------------------- components
def finetune_copy(base, Xr, rows, seed):
    r = np.random.RandomState(seed); ups = np.unique(sid[rows]); r.shuffle(ups); val_p = set(ups[: max(1, int(CFG["full_val_frac"] * len(ups)))])
    va = np.isin(sid[rows], list(val_p)); tr_ = ~va
    Xtr, ytr, Xva, yva = Xr[rows][tr_], y[rows][tr_], Xr[rows][va], y[rows][va]
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
def fit_component(kind, base, Xr, rows, seed):
    """Returns a predict(rows) -> probabilities function for the given component type fitted on rows."""
    if kind == "residual":
        f, _ = fit_adapter(np.column_stack([logits(base, Xr[rows]), Xr[rows]]), y[rows], sid[rows], offset=True)
        return lambda rr: f(np.column_stack([logits(base, Xr[rr]), Xr[rr]]))
    if kind == "head":
        f, _ = fit_adapter(penult(base, Xr[rows]), y[rows], sid[rows]); return lambda rr: f(penult(base, Xr[rr]))
    if kind == "finetuned":
        m = finetune_copy(base, Xr, rows, seed); return lambda rr: sigmoid(logits(m, Xr[rr]))
KINDS = ["residual", "head", "finetuned"]

# ---------------------------------------------------------------- runs
originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt", Xs.shape[1]) for s in CFG["seeds"]}
ARMS = ["pooled", "pooled_cf", "strict_base", "chosen", "residual", "head", "finetuned"]
for gi, g in enumerate(groups):
    tag = g["key"].replace("/", "__"); te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
    fit_rows = is_train & ~g["mem"]
    if all(os.path.exists(f"{OUT}/runs/{tag}__s{s}.json") for s in CFG["seeds"]): continue
    tg = time.time(); Xr, vocab, n_const = build_X(fit_rows); log(f"{g['key']}: strict pipeline rebuilt ({time.time()-tg:.0f}s)")
    for s in CFG["seeds"]:
        path = f"{OUT}/runs/{tag}__s{s}.json"
        if os.path.exists(path): continue
        ts = time.time(); base = load(f"{V16}/ckpt/strict_{tag}_s{s}.pt", Xr.shape[1])
        Z17 = np.load(f"{V17}/preds/{tag}__s{s}.npz"); assert np.array_equal(Z17["pid"], pid), "v17 held-out order differs"
        # selection split, patient-level, inside the group's training patients only
        r = np.random.RandomState(4000 + gi + 31 * s); ups = np.unique(sid[tr]); r.shuffle(ups); sel_p = set(ups[: max(2, int(CFG["select_val_frac"] * len(ups)))])
        sel = tr[np.isin(sid[tr], list(sel_p))]; fit = tr[~np.isin(sid[tr], list(sel_p))]
        val_ll, val_auc = {}, {}
        for k in KINDS:
            pr = fit_component(k, base, Xr, fit, s)(sel); val_ll[k] = float(log_loss(y[sel], np.clip(pr, 1e-6, 1 - 1e-6), labels=[0, 1])); val_auc[k] = auc(y[sel], pr)
        base_ll = float(log_loss(y[sel], np.clip(sigmoid(logits(base, Xr[sel])), 1e-6, 1 - 1e-6), labels=[0, 1]))
        chosen = min(val_ll, key=val_ll.get)
        p = {"pooled": Z17["pooled"], "pooled_cf": Z17["pooled_resid_crossfit"], "strict_base": sigmoid(logits(base, Xr[te]))}
        for k in KINDS: p[k] = fit_component(k, base, Xr, tr, s)(te)
        p["chosen"] = p[chosen]
        np.savez_compressed(f"{OUT}/preds/{tag}__s{s}.npz", y=yy, pid=pid, k_flag=int(Z17["k_flag"]), **p)
        rec = dict(group=g["key"], seed=s, n_train_patients=g["n_tr_p"], n_test_patients=g["n_te_p"], selection=dict(n_fit_patients=int(len(ups) - len(sel_p)), n_select_patients=len(sel_p),
                   val_logloss=val_ll, val_auroc=val_auc, base_val_logloss=base_ll, chosen=chosen),
                   arms={a: dict(auroc=auc(yy, p[a]), ece=ece(yy, p[a])) for a in ARMS}, cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - ts)
        save_json(path, rec); a = rec["arms"]
        log(f"  s{s}: chose {chosen} (val logloss " + ", ".join(f"{k} {v:.4f}" for k, v in val_ll.items()) + f"; base {base_ll:.4f}) | held-out AUROC pooled {a['pooled']['auroc']:.4f} pooled+cf {a['pooled_cf']['auroc']:.4f} "
            f"base {a['strict_base']['auroc']:.4f} chosen {a['chosen']['auroc']:.4f} [res {a['residual']['auroc']:.4f} head {a['head']['auroc']:.4f} ft {a['finetuned']['auroc']:.4f}] ({time.time()-ts:.0f}s)")

if os.environ.get("SUMMARY", "1") == "0": log("SUMMARY=0; stopping before the summary stage"); raise SystemExit(0)
# ---------------------------------------------------------------- summary from disk
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs for a, m in r["arms"].items()]
S = pd.DataFrame(rows).groupby(["group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="group", columns="arm", values="auroc"); ch = pd.Series({r["group"]: r["selection"]["chosen"] for r in recs}).groupby(level=0).agg(lambda x: ",".join(x))
print("\nAUROC on each group's held-out patients, seed means"); print(pd.concat([W[ARMS].round(4), ch.rename("chosen per seed")], axis=1).to_string())
print("\nchoices:", pd.Series([r["selection"]["chosen"] for r in recs]).value_counts().to_dict())
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); g, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(g.replace("__", "/"), int(s_))] = {k: Z[k] for k in Z.files}
seeds = sorted(set(s_ for _, s_ in P)); names = sorted(set(g for g, _ in P if all((g, s_) in P for s_ in seeds)))
pairs = {"chosen component minus pooled + cross-fitted component (protocol vs best pooled)": ("chosen", "pooled_cf"),
         "chosen component minus pooled": ("chosen", "pooled"), "chosen component minus strict base": ("chosen", "strict_base"),
         "residual minus pooled + cross-fitted component": ("residual", "pooled_cf"), "head minus pooled + cross-fitted component": ("head", "pooled_cf"),
         "fine-tuned minus pooled + cross-fitted component": ("finetuned", "pooled_cf"), "strict base minus pooled": ("strict_base", "pooled")}
def stats(g, idx):
    o = {}
    for a in ARMS:
        o[a] = (np.nanmean([fast_auc(P[(g, s_)]["y"][idx], P[(g, s_)][a][idx]) for s_ in seeds]),
                np.mean([missed_at_k(P[(g, s_)]["y"][idx], P[(g, s_)][a][idx], max(1, int(round(int(P[(g, s_)]["k_flag"]) * len(idx) / len(P[(g, s_)]["y"]))))) for s_ in seeds]))
    return o
def contrasts(st):
    d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
    d["events unflagged per 1,000 at matched flag rate, chosen minus pooled + cross-fitted component"] = np.mean([o["chosen"][1] - o["pooled_cf"][1] for o in st])
    d["events unflagged per 1,000 at matched flag rate, chosen minus pooled"] = np.mean([o["chosen"][1] - o["pooled"][1] for o in st])
    return d
ISO = ["community_trainfit/c02", "community_trainfit/c05", "community_trainfit/c12", "community_trainfit/c15", "community_trainfit/c20"]
if names:
    def run_boot(subset, label):
        pt = contrasts([stats(g, np.arange(len(P[(g, seeds[0])]["y"]))) for g in subset]); pat_idx = {}
        for g in subset:
            u, inv = np.unique(P[(g, seeds[0])]["pid"], return_inverse=True); pat_idx[g] = (u, [np.where(inv == i)[0] for i in range(len(u))])
        r = np.random.RandomState(0); boots = {k: [] for k in pt}
        for b in range(CFG["n_boot"]):
            st = []
            for g in r.choice(subset, len(subset), True):
                u, gidx = pat_idx[g]; idx = np.concatenate([gidx[j] for j in r.choice(len(u), len(u), True)]); st.append(stats(g, idx))
            for k, v in contrasts(st).items(): boots[k].append(v)
        res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(subset), n_boot=CFG["n_boot"]) for k in pt}
        lw = {k: int(sum((W.loc[g, a] - W.loc[g, b]) < 0 for g in subset)) for k, (a, b) in pairs.items()}
        print(f"\n== {label}: {len(subset)} groups, two-stage bootstrap ({CFG['n_boot']} draws)")
        for k, v in res.items(): print(f"  {k:95s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(subset)}" if k in lw else ""))
        return dict(results=res, lower_in=lw)
    out = {"all": run_boot(names, "all clustered groups")}
    iso = [g for g in names if g in ISO]; rest = [g for g in names if g not in ISO]
    if len(iso) >= 2: out["most_isolated"] = run_boot(iso, "five most isolated groups")
    if len(rest) >= 2: out["others"] = run_boot(rest, "other groups")
    save_json(f"{OUT}/intervals.json", dict(**out, date=time.strftime("%Y-%m-%d %H:%M:%S")))
log(f"done, results in {OUT}")
