"""
exp_v14_commonbase.py. Two comparisons the review asked for, on MIMIC-IV, reusing the v5 caches.

Part A, equal opportunity to adapt (review point 4). The confined system was compared with an unadapted pooled model.
Here both shared models get the same local adaptation: a residual logistic component (and a representation head) is
cross-fitted on the community's held-out patients, split in two patient-level halves, so that neither shared model has
seen the fitting rows (the pooled model trained on the community's training rows, so those cannot be used for its
adapter). Predictions are out-of-fold on the whole held-out set. Arms: pooled, pooled+adapter, base (exact retrain
without the community), base+adapter. The contrast base+adapter minus pooled+adapter is the price of keeping the
community out of shared training once both systems can personalise. 25 communities, 3 seeds.

Part B, several communities confined at once (review point 3B). One shared base is trained without a set of five
communities together (set 1: the five most isolated; set 2: five drawn at random), each community gets its own
component fitted on its training rows as in v11, and each is compared with the single-exclusion result from v5/v11 and
with the pooled model. The base's performance on everyone outside the set is reported against the pooled model on the
same rows. 3 seeds, v5 training recipe (12 epochs, Adam 1e-3, batch 2048; MIMIC was not over-trained).

Env: ROOT (default ~/mimic), QUICK=1, NBOOT (patient bootstrap per community, default 200), DEVICE, PART=A|B|both.
Writes ROOT/results/v14_commonbase/{runs_A/*.json, runs_B/*.json, ckpt/, summary_A.csv, summary_B.csv, paired.json, log}.
Resume-safe: finished JSONs are skipped; bases are cached.
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
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); PART = os.environ.get("PART", "both")
V5 = f"{ROOT}/results/v5_partitions"
OUT = f"{ROOT}/results/v14_commonbase_1" + ("_quick" if QUICK else "")
for d in ["runs_A", "runs_B", "ckpt"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], epochs=2 if QUICK else 12, lr=1e-3, batch=2048, n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 200)),
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0, set_size=5, fixed_cut=0.3, version="v14.1")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; part {PART}; out {OUT}")

# ---------- data, identical to exp_v5 / v11
COHORT = f"{ROOT}/derived/cohort_latest.parquet"
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(COHORT); feat = [c for c in df.columns if c not in META]; X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd; test_idx = np.where(is_test)[0]
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
groups = {}
for c in range(25):
    members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    ref = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
    if ref["n_train_patients"] != len(set(sid[f_tr])) or ref["n_test_patients"] != len(set(sid[f_te])): raise SystemExit(f"c{c:02d}: membership differs from v5")
    groups[f"c{c:02d}"] = dict(f_tr=f_tr, f_te=f_te, n_tr_p=len(set(sid[f_tr])))
log("25 communities match the v5 records")

# ---------- model and helpers (as v5 / v11)
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
def load(path):
    if not os.path.exists(path): raise SystemExit(f"missing checkpoint {path}")
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
    """Fallback adapter: logistic on the first column only (the shared logit); monotone, so it leaves the ranking unchanged."""
    def __init__(self, Z, yy): self.m = LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, :1], yy)
    def predict_proba(self, Z): return self.m.predict_proba(Z[:, :1])
def fit_adapter(Z, yy, pid, seed, offset=False):
    """Ridge logistic adapter with C chosen by patient-grouped CV log loss. With offset=True the first column is the shared logit:
    it is scaled by logit_scale so the L2 penalty barely touches its coefficient, and a logit-only (Platt) candidate competes in the CV,
    so the adapted model cannot be chosen when it does worse in CV than leaving the shared ranking alone."""
    Z = Z.copy()
    if offset: Z[:, 0] *= CFG["logit_scale"]
    n_p = len(np.unique(pid)); grid = CFG["adapter_C_grid"]; chosen, cv = "platt" if offset else grid[len(grid) // 2], None
    if n_p >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pid)); cv = {}
        cands = (["platt"] if offset else []) + list(grid)
        for C in cands:
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = Platt(Z[tr], yy[tr]) if C == "platt" else LogisticRegression(C=C, max_iter=3000).fit(Z[tr], yy[tr])
                ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[C] = float(np.mean(ll))
        if cv: chosen = min(cv, key=cv.get)
    m = Platt(Z, yy) if chosen == "platt" else LogisticRegression(C=chosen, max_iter=3000).fit(Z, yy)
    class Wrapped:
        def predict_proba(self, Zn):
            Zn = Zn.copy()
            if offset: Zn[:, 0] *= CFG["logit_scale"]
            return m.predict_proba(Zn)
    return Wrapped(), chosen
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def boot_auc(yy, pp, pid):
    r = np.random.RandomState(1); ups = np.unique(pid); gi = {p: np.where(pid == p)[0] for p in ups}; vals = []
    for _ in range(CFG["n_boot"]): i = np.concatenate([gi[p] for p in r.choice(ups, len(ups), True)]); vals.append(auc(yy[i], pp[i]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
def cross_fit(z_full_logit, h_full, te, yy, pid, seed):
    """Two patient-level halves of the community's held-out rows; adapters fitted on one half, applied to the other."""
    r = np.random.RandomState(2000 + seed); ups = np.unique(pid); r.shuffle(ups); half = set(ups[: len(ups) // 2]); A = np.isin(pid, list(half)); B = ~A
    p_res, p_rep = np.zeros(len(te)), np.zeros(len(te)); Cs = []
    for fit, app in [(A, B), (B, A)]:
        ad, C = fit_adapter(np.column_stack([z_full_logit[fit], X[te][fit]]), yy[fit], pid[fit], seed, offset=True); p_res[app] = ad.predict_proba(np.column_stack([z_full_logit[app], X[te][app]]))[:, 1]; Cs.append(C)
        ad, C = fit_adapter(h_full[fit], yy[fit], pid[fit], seed); p_rep[app] = ad.predict_proba(h_full[app])[:, 1]
    return p_res, p_rep, Cs

names = sorted(groups)
originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt") for s in CFG["seeds"]}

# ============================================================ Part A: equal opportunity to adapt
def part_A():
    for n in names:
        g = groups[n]; te = np.where(g["f_te"])[0]; yy, pid = y[te], sid[te]
        for s in CFG["seeds"]:
            path = f"{OUT}/runs_A/{n}__s{s}.json"
            if os.path.exists(path): continue
            tg = time.time(); orig = originals[s]; base = load(f"{V5}/ckpt/exact_community_trainfit_{n}_s{s}.pt")
            z_o, h_o = logits(orig, X[te]), penult(orig, X[te]); z_b, h_b = logits(base, X[te]), penult(base, X[te])
            po_res, po_rep, Co = cross_fit(z_o, h_o, te, yy, pid, s); pb_res, pb_rep, Cb = cross_fit(z_b, h_b, te, yy, pid, s)
            arms = {"pooled": sigmoid(z_o), "pooled_resid": po_res, "pooled_rep": po_rep, "base": sigmoid(z_b), "base_resid": pb_res, "base_rep": pb_rep}
            res = {a: dict(auroc=auc(yy, p), auroc_ci=boot_auc(yy, p, pid), ece=ece(yy, p), brier=float(np.mean((p - yy) ** 2))) for a, p in arms.items()}
            save_json(path, dict(community=n, seed=s, n_test_patients=int(len(np.unique(pid))), n_test_rows=int(len(te)), adapter_C=dict(pooled=Co, base=Cb), arms=res, cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S")))
            log(f"A {n} s{s}: pooled {res['pooled']['auroc']:.4f} pooled+resid {res['pooled_resid']['auroc']:.4f} base {res['base']['auroc']:.4f} base+resid {res['base_resid']['auroc']:.4f} ({time.time()-tg:.0f}s)")

# ============================================================ Part B: several communities confined at once
def choose_sets():
    iso = None
    if os.path.exists(f"{V5}/results.csv"):
        R = pd.read_csv(f"{V5}/results.csv"); R = R[(R.family == "community_trainfit") & (R.model == "original")]
        iso = R.groupby("group").isolation_ratio.mean().sort_values(ascending=False)
    s1 = list(iso.index[: CFG["set_size"]]) if iso is not None else names[: CFG["set_size"]]
    r = np.random.RandomState(0); s2 = sorted(r.choice([n for n in names if n not in s1], CFG["set_size"], False).tolist())
    return {"most_isolated": sorted(s1), "random": s2}
def part_B():
    sets = choose_sets(); log(f"sets: {sets}"); json.dump(sets, open(f"{OUT}/sets_B.json", "w"))
    for sname, members in sets.items():
        in_set = np.zeros(len(df), bool)
        for n in members: in_set |= groups[n]["f_tr"] | groups[n]["f_te"]
        fit_rows = is_train & ~in_set; others_te = test_idx[~in_set[test_idx]]
        for s in CFG["seeds"]:
            path = f"{OUT}/runs_B/{sname}__s{s}.json"
            if os.path.exists(path): continue
            tg = time.time(); base = cached(f"base_{sname}_s{s}", lambda: train(X[fit_rows], y[fit_rows], s)); orig = originals[s]
            others = dict(pooled=auc(y[others_te], sigmoid(logits(orig, X[others_te]))), common_base=auc(y[others_te], sigmoid(logits(base, X[others_te]))), n_rows=int(len(others_te)))
            per = {}
            for n in members:
                g = groups[n]; te = np.where(g["f_te"])[0]; tr = np.where(g["f_tr"])[0]; yy, pid = y[te], sid[te]
                single = load(f"{V5}/ckpt/exact_community_trainfit_{n}_s{s}.pt")
                p_pool = sigmoid(logits(orig, X[te])); p_single = sigmoid(logits(single, X[te])); p_common = sigmoid(logits(base, X[te]))
                ad, _ = fit_adapter(np.column_stack([logits(single, X[tr]), X[tr]]), y[tr], sid[tr], s, offset=True); p_single_res = ad.predict_proba(np.column_stack([logits(single, X[te]), X[te]]))[:, 1]
                ad, _ = fit_adapter(np.column_stack([logits(base, X[tr]), X[tr]]), y[tr], sid[tr], s, offset=True); p_common_res = ad.predict_proba(np.column_stack([logits(base, X[te]), X[te]]))[:, 1]
                ad, _ = fit_adapter(penult(base, X[tr]), y[tr], sid[tr], s); p_common_rep = ad.predict_proba(penult(base, X[te]))[:, 1]
                arms = dict(pooled=p_pool, single_base=p_single, single_base_resid=p_single_res, common_base=p_common, common_base_resid=p_common_res, common_base_rep=p_common_rep)
                per[n] = {a: dict(auroc=auc(yy, p), auroc_ci=boot_auc(yy, p, pid), ece=ece(yy, p)) for a, p in arms.items()}
                log(f"B {sname} s{s} {n}: pooled {per[n]['pooled']['auroc']:.4f} single {per[n]['single_base']['auroc']:.4f} common {per[n]['common_base']['auroc']:.4f} common+resid {per[n]['common_base_resid']['auroc']:.4f}")
            save_json(path, dict(set=sname, members=members, seed=s, n_fit_rows=int(fit_rows.sum()), others=others, communities=per, cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg))

# ============================================================ summaries from disk
def paired(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 2: return dict(d=float(d.mean()) if len(d) else float("nan"), ci=[float("nan")] * 2, frac_worse=float("nan"), n=int(len(d)))
    r = np.random.RandomState(0); bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
    return dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), n=int(len(d)))
def summary():
    out = {}
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs_A/*.json"))]
    if recs:
        rows = [dict(community=r["community"], seed=r["seed"], arm=a, auroc=m["auroc"], ece=m["ece"]) for r in recs for a, m in r["arms"].items()]
        S = pd.DataFrame(rows).groupby(["community", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary_A.csv", index=False)
        W = S.pivot_table(index="community", columns="arm", values="auroc"); E = S.pivot_table(index="community", columns="arm", values="ece")
        print(f"\n== Part A, equal opportunity to adapt (cross-fitted adapters on the community's held-out patients), n={len(W)} communities, seed means")
        for a, b, lab in [("base_resid", "pooled_resid", "base+adapter minus pooled+adapter (price of exclusion from shared training, both personalised)"),
                          ("base_rep", "pooled_rep", "same with the representation head"),
                          ("pooled_resid", "pooled", "pooled+adapter minus pooled (what adaptation alone gives the pooled model)"),
                          ("base_resid", "base", "base+adapter minus base"), ("base", "pooled", "base minus pooled (unadapted, as before)"),
                          ("base_resid", "pooled", "base+adapter minus pooled (the comparison made in v11, now cross-fitted)")]:
            p = paired(W[a] - W[b]); pe = paired(E[a] - E[b]); out[f"A|{a}_vs_{b}"] = dict(auroc=p, ece=pe)
            print(f"  {lab:95s} dAUROC {p['d']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] worse {100*p['frac_worse']:.0f}%  dECE {pe['d']:+.4f}")
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs_B/*.json"))]
    if recs:
        rows = [dict(set=r["set"], seed=r["seed"], community=n, arm=a, auroc=m["auroc"], ece=m["ece"]) for r in recs for n, per in r["communities"].items() for a, m in per.items()]
        S = pd.DataFrame(rows).groupby(["set", "community", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary_B.csv", index=False)
        oth = pd.DataFrame([dict(set=r["set"], seed=r["seed"], pooled=r["others"]["pooled"], common_base=r["others"]["common_base"]) for r in recs]).groupby("set").mean(numeric_only=True)
        print("\n== Part B, five communities confined at once (one shared base without all five), seed means")
        for sname in S.set.unique():
            W = S[S.set == sname].pivot_table(index="community", columns="arm", values="auroc")
            print(f"  set '{sname}': {list(W.index)}; others' AUROC pooled {oth.loc[sname, 'pooled']:.4f} vs common base {oth.loc[sname, 'common_base']:.4f}")
            print(W[["pooled", "single_base", "common_base", "single_base_resid", "common_base_resid", "common_base_rep"]].round(4).to_string())
            for a, b, lab in [("common_base", "single_base", "common base minus single-exclusion base (unadapted)"),
                              ("common_base_resid", "single_base_resid", "common base+component minus single-exclusion base+component"),
                              ("common_base_resid", "pooled", "common base+component minus pooled")]:
                p = paired(W[a] - W[b]); out[f"B|{sname}|{a}_vs_{b}"] = p
                print(f"    {lab:70s} {p['d']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] worse {100*p['frac_worse']:.0f}% (n={p['n']})")
    json.dump(out, open(f"{OUT}/paired.json", "w"), indent=1)

if PART in ("A", "both"): part_A()
if PART in ("B", "both"): part_B()
summary(); log(f"done, results in {OUT}")
