"""
exp_v10_fu_trainfit.py. Fair Unlearning (Chen et al., Nat Commun 2026) on the 25 pseudo-communities refitted on
training patients only, reusing the exp_v5_partitions checkpoints (originals, exact, fine-tune, NegGrad+, and the
same patient-matched random controls). FU-race protects equalised odds across race groups; FU-community sets the
groups to the withdrawing community and everyone else. Evaluated on the community's own held-out patients, with the
race equalised-odds gaps on the full test set. Env: ROOT, QUICK=1. Output ROOT/results/v10_fu_trainfit[_quick]/.
"""
import os, json, time, glob, zipfile, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
COHORT = os.environ.get("COHORT", f"{ROOT}/derived/cohort_latest.parquet")
CKPT = os.environ.get("CKPT", f"{ROOT}/results/v5_partitions/ckpt")
OUT = f"{ROOT}/results/v10_fu_trainfit" + ("_quick" if QUICK else "")
RUNS = f"{OUT}/runs"; os.makedirs(RUNS, exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], k_communities=25, min_patients=300, unlearn_epochs=1 if QUICK else 2, unlearn_lr=1e-4, batch=2048,
           alpha=0.1, beta=1.0, gamma=1.0, forget_cap=0.6, min_group_in_batch=5, version="v10.0")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; ckpt {CKPT}; out {OUT}")

# ---------- data and groups, replicated from exp_v5_partitions so the random controls are the same draws
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location",
        "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(COHORT)
feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
race = pd.Categorical(df["race_group"]).codes.astype(np.int64)
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids)
test_p = set(pids[: int(0.2 * len(pids))]); is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
def modal_label(col):
    cnt = df.groupby(["subject_id", col]).size().reset_index(name="n")
    cnt = cnt.sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id"); return cnt.set_index("subject_id")[col]
labels = {}
rg = modal_label("race_group")
for g in sorted(rg.unique()): labels[("race", str(g))] = set(rg.index[rg == g])
if "english" in df.columns:
    eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
ins = modal_label("insurance")
if (ins == "Medicaid").any(): labels[("insurance", "medicaid")] = set(ins.index[ins == "Medicaid"])
ayg = modal_label("anchor_year_group")
for g in sorted(ayg.unique()): labels[("anchor_year", str(g).replace(" ", "").replace("-", "_"))] = set(ayg.index[ayg == g])
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index); comm = assign.loc[sid].values
for c in range(CFG["k_communities"]): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
groups = []
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p = len(set(sid[f_tr])); n_te_p = len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p))
for i, g in enumerate(groups):
    r = np.random.RandomState(100 + i); tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]])))
    g["r_pat"] = np.isin(sid, r.choice(tr_p, min(g["n_tr_p"], len(tr_p)), False)) & is_train
comms = [g for g in groups if g["family"] == "community_trainfit"]
if QUICK: comms = comms[:2]
log(f"{len(comms)} communities (of {len(groups)} groups)")

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def load(key):
    path = f"{CKPT}/{key}.pt"
    if not os.path.exists(path): return None
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m
def cached(key, fn):
    m = load(key)
    if m is None:
        m = fn(); torch.save(m.state_dict(), f"{CKPT}/{key}.pt.tmp"); os.replace(f"{CKPT}/{key}.pt.tmp", f"{CKPT}/{key}.pt")
    return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    m.train(); return np.concatenate(out).astype(np.float64)
def equality_loss(p, yy, g):
    loss = p.new_zeros(()); gs = torch.unique(g)
    for lab in (0.0, 1.0):
        means = []
        for gi in gs:
            mk = (g == gi) & (yy == lab)
            if mk.sum() >= CFG["min_group_in_batch"]: means.append(p[mk].mean())
        for i in range(len(means)):
            for j in range(i + 1, len(means)): loss = loss + (means[i] - means[j]) ** 2
    return loss
def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
def fair_unlearn(init, r_mask, f_mask, group, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(init); params = [p for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=CFG["unlearn_lr"]); bce = nn.BCEWithLogitsLoss()
    Xr, yr, gr = to_t(X[r_mask]), to_t(y[r_mask]), to_t(group[r_mask]); Xf, yf = to_t(X[f_mask]), to_t(y[f_mask]); n, B = len(Xr), CFG["batch"]
    for ep in range(CFG["unlearn_epochs"]):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; logits = m(Xr[idx]); p = torch.sigmoid(logits)
            L_r = bce(logits, yr[idx]); L_e = equality_loss(p, yr[idx], gr[idx])
            j = torch.randint(0, len(Xf), (min(B, len(Xf)),), device=DEVICE); L_f = torch.clamp(bce(m(Xf[j]), yf[j]), max=CFG["forget_cap"])
            g_r = flat_grad(L_r, params); g_e = flat_grad(L_e, params) if L_e.requires_grad else torch.zeros_like(g_r); g_f = flat_grad(-L_f, params)
            v1 = g_r / (g_r.norm() + 1e-12); e2 = g_e - (g_e @ v1) * v1; v2 = e2 / (e2.norm() + 1e-12) if e2.norm() > 1e-8 else torch.zeros_like(v1)
            g_f_perp = g_f - (g_f @ v1) * v1 - (g_f @ v2) * v2; g = CFG["alpha"] * g_f_perp + CFG["beta"] * g_r + CFG["gamma"] * g_e
            opt.zero_grad(); off = 0
            for p_ in params:
                k = p_.numel(); p_.grad = g[off:off + k].reshape(p_.shape).clone(); off += k
            opt.step()
    return m
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
Xte, yte, race_te = X[is_test], y[is_test], race[is_test]
def eo_gaps(p):
    thr = np.quantile(p, 0.8); yhat = p >= thr; tpr, fpr = [], []
    for gi in np.unique(race_te):
        mk = race_te == gi
        if (yte[mk] == 1).sum() < 50 or (yte[mk] == 0).sum() < 50: continue
        tpr.append(yhat[mk & (yte == 1)].mean()); fpr.append(yhat[mk & (yte == 0)].mean())
    return float(max(tpr) - min(tpr)), float(max(fpr) - min(fpr))
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

for g in comms:
    key = f"community_trainfit_{g['name']}"; path = f"{RUNS}/{key}.json"
    if os.path.exists(path):
        try: json.load(open(path)); log(f"skip {key} (done)"); continue
        except Exception: pass
    tg = time.time(); c = int(g["name"][1:]); te = np.where(g["f_te"])[0]; yy, Xc = y[te], X[te]
    comm_group = (comm == c).astype(np.int64); rows = []
    for kind, f_tr, suffix in [("community", g["f_tr"], ""), ("random", g["r_pat"], "rp")]:
        r_tr = is_train & ~f_tr
        for s in CFG["seeds"]:
            orig = load(f"orig_s{s}"); init = orig.state_dict()
            models = {"original": orig, "exact": load(f"exact{suffix}_{key}_s{s}"), "finetune": load(f"ft{suffix}_{key}_s{s}"), "neggrad": load(f"ng{suffix}_{key}_s{s}"),
                      "fu_race": cached(f"furace{suffix}_{key}_s{s}", lambda: fair_unlearn(init, r_tr, f_tr, race, s)),
                      "fu_community": cached(f"fucomm{suffix}_{key}_s{s}", lambda: fair_unlearn(init, r_tr, f_tr, comm_group, s))}
            missing = [k for k, v in models.items() if v is None]
            if missing: log(f"{key} {kind} seed {s}: missing {missing}"); continue
            p0 = predict(orig, Xc); a0, e0 = roc_auc_score(yy, p0), ece(yy, p0)
            for name, m in models.items():
                p = predict(m, Xc); pt = predict(m, Xte); tg_, fg_ = eo_gaps(pt)
                rows.append(dict(kind=kind, seed=s, model=name, comm_auroc=float(roc_auc_score(yy, p)), d_comm_auroc=float(roc_auc_score(yy, p) - a0),
                                 comm_ece=ece(yy, p), d_comm_ece=ece(yy, p) - e0, test_auroc=float(roc_auc_score(yte, pt)), race_tpr_gap=tg_, race_fpr_gap=fg_))
    save_json(path, dict(community=c, key=key, n_train_patients=g["n_tr_p"], rows=rows, cfg=CFG, device=DEVICE, torch=torch.__version__,
                         python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg))
    fu = [r_ for r_ in rows if r_["kind"] == "community" and r_["model"] == "fu_race"]
    log(f"{key}: FU-race community dAUROC {np.mean([r_['d_comm_auroc'] for r_ in fu]):+.4f} ({time.time()-tg:.0f}s)")

def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs"); return
    R = pd.DataFrame([dict(community=rc["community"], **r_) for rc in recs for r_ in rc["rows"]]); R.to_csv(f"{OUT}/results.csv", index=False)
    S = R.groupby(["kind", "community", "model"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 200)
    W = S.pivot_table(index="community", columns=["kind", "model"], values="d_comm_auroc"); WE = S.pivot_table(index="community", columns=["kind", "model"], values="d_comm_ece")
    print("\nchange in AUROC on the community's held-out patients, mean over communities"); print(W.mean().unstack().round(4).to_string())
    print("\nchange in ECE"); print(WE.mean().unstack().round(4).to_string())
    out = {}
    print("\npaired contrast, community removal minus random removal, same test patients")
    for mdl in ["exact", "finetune", "neggrad", "fu_race", "fu_community"]:
        if ("community", mdl) not in W.columns or ("random", mdl) not in W.columns: continue
        d = (W[("community", mdl)] - W[("random", mdl)]).dropna().values; de = (WE[("community", mdl)] - WE[("random", mdl)]).dropna().values
        r = np.random.RandomState(0); bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
        out[mdl] = dict(d_auroc=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), d_ece=float(de.mean()), n=int(len(d)))
        print(f"  {mdl:13s} dAUROC {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] worse {100*(d<0).mean():.0f}%  dECE {de.mean():+.4f} (n={len(d)})")
    with open(f"{OUT}/paired.json", "w") as f: json.dump(out, f, indent=1)
    print("\nrace equalised-odds gaps on the full test set (top-20% threshold), community forget sets")
    print(S[S.kind == "community"].groupby("model")[["race_tpr_gap", "race_fpr_gap", "test_auroc"]].mean().round(4).to_string())
def export():
    z = f"{OUT}/v10_fu_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv", f"{OUT}/paired.json"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
