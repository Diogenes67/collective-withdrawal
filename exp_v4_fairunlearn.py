"""
Fair Unlearning baseline (Chen et al., Nature Communications 2026) for the collective revocation experiment.

FU update: g = alpha * g_f_perp + beta * g_r + gamma * g_e, where g_r is the retained-data BCE gradient,
g_e the gradient of an equalised-odds surrogate across demographic groups, and g_f_perp the forget-set
ascent gradient projected orthogonal to span{g_r, g_e} (Gram-Schmidt). Two instantiations:
  FU-race       groups = race_group, as in Chen et al. (ethnicity); the forget set is the community.
  FU-community  groups = {revoking community, everyone else}; the closest a group-fairness constraint
                can get to protecting the forget set itself.
Both start from the original weights and run for the same budget as fine-tune and NegGrad+.
Evaluated on the community's own held-out patients (as exp_v2_sameheldout.py), for community and
size-matched random forget sets, plus the race equalised-odds gaps on the full test set.

Usage: python exp_v4_fairunlearn.py
Output: ~/mimic/results/fu_<stamp>/ results.csv, summary printed
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

HOME = os.path.expanduser("~/mimic")
COHORT = f"{HOME}/derived/cohort_latest.parquet"
CACHE = f"{HOME}/results/model_cache_v2"
STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN = f"{HOME}/results/fu_{STAMP}"
os.makedirs(RUN, exist_ok=True)
CFG = dict(k_communities=25, seeds=[0, 1, 2], unlearn_epochs=2, unlearn_lr=1e-4, batch=2048,
           alpha=0.1, beta=1.0, gamma=1.0, forget_cap=0.6, min_group_in_batch=5, n_boot=200,
           n_neighbour_clusters=3, n_unrelated_clusters=3)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
json.dump(CFG, open(f"{RUN}/config.json", "w"), indent=2)

# ============================================================== data (identical to exp_v2.py)
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
race = pd.Categorical(df["race_group"]).codes.astype(np.int64)
race_names = list(pd.Categorical(df["race_group"]).categories)
rng = np.random.RandomState(0)
pids = df.subject_id.unique(); rng.shuffle(pids)
test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6
X = (X - mu) / sd

pat_mean = pd.DataFrame(X).groupby(sid).mean()
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.values)
pat_comm = pd.Series(km.labels_, index=pat_mean.index)
comm = pat_comm.loc[sid].values
cent = km.cluster_centers_
dist = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)
pat_sizes = pat_comm.value_counts().sort_index()
picks = pat_sizes[pat_sizes >= 300].sort_values().index.tolist()

def community_target(c, r):
    d = dist[c].copy(); d[c] = np.inf
    near = np.argsort(d)[: CFG["n_neighbour_clusters"]]
    others = [k for k in range(CFG["k_communities"]) if k != c and k not in near]
    r.choice(others, CFG["n_unrelated_clusters"], replace=False)
    return dict(name=f"community_{c}", kind="community", community=int(c), f_tr=is_train & (comm == c), f_te=is_test & (comm == c))
def random_target(c, r):
    n_pat = int(pat_sizes[c])
    tr_p = np.array(sorted(set(sid[is_train]))); te_p = np.array(sorted(set(sid[is_test])))
    fp = r.choice(tr_p, n_pat, False); rest = np.setdiff1d(tr_p, fp)
    nbp = r.choice(rest, n_pat, False); rest = np.setdiff1d(rest, nbp)
    r.choice(rest, n_pat, False)
    n_te = max(50, int(n_pat * 0.25))
    ftp = r.choice(te_p, n_te, False); rest_te = np.setdiff1d(te_p, ftp)
    nbtp = r.choice(rest_te, n_te, False); rest_te = np.setdiff1d(rest_te, nbtp)
    r.choice(rest_te, n_te, False)
    return dict(name=f"random_match_{c}", kind="random", community=int(c), f_tr=np.isin(sid, fp), f_te=is_test & (comm == c))
r_t = np.random.RandomState(7)
targets = []
for c in picks:
    targets.append(community_target(c, r_t)); targets.append(random_target(c, r_t))
log(f"{len(targets)} forget sets; device {DEVICE}; race groups {race_names}")

# ============================================================== model
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def load(key):
    path = f"{CACHE}/{key}.pt"
    if not os.path.exists(path): return None
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    m.train(); return np.concatenate(out).astype(np.float64)

def equality_loss(p, yy, g):
    """equalised-odds surrogate: squared differences in mean predicted probability between every pair of
    groups, conditioned on the true label (Chen et al., eq. 5 in spirit)"""
    loss = p.new_zeros(()); groups = torch.unique(g)
    for lab in (0.0, 1.0):
        means = []
        for gi in groups:
            mk = (g == gi) & (yy == lab)
            if mk.sum() >= CFG["min_group_in_batch"]: means.append(p[mk].mean())
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                loss = loss + (means[i] - means[j]) ** 2
    return loss

def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])

def fair_unlearn(init, r_mask, f_mask, group, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(init)
    params = [p for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=CFG["unlearn_lr"])
    bce = nn.BCEWithLogitsLoss()
    Xr, yr, gr = to_t(X[r_mask]), to_t(y[r_mask]), to_t(group[r_mask])
    Xf, yf = to_t(X[f_mask]), to_t(y[f_mask])
    n, B = len(Xr), CFG["batch"]
    for ep in range(CFG["unlearn_epochs"]):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            logits = m(Xr[idx]); p = torch.sigmoid(logits)
            L_r = bce(logits, yr[idx])
            L_e = equality_loss(p, yr[idx], gr[idx])
            j = torch.randint(0, len(Xf), (min(B, len(Xf)),), device=DEVICE)
            L_f = torch.clamp(bce(m(Xf[j]), yf[j]), max=CFG["forget_cap"])
            g_r = flat_grad(L_r, params)
            g_e = flat_grad(L_e, params) if L_e.requires_grad else torch.zeros_like(g_r)
            g_f = flat_grad(-L_f, params)
            # Gram-Schmidt on {g_r, g_e}; project g_f onto the orthogonal complement
            v1 = g_r / (g_r.norm() + 1e-12)
            e2 = g_e - (g_e @ v1) * v1; v2 = e2 / (e2.norm() + 1e-12) if e2.norm() > 1e-8 else torch.zeros_like(v1)
            g_f_perp = g_f - (g_f @ v1) * v1 - (g_f @ v2) * v2
            g = CFG["alpha"] * g_f_perp + CFG["beta"] * g_r + CFG["gamma"] * g_e
            opt.zero_grad(); off = 0
            for p_ in params:
                k = p_.numel(); p_.grad = g[off:off + k].reshape(p_.shape).clone(); off += k
            opt.step()
    return m

def cached(key, fn):
    m = load(key)
    if m is None:
        m = fn(); torch.save(m.state_dict(), f"{CACHE}/{key}.pt")
    return m

# ============================================================== metrics
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return e
Xte, yte, race_te = X[is_test], y[is_test], race[is_test]
def eo_gaps(p):
    """max pairwise TPR and FPR gap across race groups at the top-20% risk threshold, full test set"""
    thr = np.quantile(p, 0.8); yhat = p >= thr; tpr, fpr = [], []
    for gi in np.unique(race_te):
        mk = race_te == gi
        if (yte[mk] == 1).sum() < 50 or (yte[mk] == 0).sum() < 50: continue
        tpr.append(yhat[mk & (yte == 1)].mean()); fpr.append(yhat[mk & (yte == 0)].mean())
    return max(tpr) - min(tpr), max(fpr) - min(fpr)

# ============================================================== run
rows = []
for T in targets:
    c = T["community"]; f_tr = T["f_tr"]; r_tr = is_train & ~f_tr
    te = np.where(T["f_te"])[0]; yy, Xc = y[te], X[te]
    comm_group = (comm == c).astype(np.int64)
    for s in CFG["seeds"]:
        init = load(f"orig_s{s}").state_dict()
        models = {"original": load(f"orig_s{s}"), "exact": load(f"exact_{T['name']}_s{s}"),
                  "finetune": load(f"ft_{T['name']}_s{s}"), "neggrad": load(f"ng_{T['name']}_s{s}"),
                  "fu_race": cached(f"furace_{T['name']}_s{s}", lambda: fair_unlearn(init, r_tr, f_tr, race, s)),
                  "fu_community": cached(f"fucomm_{T['name']}_s{s}", lambda: fair_unlearn(init, r_tr, f_tr, comm_group, s))}
        p0 = predict(models["original"], Xc); a0, e0 = roc_auc_score(yy, p0), ece(yy, p0)
        for name, m in models.items():
            p = predict(m, Xc); pt = predict(m, Xte); tg, fg = eo_gaps(pt)
            rows.append(dict(target=T["name"], kind=T["kind"], community=c, seed=s, model=name,
                             comm_auroc=roc_auc_score(yy, p), d_comm_auroc=roc_auc_score(yy, p) - a0,
                             comm_ece=ece(yy, p), d_comm_ece=ece(yy, p) - e0,
                             test_auroc=roc_auc_score(yte, pt), race_tpr_gap=tg, race_fpr_gap=fg))
    pd.DataFrame(rows).to_csv(f"{RUN}/results.csv", index=False)
    log(f"{T['name']} done")

R = pd.DataFrame(rows)
S = R.groupby(["kind", "community", "model"]).mean(numeric_only=True).reset_index()
S.to_csv(f"{RUN}/summary.csv", index=False)
def paired(a, b):
    d = a.values - b.values; r = np.random.RandomState(0)
    bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d < 0).mean()
W = S.pivot_table(index="community", columns=["kind", "model"], values="d_comm_auroc")
WE = S.pivot_table(index="community", columns=["kind", "model"], values="d_comm_ece")
print("\nChange in AUROC on the community's held-out patients, mean over communities:")
print(W.mean().unstack().round(4).to_string())
print("\nChange in ECE:")
print(WE.mean().unstack().round(4).to_string())
print("\nPaired contrast, community removal minus random removal, same test patients:")
out = {}
for mdl in ["exact", "finetune", "neggrad", "fu_race", "fu_community"]:
    m_, lo, hi, fneg = paired(W[("community", mdl)], W[("random", mdl)])
    me, elo, ehi, _ = paired(WE[("community", mdl)], WE[("random", mdl)])
    out[mdl] = dict(d_auroc=m_, ci=(lo, hi), frac_worse=fneg, d_ece=me, ece_ci=(elo, ehi))
    print(f"  {mdl:13s} dAUROC {m_:+.4f} [{lo:+.4f}, {hi:+.4f}] community worse {100*fneg:.0f}%   dECE {me:+.4f} [{elo:+.4f}, {ehi:+.4f}]")
json.dump(out, open(f"{RUN}/paired.json", "w"), indent=2, default=float)
print("\nRace equalised-odds gaps on the full test set (top-20% threshold), mean over community forget sets:")
print(S[S.kind == "community"].groupby("model")[["race_tpr_gap", "race_fpr_gap", "test_auroc"]].mean().round(4).to_string())
log(f"done, run folder {RUN}")
