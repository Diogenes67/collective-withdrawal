"""
Same-held-out evaluation for the collective revocation experiment.

For every community c, evaluate on c's OWN held-out patients:
  original, exact/finetune/neggrad after removing community c,
  exact/finetune/neggrad after removing the size-matched random set (models cached by exp_v2.py),
  plus a new exact retrain after removing an ADMISSION-matched random set (trained here, cached).

Answers the clean question: does concentrated removal harm community c more than removing the
same amount of dispersed training data, judged on the same patients?

Usage: python exp_v2_sameheldout.py            (reads ~/mimic/results/model_cache_v2)
Output: ~/mimic/results/sameheldout_<stamp>/  results.csv, summary.csv, figure.png
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~/mimic")
COHORT = f"{HOME}/derived/cohort_latest.parquet"
CACHE = f"{HOME}/results/model_cache_v2"
STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN = f"{HOME}/results/sameheldout_{STAMP}"
os.makedirs(RUN, exist_ok=True)
CFG = dict(k_communities=25, seeds=[0, 1, 2], epochs=12, lr=1e-3, batch=2048, n_boot=200,
           n_neighbour_clusters=3, n_unrelated_clusters=3)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

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
order = pat_sizes[pat_sizes >= 300].sort_values().index.tolist()
picks = order

def community_target(c, r):
    d = dist[c].copy(); d[c] = np.inf
    near = np.argsort(d)[: CFG["n_neighbour_clusters"]]
    others = [k for k in range(CFG["k_communities"]) if k != c and k not in near]
    unrel = r.choice(others, CFG["n_unrelated_clusters"], replace=False)
    return dict(name=f"community_{c}", community=int(c), f_tr=is_train & (comm == c), f_te=is_test & (comm == c))

def random_target(c, r):
    n_pat = int(pat_sizes[c])
    tr_p = np.array(sorted(set(sid[is_train]))); te_p = np.array(sorted(set(sid[is_test])))
    fp = r.choice(tr_p, n_pat, False); rest = np.setdiff1d(tr_p, fp)
    nbp = r.choice(rest, n_pat, False); rest = np.setdiff1d(rest, nbp)
    unp = r.choice(rest, n_pat, False)
    n_te = max(50, int(n_pat * 0.25))
    ftp = r.choice(te_p, n_te, False); rest_te = np.setdiff1d(te_p, ftp)
    nbtp = r.choice(rest_te, n_te, False); rest_te = np.setdiff1d(rest_te, nbtp)
    untp = r.choice(rest_te, n_te, False)
    return dict(name=f"random_match_{c}", community=int(c), f_tr=np.isin(sid, fp))

r_t = np.random.RandomState(7)
T = {}
for c in picks:
    T[("community", c)] = community_target(c, r_t)
    T[("random", c)] = random_target(c, r_t)
log(f"{len(picks)} communities reconstructed; cache {CACHE}")

# admission-matched random forget sets: draw random training patients until admission count matches
def admission_matched(c, r):
    target_rows = int(T[("community", c)]["f_tr"].sum())
    tr_p = np.array(sorted(set(sid[is_train]))); r.shuffle(tr_p)
    rows_per = pd.Series(sid[is_train]).value_counts()
    chosen, tot = [], 0
    for p in tr_p:
        if tot >= target_rows: break
        chosen.append(p); tot += int(rows_per[p])
    return np.isin(sid, np.array(chosen))
r_a = np.random.RandomState(11)
for c in picks:
    T[("admatch", c)] = dict(name=f"random_admatch_{c}", community=int(c), f_tr=admission_matched(c, r_a))

# ============================================================== model
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train(Xtr, ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(Xtr), to_t(ytr); n, B = len(Xtr), CFG["batch"]
    for ep in range(CFG["epochs"]):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m
def load(key):
    path = f"{CACHE}/{key}.pt"
    if not os.path.exists(path): return None
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m
def cached(key, fn):
    m = load(key)
    if m is None:
        m = fn(); torch.save(m.state_dict(), f"{CACHE}/{key}.pt")
    return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return e
def auc(yy, pp): return roc_auc_score(yy, pp)
def boot_patient(fn, yy, pp, pid):
    """bootstrap over patients, not admissions"""
    r = np.random.RandomState(1); ups = np.unique(pid); groups = {p: np.where(pid == p)[0] for p in ups}; vals = []
    for _ in range(CFG["n_boot"]):
        i = np.concatenate([groups[p] for p in r.choice(ups, len(ups), True)]); vals.append(fn(yy[i], pp[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

# ============================================================== evaluate on the community's own held-out patients
rows = []
for c in picks:
    te = np.where(T[("community", c)]["f_te"])[0]
    yy, pid, Xc = y[te], sid[te], X[te]
    comm_rows = int(T[("community", c)]["f_tr"].sum()); rand_rows = int(T[("random", c)]["f_tr"].sum())
    adm_rows = int(T[("admatch", c)]["f_tr"].sum())
    prev_c = float(y[T[("community", c)]["f_tr"]].mean()); prev_r = float(y[T[("random", c)]["f_tr"]].mean())
    for s in CFG["seeds"]:
        models = {"original": load(f"orig_s{s}"),
                  "exact_community": load(f"exact_community_{c}_s{s}"),
                  "finetune_community": load(f"ft_community_{c}_s{s}"),
                  "neggrad_community": load(f"ng_community_{c}_s{s}"),
                  "exact_random": load(f"exact_random_match_{c}_s{s}"),
                  "finetune_random": load(f"ft_random_match_{c}_s{s}"),
                  "neggrad_random": load(f"ng_random_match_{c}_s{s}")}
        r_tr = is_train & ~T[("admatch", c)]["f_tr"]
        models["exact_random_admatched"] = cached(f"exact_random_admatch_{c}_s{s}", lambda: train(X[r_tr], y[r_tr], s))
        missing = [k for k, v in models.items() if v is None]
        if missing: log(f"community {c} seed {s}: missing {missing}"); continue
        p0 = predict(models["original"], Xc); a0, e0 = auc(yy, p0), ece(yy, p0)
        for name, m in models.items():
            p = predict(m, Xc); a, e = auc(yy, p), ece(yy, p)
            alo, ahi = boot_patient(auc, yy, p, pid)
            rows.append(dict(community=int(c), seed=s, model=name, n_test_rows=len(te), n_test_patients=len(np.unique(pid)),
                             forget_rows_community=comm_rows, forget_rows_random=rand_rows, forget_rows_admatched=adm_rows,
                             prevalence_community=prev_c, prevalence_random=prev_r,
                             auroc=a, auroc_lo=alo, auroc_hi=ahi, ece=e, d_auroc=a - a0, d_ece=e - e0))
    pd.DataFrame(rows).to_csv(f"{RUN}/results.csv", index=False)
    log(f"community {c}: done")

R = pd.DataFrame(rows)
S = R.groupby(["community", "model"]).mean(numeric_only=True).reset_index()
S.to_csv(f"{RUN}/summary.csv", index=False)

def paired(a, b):
    d = (a.values - b.values); r = np.random.RandomState(0)
    bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d < 0).mean()
W = S.pivot(index="community", columns="model", values="d_auroc")
WE = S.pivot(index="community", columns="model", values="d_ece")
out = {}
for k, (a, b) in {"exact_comm_vs_exact_random": ("exact_community", "exact_random"),
                  "exact_comm_vs_exact_admatched": ("exact_community", "exact_random_admatched"),
                  "finetune_comm_vs_finetune_random": ("finetune_community", "finetune_random"),
                  "neggrad_comm_vs_neggrad_random": ("neggrad_community", "neggrad_random")}.items():
    m, lo, hi, fneg = paired(W[a], W[b]); me, elo, ehi, _ = paired(WE[a], WE[b])
    out[k] = dict(d_auroc=m, ci=(lo, hi), frac_community_worse=fneg, d_ece=me, ece_ci=(elo, ehi))
with open(f"{RUN}/paired.json", "w") as f: json.dump(out, f, indent=2, default=float)

print("\nmean change vs original on the community's own held-out patients (averaged over seeds, then communities):")
print(W.mean().round(4).to_string())
print("\npaired contrasts (community removal minus random removal, same test patients):")
for k, v in out.items():
    print(f"  {k}: dAUROC {v['d_auroc']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  community worse in {100*v['frac_community_worse']:.0f}%   dECE {v['d_ece']:+.4f} [{v['ece_ci'][0]:+.4f}, {v['ece_ci'][1]:+.4f}]")
print("\nforget-set sizes (rows): community vs patient-matched random vs admission-matched random")
print(S[S.model == "original"][["community", "forget_rows_community", "forget_rows_random", "forget_rows_admatched", "prevalence_community", "prevalence_random"]].round(3).to_string(index=False))

# figure: per community, exact removal of the community vs random removal, on the same patients
sz = S[S.model == "original"].set_index("community")["forget_rows_community"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
o = sz.sort_values().index
xs = np.arange(len(o))
for i, (col, lab, mk) in enumerate([("exact_community", "remove the community", "o"), ("exact_random", "remove matched random patients", "s"), ("exact_random_admatched", "remove admission-matched random patients", "^")]):
    ax[0].plot(xs, W.loc[o, col], mk, label=lab, alpha=0.8)
    ax[1].plot(xs, WE.loc[o, col], mk, label=lab, alpha=0.8)
for a, ttl in zip(ax, ["Change in AUROC on the community's held-out patients", "Change in ECE on the community's held-out patients"]):
    a.axhline(0, ls=":", color="grey"); a.set_xticks(xs); a.set_xticklabels([f"{c}\n{int(sz[c]):,}" for c in o], fontsize=7)
    a.set_xlabel("community (rows removed)"); a.set_title(ttl)
ax[1].axhline(0.01, ls=":", color="red"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RUN}/figure.png", dpi=150)
log(f"done, run folder {RUN}")
