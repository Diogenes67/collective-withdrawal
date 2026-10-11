"""
Validate the verification tests for the collective revocation experiment.

Part A. Reference-model membership attack (offline LiRA, Carlini et al. 2022): K shadow models on
        random halves of the training data give each record an OUT distribution of logit-confidence;
        the attack scores how far a target model's confidence sits above that distribution.
        Sensitivity check: an over-trained copy of the original model, which should be attackable.
Part B. Noise floor with ten exact retrains per community (seeds 0-9) instead of three.

Usage: python exp_v3_verify.py          (reuses ~/mimic/results/model_cache_v2)
Output: ~/mimic/results/verify_<stamp>/  attack.csv, noise.csv, summary printed
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

HOME = os.path.expanduser("~/mimic")
COHORT = f"{HOME}/derived/cohort_latest.parquet"
CACHE = f"{HOME}/results/model_cache_v2"
STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN = f"{HOME}/results/verify_{STAMP}"
os.makedirs(RUN, exist_ok=True)
CFG = dict(k_communities=25, epochs=12, lr=1e-3, batch=2048, n_shadow=16, overfit_epochs=60,
           noise_seeds=list(range(10)), n_per_side=1500, n_div_rows=5000,
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
tr_idx = np.where(is_train)[0]

pat_mean = pd.DataFrame(X).groupby(sid).mean()
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.values)
pat_comm = pd.Series(km.labels_, index=pat_mean.index)
comm = pat_comm.loc[sid].values
pat_sizes = pat_comm.value_counts().sort_index()
picks = pat_sizes[pat_sizes >= 300].sort_values().index.tolist()
log(f"{len(picks)} communities; device {DEVICE}")

# ============================================================== model
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train(Xtr, ytr, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(Xtr), to_t(ytr); n, B = len(Xtr), CFG["batch"]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m
def cached(key, fn):
    path = f"{CACHE}/{key}.pt"; m = MLP(X.shape[1]).to(DEVICE)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); return m
    m = fn(); torch.save(m.state_dict(), path); return m
def load_required(key):
    path = f"{CACHE}/{key}.pt"
    if not os.path.exists(path): raise FileNotFoundError(path)
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def phi(p, yy):
    p = np.clip(p, 1e-6, 1 - 1e-6); py = np.where(yy == 1, p, 1 - p); return np.log(py / (1 - py))

# ============================================================== Part A: shadow models and attack
log("Part A: shadow models")
shadows, shadow_masks = [], []
for k in range(CFG["n_shadow"]):
    r = np.random.RandomState(100 + k)
    mask = np.zeros(len(X), bool); mask[r.choice(tr_idx, len(tr_idx) // 2, False)] = True
    mp = f"{CACHE}/shadow_{k}_mask.npy"
    if os.path.exists(mp): mask = np.load(mp)
    else: np.save(mp, mask)
    shadows.append(cached(f"shadow_{k}", lambda mask=mask, k=k: train(X[mask], y[mask], 100 + k, CFG["epochs"])))
    shadow_masks.append(mask)
    log(f"shadow {k} ready")
overfit = cached("orig_overfit", lambda: train(X[is_train], y[is_train], 0, CFG["overfit_epochs"]))
log(f"overfit control ready; test AUROC {roc_auc_score(y[is_test], predict(overfit, X[is_test])):.4f}")

def lira_scores(m, idx):
    """offline LiRA: score = P_out(phi <= observed) per record, using shadows that did not train on it"""
    obs = phi(predict(m, X[idx]), y[idx])
    outs = np.full((len(idx), CFG["n_shadow"]), np.nan)
    for k, (sm, mk) in enumerate(zip(shadows, shadow_masks)):
        ph = phi(predict(sm, X[idx]), y[idx]); outs[:, k] = np.where(mk[idx], np.nan, ph)
    mu_out = np.nanmean(outs, 1); sd_out = np.nanstd(outs); sd_out = max(sd_out, 1e-3)   # global variance variant
    return norm.cdf((obs - mu_out) / sd_out)

def strat_sample(idx, n, r):
    yy = y[idx]; i1, i0 = idx[yy == 1], idx[yy == 0]
    n1 = min(len(i1), int(round(n * 0.5))); n0 = min(len(i0), n - n1)
    return np.r_[r.choice(i1, n1, False), r.choice(i0, n0, False)]

attack_rows = []
for c in picks:
    r = np.random.RandomState(int(c))
    f_idx = np.where(is_train & (comm == c))[0]; h_idx = np.where(is_test & (comm == c))[0]
    n = min(CFG["n_per_side"], len(f_idx), len(h_idx))
    F = strat_sample(f_idx, n, r); H = strat_sample(h_idx, n, r)
    lab = np.r_[np.ones(len(F)), np.zeros(len(H))]; idx = np.r_[F, H]
    for name, key in [("original", "orig_s0"), ("overfit_control", "orig_overfit"), ("exact", f"exact_community_{c}_s0"),
                      ("finetune", f"ft_community_{c}_s0"), ("neggrad", f"ng_community_{c}_s0")]:
        m = load_required(key)
        sc = lira_scores(m, idx)
        attack_rows.append(dict(community=int(c), model=name, n_per_side=n, lira_auc=float(roc_auc_score(lab, sc)),
                                lira_tpr_at_1fpr=float(np.mean(sc[lab == 1] > np.quantile(sc[lab == 0], 0.99)))))
    pd.DataFrame(attack_rows).to_csv(f"{RUN}/attack.csv", index=False)
    log(f"attack community {c} done")
A = pd.DataFrame(attack_rows)

# ============================================================== Part B: ten-seed noise floor
log("Part B: ten-seed exact retrains")
originals = {s: cached(f"orig_s{s}", lambda s=s: train(X[is_train], y[is_train], s, CFG["epochs"])) for s in CFG["noise_seeds"]}
noise_rows = []
for c in picks:
    f_tr = is_train & (comm == c); r_tr = is_train & ~f_tr
    sub = np.where(f_tr)[0][: CFG["n_div_rows"]]
    ex = {s: predict(cached(f"exact_community_{c}_s{s}", lambda s=s: train(X[r_tr], y[r_tr], s, CFG["epochs"])), X[sub])
          for s in CFG["noise_seeds"]}
    orig = {s: predict(originals[s], X[sub]) for s in CFG["noise_seeds"]}
    ft = {s: predict(load_required(f"ft_community_{c}_s{s}"), X[sub]) for s in [0, 1, 2]}
    ng = {s: predict(load_required(f"ng_community_{c}_s{s}"), X[sub]) for s in [0, 1, 2]}
    S = CFG["noise_seeds"]
    floor = [np.mean(np.abs(ex[a] - ex[b])) for i, a in enumerate(S) for b in S[i + 1:]]
    o_vs_e = [np.mean(np.abs(orig[a] - ex[b])) for a in S for b in S]
    f_vs_e = [np.mean(np.abs(ft[a] - ex[b])) for a in ft for b in S]
    n_vs_e = [np.mean(np.abs(ng[a] - ex[b])) for a in ng for b in S]
    noise_rows.append(dict(community=int(c), rows=len(sub), noise_floor=np.mean(floor), noise_floor_sd=np.std(floor),
                           orig_vs_exact=np.mean(o_vs_e), finetune_vs_exact=np.mean(f_vs_e), neggrad_vs_exact=np.mean(n_vs_e),
                           ratio_orig=np.mean(o_vs_e) / np.mean(floor), ratio_finetune=np.mean(f_vs_e) / np.mean(floor),
                           ratio_neggrad=np.mean(n_vs_e) / np.mean(floor),
                           # fraction of (orig, exact) pairs whose divergence exceeds the 95th percentile of retrain pairs
                           orig_above_floor95=float(np.mean(np.array(o_vs_e) > np.percentile(floor, 95))),
                           finetune_above_floor95=float(np.mean(np.array(f_vs_e) > np.percentile(floor, 95))),
                           neggrad_above_floor95=float(np.mean(np.array(n_vs_e) > np.percentile(floor, 95)))))
    pd.DataFrame(noise_rows).to_csv(f"{RUN}/noise.csv", index=False)
    log(f"noise community {c} done")
N = pd.DataFrame(noise_rows)

# ============================================================== summary
print("\nPart A. Offline LiRA attack AUC (forget-set training records vs held-out community records), mean [min, max] over communities:")
print(A.groupby("model").lira_auc.agg(["mean", "min", "max"]).round(3).to_string())
print("\nTPR at 1% FPR:")
print(A.groupby("model").lira_tpr_at_1fpr.agg(["mean", "min", "max"]).round(3).to_string())
print("\nPart B. Ten-seed noise floor. Divergence from exact retrain on the forget set, as a multiple of the retrain-vs-retrain floor:")
print(N[["ratio_orig", "ratio_finetune", "ratio_neggrad"]].agg(["mean", "min", "max"]).round(2).to_string())
print("\nFraction of model pairs above the 95th percentile of retrain-vs-retrain divergence:")
print(N[["orig_above_floor95", "finetune_above_floor95", "neggrad_above_floor95"]].agg(["mean", "min", "max"]).round(2).to_string())
with open(f"{RUN}/config.json", "w") as f: json.dump(CFG, f, indent=2)
log(f"done, run folder {RUN}")
