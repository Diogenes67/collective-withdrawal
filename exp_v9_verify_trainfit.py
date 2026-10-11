"""
exp_v9_verify_trainfit.py. Record-level audits on the 25 pseudo-communities refitted on training patients only
(the clustering of exp_v5_partitions): the loss-based membership-inference attack and the offline reference-model
attack (LiRA, 16 shadows, over-trained control), for the original, exact-retrain, fine-tuned and NegGrad+ models,
three seeds for the loss attack and seed 0 for LiRA. Shadows and the over-trained control are reused from
results/model_cache_v2 (they do not depend on the clustering). The noise floor on these communities is in
results/v6_groupaudit (div_ratio). Env: ROOT, QUICK=1. Output ROOT/results/v9_verify_trainfit[_quick]/.
"""
import os, json, time, glob, zipfile, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
COHORT = os.environ.get("COHORT", f"{ROOT}/derived/cohort_latest.parquet")
CKPT = os.environ.get("CKPT", f"{ROOT}/results/v5_partitions/ckpt")
SHADOW = os.environ.get("SHADOW", f"{ROOT}/results/model_cache_v2")
OUT = f"{ROOT}/results/v9_verify_trainfit" + ("_quick" if QUICK else "")
RUNS = f"{OUT}/runs"; os.makedirs(RUNS, exist_ok=True); os.makedirs(SHADOW, exist_ok=True)
CFG = dict(seeds=[0, 1, 2], k_communities=25, min_patients=300, epochs=2 if QUICK else 12, lr=1e-3, batch=2048,
           n_shadow=2 if QUICK else 16, overfit_epochs=4 if QUICK else 60, n_per_side=1500, version="v9.0")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; ckpt {CKPT}; shadows {SHADOW}; out {OUT}")

META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location",
        "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(COHORT)
feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids)
test_p = set(pids[: int(0.2 * len(pids))]); is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd; tr_idx = np.where(is_train)[0]
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index); comm = assign.loc[sid].values
picks = [c for c in range(CFG["k_communities"]) if len(set(sid[is_train & (comm == c)])) >= CFG["min_patients"] and len(set(sid[is_test & (comm == c)])) >= 50]
if QUICK: picks = picks[:2]
log(f"{len(picks)} communities")

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
def load_from(folder, key):
    path = f"{folder}/{key}.pt"
    if not os.path.exists(path): return None
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m
def cached(folder, key, fn):
    m = load_from(folder, key)
    if m is None:
        m = fn(); torch.save(m.state_dict(), f"{folder}/{key}.pt.tmp"); os.replace(f"{folder}/{key}.pt.tmp", f"{folder}/{key}.pt")
    return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    m.train(); return np.concatenate(out).astype(np.float64)
def phi(p, yy):
    p = np.clip(p, 1e-6, 1 - 1e-6); py = np.where(yy == 1, p, 1 - p); return np.log(py / (1 - py))

# shadows and over-trained control (clustering-independent, reused)
sfolder = SHADOW if not QUICK else f"{OUT}/shadow_quick"; os.makedirs(sfolder, exist_ok=True)
shadows, shadow_masks = [], []
for k in range(CFG["n_shadow"]):
    mp = f"{sfolder}/shadow_{k}_mask.npy"
    if os.path.exists(mp): mask = np.load(mp)
    else:
        r = np.random.RandomState(100 + k); mask = np.zeros(len(X), bool); mask[r.choice(tr_idx, len(tr_idx) // 2, False)] = True; np.save(mp, mask)
    shadows.append(cached(sfolder, f"shadow_{k}", lambda mask=mask, k=k: train(X[mask], y[mask], 100 + k, CFG["epochs"]))); shadow_masks.append(mask)
overfit = cached(sfolder, "orig_overfit", lambda: train(X[is_train], y[is_train], 0, CFG["overfit_epochs"]))
log(f"{len(shadows)} shadows and over-trained control ready (test AUROC {roc_auc_score(y[is_test], predict(overfit, X[is_test])):.4f})")

def lira_scores(m, idx):
    obs = phi(predict(m, X[idx]), y[idx]); outs = np.full((len(idx), CFG["n_shadow"]), np.nan)
    for k, (sm, mk) in enumerate(zip(shadows, shadow_masks)):
        outs[:, k] = np.where(mk[idx], np.nan, phi(predict(sm, X[idx]), y[idx]))
    mu_out = np.nanmean(outs, 1); sd_out = max(np.nanstd(outs), 1e-3); mu_out = np.where(np.isnan(mu_out), np.nanmean(outs), mu_out)
    return norm.cdf((obs - mu_out) / sd_out)
def strat_sample(idx, n, r):
    yy = y[idx]; i1, i0 = idx[yy == 1], idx[yy == 0]
    n1 = min(len(i1), int(round(n * 0.5))); n0 = min(len(i0), n - n1)
    return np.r_[r.choice(i1, n1, False), r.choice(i0, n0, False)]
def loss_mia(m, Xf, yf, Xh, yh):
    r = np.random.RandomState(2)
    def strat(Xa, ya, n1, n0):
        i = np.r_[r.choice(np.where(ya == 1)[0], n1, False), r.choice(np.where(ya == 0)[0], n0, False)]; return Xa[i], ya[i]
    n1 = min((yf == 1).sum(), (yh == 1).sum()); n0 = min((yf == 0).sum(), (yh == 0).sum()); n = n1 + n0
    Xf, yf = strat(Xf, yf, n1, n0); Xh, yh = strat(Xh, yh, n1, n0)
    def feats(Xa, ya):
        p = np.clip(predict(m, Xa), 1e-6, 1 - 1e-6); loss = -(ya * np.log(p) + (1 - ya) * np.log(1 - p)); ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return np.c_[loss, np.abs(p - 0.5), ent, ya]
    A = np.r_[feats(Xf, yf), feats(Xh, yh)]; lab = np.r_[np.ones(n), np.zeros(n)]
    return float(cross_val_score(LogisticRegression(max_iter=500), A, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean())
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

for c in picks:
    key = f"community_trainfit_c{c:02d}"; path = f"{RUNS}/{key}.json"
    if os.path.exists(path):
        try: json.load(open(path)); log(f"skip {key} (done)"); continue
        except Exception: pass
    tg = time.time(); f_idx = np.where(is_train & (comm == c))[0]; h_idx = np.where(is_test & (comm == c))[0]
    rows = []
    for s in CFG["seeds"]:
        models = {"original": load_from(CKPT, f"orig_s{s}"), "exact": load_from(CKPT, f"exact_{key}_s{s}"),
                  "finetune": load_from(CKPT, f"ft_{key}_s{s}"), "neggrad": load_from(CKPT, f"ng_{key}_s{s}")}
        if s == 0: models["overfit_control"] = overfit
        missing = [k for k, v in models.items() if v is None]
        if missing: log(f"{key} seed {s}: missing {missing}"); continue
        r = np.random.RandomState(c); n = min(CFG["n_per_side"], len(f_idx), len(h_idx))
        F = strat_sample(f_idx, n, r); H = strat_sample(h_idx, n, r); lab = np.r_[np.ones(len(F)), np.zeros(len(H))]; idx = np.r_[F, H]
        for name, m in models.items():
            row = dict(model=name, seed=s, loss_mia_auc=loss_mia(m, X[f_idx], y[f_idx], X[h_idx], y[h_idx]))
            if s == 0:
                sc = lira_scores(m, idx); row.update(lira_auc=float(roc_auc_score(lab, sc)), lira_tpr_at_1fpr=float(np.mean(sc[lab == 1] > np.quantile(sc[lab == 0], 0.99))), n_per_side=int(n))
            rows.append(row)
    save_json(path, dict(community=c, key=key, n_contrib_rows=int(len(f_idx)), n_heldout_rows=int(len(h_idx)), rows=rows, cfg=CFG,
                         device=DEVICE, torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg))
    lira = {r_["model"]: r_.get("lira_auc") for r_ in rows if r_["seed"] == 0}
    log(f"{key}: LiRA " + " ".join(f"{k} {v:.3f}" for k, v in lira.items() if v is not None) + f" ({time.time()-tg:.0f}s)")

def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs"); return
    R = pd.DataFrame([dict(community=rc["community"], **r_) for rc in recs for r_ in rc["rows"]]); R.to_csv(f"{OUT}/results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nloss-based membership attack AUC (per-community seed means), mean [min, max] over communities")
    L = R.groupby(["community", "model"]).loss_mia_auc.mean().reset_index(); print(L.groupby("model").loss_mia_auc.agg(["mean", "min", "max"]).round(3).to_string())
    print("\noffline LiRA attack AUC (seed 0), mean [min, max] over communities")
    A = R[R.seed == 0].dropna(subset=["lira_auc"]); print(A.groupby("model").lira_auc.agg(["mean", "min", "max"]).round(3).to_string())
    print("\nLiRA TPR at 1% FPR"); print(A.groupby("model").lira_tpr_at_1fpr.agg(["mean", "min", "max"]).round(3).to_string())
    S = pd.concat([L.groupby("model").loss_mia_auc.agg(["mean", "min", "max"]).add_prefix("loss_"), A.groupby("model").lira_auc.agg(["mean", "min", "max"]).add_prefix("lira_")], axis=1)
    S.to_csv(f"{OUT}/summary.csv")
def export():
    z = f"{OUT}/v9_verify_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
