"""
exp_v6_groupaudit.py. Group-level verification of collective withdrawal, on the 25 pseudo-communities refitted on
training patients only (same clustering as exp_v5_partitions with COMMUNITIES=1, whose checkpoints it reuses).

Two audits a withdrawing group can run on a candidate model.
  A. Reference audit (needs retrains without the group). Statistic on the group's contributed records (mean per-record
     loss, AUROC on the group's held-out patients) compared with its distribution over ten exact retrains without the
     group; z-score and one-sided flag at the 5th/95th percentile. False-positive rate from leave-one-out over the
     retrains; power against the original model, fine-tuning and NegGrad+.
  B. Query-only audit (needs no retrain). Generalisation gap of the candidate between the group's contributed records
     and the group's records that were never contributed (held-out patients), with a permutation null over which
     records are called contributed. One-sided p-value; flag at 0.05.
Also records the noise floor (mean absolute prediction difference between exact retrains differing only by seed) and
the divergence of each candidate from the retrains, as in exp_v3_verify.

Env: ROOT (default ~/mimic), QUICK=1 for a smoke run. Reads ROOT/results/v5_partitions/ckpt, writes ROOT/results/v6_groupaudit[_quick]/.
Seven extra exact retrains per community (seeds 3 to 9) are trained here and cached in the same ckpt folder.
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
OUT = f"{ROOT}/results/v6_groupaudit" + ("_quick" if QUICK else "")
RUNS = f"{OUT}/runs"; os.makedirs(RUNS, exist_ok=True)
CFG = dict(seeds=[0, 1, 2], retrain_seeds=list(range(10)) if not QUICK else [0, 1, 2, 3], epochs=2 if QUICK else 12, lr=1e-3,
           batch=2048, k_communities=25, min_patients=300, n_perm=200 if QUICK else 1000, max_side=20000, alpha=0.05,
           n_div_rows=5000, version="v6.1")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; ckpt {CKPT}; out {OUT}")

# ---------- data, identical to exp_v5_partitions
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
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
pat_mean = pd.DataFrame(X).groupby(sid).mean()
train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
comm = assign.loc[sid].values
groups = []
for c in range(CFG["k_communities"]):
    f_tr, f_te = is_train & (comm == c), is_test & (comm == c)
    if len(set(sid[f_tr])) >= CFG["min_patients"] and len(set(sid[f_te])) >= 50:
        groups.append(dict(c=c, key=f"community_trainfit_c{c:02d}", f_tr=f_tr, f_te=f_te))
if QUICK: groups = groups[:2]
log(f"{len(df)} admissions; {len(groups)} communities")

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
def losses(p, yy):
    p = np.clip(p, 1e-7, 1 - 1e-7); return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")

def perm_gap_test(l_in, l_out, pid_in, pid_out, r):
    """one-sided permutation test, at PATIENT level, that contributed records have lower loss than never-contributed records"""
    a = pd.Series(l_in).groupby(pid_in).mean().values; b = pd.Series(l_out).groupby(pid_out).mean().values
    if len(a) > CFG["max_side"]: a = r.choice(a, CFG["max_side"], False)
    if len(b) > CFG["max_side"]: b = r.choice(b, CFG["max_side"], False)
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); n_in = len(a); cnt = 0
    for _ in range(CFG["n_perm"]):
        r.shuffle(pool); cnt += (pool[:n_in].mean() - pool[n_in:].mean()) <= obs
    return float(obs), float((cnt + 1) / (CFG["n_perm"] + 1))

def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

for g in groups:
    path = f"{RUNS}/{g['key']}.json"
    if os.path.exists(path):
        try: json.load(open(path)); log(f"skip {g['key']} (done)"); continue
        except Exception: pass
    tg = time.time(); c, key = g["c"], g["key"]
    f_idx, h_idx = np.where(g["f_tr"])[0], np.where(g["f_te"])[0]
    r_tr = is_train & ~g["f_tr"]
    retrains = {s: cached(f"exact_{key}_s{s}", lambda s=s: train(X[r_tr], y[r_tr], s)) for s in CFG["retrain_seeds"]}
    cands = {}
    for s in CFG["seeds"]:
        for name, k in [("original", f"orig_s{s}"), ("finetune", f"ft_{key}_s{s}"), ("neggrad", f"ng_{key}_s{s}")]:
            m = load(k)
            if m is None: log(f"{key}: missing {k}"); continue
            cands[(name, s)] = m
    # predictions
    Xf, yf, Xh, yh = X[f_idx], y[f_idx], X[h_idx], y[h_idx]
    r = np.random.RandomState(5); div_rows = r.choice(len(f_idx), min(CFG["n_div_rows"], len(f_idx)), False)
    P_re = {s: (predict(m, Xf), predict(m, Xh)) for s, m in retrains.items()}
    P_ca = {k: (predict(m, Xf), predict(m, Xh)) for k, m in cands.items()}
    def stats(pf, ph):
        return dict(loss_contrib=float(losses(pf, yf).mean()), loss_heldout=float(losses(ph, yh).mean()),
                    auroc_contrib=auc(yf, pf), auroc_heldout=auc(yh, ph), gap=float(losses(pf, yf).mean() - losses(ph, yh).mean()))
    S_re = {s: stats(*P_re[s]) for s in retrains}
    rows = []
    def ref_audit(st, ref_seeds):
        out = {}
        for stat, side in [("loss_contrib", "low"), ("auroc_heldout", "high"), ("gap", "low")]:
            ref = np.array([S_re[s][stat] for s in ref_seeds]); m_, sd_ = ref.mean(), ref.std(ddof=1) + 1e-12
            z = (st[stat] - m_) / sd_
            thr = np.percentile(ref, 5) if side == "low" else np.percentile(ref, 95)
            flag = (st[stat] < thr) if side == "low" else (st[stat] > thr)
            out[stat] = dict(value=st[stat], ref_mean=float(m_), ref_sd=float(sd_), z=float(z), flag=bool(flag))
        return out
    seeds_all = list(retrains.keys())
    # null: each retrain audited against the other retrains
    for s in seeds_all:
        st = S_re[s]; ra = ref_audit(st, [t for t in seeds_all if t != s])
        lf, lh = losses(P_re[s][0], yf), losses(P_re[s][1], yh)
        obs, pval = perm_gap_test(lf, lh, sid[f_idx], sid[h_idx], np.random.RandomState(100 + s))
        rows.append(dict(model="exact_retrain", seed=s, **{f"A_{k}_{kk}": vv for k, v in ra.items() for kk, vv in v.items()},
                         B_gap=obs, B_p=pval, B_flag=bool(pval < CFG["alpha"]), **st))
    # candidates against all retrains
    for (name, s), (pf, ph) in P_ca.items():
        st = stats(pf, ph); ra = ref_audit(st, seeds_all)
        lf, lh = losses(pf, yf), losses(ph, yh)
        obs, pval = perm_gap_test(lf, lh, sid[f_idx], sid[h_idx], np.random.RandomState(200 + s))
        # divergence from retrains versus the retrain-to-retrain floor, on contributed records
        d_cand = np.mean([np.abs(pf[div_rows] - P_re[t][0][div_rows]).mean() for t in seeds_all])
        d_floor = [np.abs(P_re[a][0][div_rows] - P_re[b][0][div_rows]).mean() for i, a in enumerate(seeds_all) for b in seeds_all[i + 1:]]
        rows.append(dict(model=name, seed=s, **{f"A_{k}_{kk}": vv for k, v in ra.items() for kk, vv in v.items()},
                         B_gap=obs, B_p=pval, B_flag=bool(pval < CFG["alpha"]), div_from_retrains=float(d_cand),
                         noise_floor=float(np.mean(d_floor)), noise_floor_p95=float(np.percentile(d_floor, 95)),
                         div_ratio=float(d_cand / max(np.mean(d_floor), 1e-12)), **st))
    rec = dict(community=c, key=key, n_contrib_rows=int(len(f_idx)), n_contrib_patients=int(len(set(sid[f_idx]))),
               n_heldout_rows=int(len(h_idx)), n_heldout_patients=int(len(set(sid[h_idx]))), n_retrains=len(seeds_all),
               rows=rows, cfg=CFG, device=DEVICE, torch=torch.__version__, python=platform.python_version(),
               date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
    save_json(path, rec)
    fl = lambda name, k: np.mean([rr[k] for rr in rows if rr["model"] == name])
    log(f"{key}: A loss flag null {fl('exact_retrain','A_loss_contrib_flag'):.2f} orig {fl('original','A_loss_contrib_flag'):.2f} "
        f"ft {fl('finetune','A_loss_contrib_flag'):.2f} ng {fl('neggrad','A_loss_contrib_flag'):.2f} | B flag null {fl('exact_retrain','B_flag'):.2f} "
        f"orig {fl('original','B_flag'):.2f} ft {fl('finetune','B_flag'):.2f} ng {fl('neggrad','B_flag'):.2f} ({time.time()-tg:.0f}s)")

def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs"); return
    rows = []
    for rc in recs:
        for rr in rc["rows"]: rows.append(dict(community=rc["community"], n_contrib_patients=rc["n_contrib_patients"], **rr))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nflag rates by audit and model (fraction of community x seed cases flagged; exact_retrain rows are the null)")
    cols = ["A_loss_contrib_flag", "A_auroc_heldout_flag", "A_gap_flag", "B_flag"]
    T = R.groupby("model")[cols].mean().round(3); T["n"] = R.groupby("model").size(); print(T.to_string())
    print("\nmean z-scores (audit A) and mean gap (audit B)")
    print(R.groupby("model")[["A_loss_contrib_z", "A_auroc_heldout_z", "A_gap_z", "B_gap", "B_p"]].mean().round(3).to_string())
    print("\ndivergence from retrains relative to the retrain-to-retrain floor")
    D = R[R.model != "exact_retrain"].groupby("model")[["div_ratio"]].agg(["mean", "min", "max"]).round(2); print(D.to_string())
    per = R.groupby(["model", "community"])[cols].mean().reset_index()
    print("\nper-community power (majority of seeds flagged), audit A loss and audit B")
    for mname in ["original", "finetune", "neggrad"]:
        sub = per[per.model == mname]
        print(f"  {mname:10s} A_loss {(sub.A_loss_contrib_flag > 0.5).mean():.2f}  A_auroc {(sub.A_auroc_heldout_flag > 0.5).mean():.2f}  B {(sub.B_flag > 0.5).mean():.2f}")
    S = R.groupby("model")[cols + ["A_loss_contrib_z", "A_auroc_heldout_z", "B_gap", "B_p", "div_ratio"]].mean().reset_index()
    S.to_csv(f"{OUT}/summary.csv", index=False)
def export():
    z = f"{OUT}/v6_groupaudit_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
