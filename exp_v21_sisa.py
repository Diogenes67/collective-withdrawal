"""
exp_v21_sisa.py. Group-aligned sharded training (SISA, Bourtoule et al. 2021) as a design for collective withdrawal, against
the pooled model and contribution confinement. MIMIC-IV, 25 clustered groups x 3 seeds.

Random SISA shards would place a group of hundreds to thousands of patients in every shard, so withdrawal would retrain all
of them. The variant that makes collective withdrawal a deletion uses the groups themselves as shards: the 25 k-means clusters
partition the training patients, one multilayer perceptron is trained per shard, and the ensemble averages the shard models'
predicted probabilities. Withdrawal of a group drops its shard; no model is retrained.

Each shard model has the architecture of the pooled model and is trained from scratch on its shard's training admissions
(Adam, lr 1e-3, batch 256, early stopping on 20 percent of the shard's patients, patience 5, at most 50 epochs), so small
shards are not undertrained by a fixed epoch budget. Preprocessing is the standard whole-training pipeline used for the
pooled model (diagnosis vocabulary, imputation and scaling fitted on all training rows); this is weaker than v16's
retained-only pipeline and is stated as such.

Arms on each group's held-out patients: pooled (v5 original), sisa_full (all 25 shards, uniform average), sisa_minus (the
group's shard dropped), sisa_full_w and sisa_minus_w (average weighted by shard training admissions), own_shard (the group's
shard model alone). When results/v19_protocol/preds exist, confinement arms are read from them on the same patients: chosen
(base + validation-chosen component), strict_base, pooled_cf (best pooled system). Effect on other patients: AUROC on all
held-out patients outside the group under sisa_full and sisa_minus, and the mean absolute change in their predicted
probability when the group's shard is deleted (zero for confinement by construction).

Env: ROOT, QUICK=1, DEVICE, SEED (train one seed and exit), SUMMARY=0 (skip summary), NBOOT. Reads derived/cohort_latest.parquet,
results/v5_partitions/{runs,ckpt}, results/v19_protocol/preds (optional). Writes results/v21_sisa/{ckpt, preds, runs, summary.csv, intervals.json}.
"""
import os, json, time, glob, copy, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); V5 = f"{ROOT}/results/v5_partitions"
OUT = f"{ROOT}/results/v21_sisa" + ("_quick" if QUICK else ""); V19 = f"{ROOT}/results/v19_protocol"
for d in ["ckpt", "preds", "runs"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], lr=1e-3, batch=256, max_epochs=3 if QUICK else 50, patience=5, val_frac=0.2,
           min_patients=300, n_shards=25, n_boot=int(os.environ.get("NBOOT", 50 if QUICK else 1000)), version="v21.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
ONLY_SEED = os.environ.get("SEED"); DO_SUMMARY = os.environ.get("SUMMARY", "1") == "1"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ---------------------------------------------------------------- data, split, pipeline and groups exactly as v5/v19
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
feat_lab = [c for c in feat if c.startswith("lab_")]
y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test; test_idx = np.where(is_test)[0]
Xs = df[feat].astype(float)
for c in feat_lab: Xs[c + "_missing"] = Xs[c].isna().astype(float); Xs[c] = Xs[c].fillna(Xs[c].median())
Xs = Xs.values.astype(np.float32); mu0, sd0 = Xs[is_train].mean(0), Xs[is_train].std(0) + 1e-6; Xs = (Xs - mu0) / sd0
pat_mean = pd.DataFrame(Xs).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
cluster = assign.reindex(sid).values                                   # cluster of every admission (train and held-out)
shards = [f"c{c:02d}" for c in range(CFG["n_shards"])]
shard_rows = {sh: np.where(is_train & (cluster == c))[0] for c, sh in enumerate(shards)}
assert sum(len(v) for v in shard_rows.values()) == is_train.sum(), "shards must partition the training admissions"
groups = []
for c, sh in enumerate(shards):
    mem = cluster == c; f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    ref = f"{V5}/runs/community_trainfit__{sh}__s0.json"
    if os.path.exists(ref):
        r = json.load(open(ref))
        if r["n_train_patients"] != n_tr_p or r["n_test_patients"] != n_te_p: raise SystemExit(f"{sh}: membership differs from v5")
    groups.append(dict(key=f"community_trainfit/{sh}", shard=sh, c=c, f_te=f_te, mem=mem, n_tr_p=n_tr_p, n_te_p=n_te_p))
log(f"{len(shards)} shards partition {is_train.sum()} training admissions; {len(groups)} groups evaluated")

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def load(path, d):
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def probs(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def train_es(rows, seed):
    """From scratch on rows, early stopping on a patient-level 20 percent of rows."""
    r = np.random.RandomState(seed); ups = np.unique(sid[rows]); r.shuffle(ups); val_p = set(ups[: max(1, int(CFG["val_frac"] * len(ups)))])
    va = np.isin(sid[rows], list(val_p)); tr_ = ~va
    torch.manual_seed(seed); m = MLP(Xs.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt, Xv, yv = to_t(Xs[rows][tr_]), to_t(y[rows][tr_]), to_t(Xs[rows][va]), to_t(y[rows][va]); n, B = int(tr_.sum()), CFG["batch"]
    def vl():
        m.eval()
        with torch.no_grad(): v = float(bce(m(Xv), yv))
        m.train(); return v
    m.train(); best, best_state, bad, ep_best = vl(), copy.deepcopy(m.state_dict()), 0, 0
    for ep in range(1, CFG["max_epochs"] + 1):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B): idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        v = vl()
        if v < best - 1e-5: best, best_state, bad, ep_best = v, copy.deepcopy(m.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    m.load_state_dict(best_state); m.eval(); return m, dict(best_val_loss=best, best_epoch=ep_best, epochs_run=ep, n_rows=len(rows), n_patients=len(ups))
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
def free():
    gc.collect()
    if DEVICE == "mps": torch.mps.empty_cache()

# ---------------------------------------------------------------- training: one model per shard per seed, held-out probabilities cached
seeds_to_run = [int(ONLY_SEED)] if ONLY_SEED is not None else CFG["seeds"]
for s in seeds_to_run:
    mat = f"{OUT}/preds/heldout_matrix_s{s}.npz"
    if os.path.exists(mat): log(f"s{s}: held-out matrix exists"); continue
    info = {}
    for sh in shards:
        ck = f"{OUT}/ckpt/shard_{sh}_s{s}.pt"; meta = f"{OUT}/ckpt/shard_{sh}_s{s}.json"
        if os.path.exists(ck) and os.path.exists(meta): info[sh] = json.load(open(meta)); continue
        ts = time.time(); m, inf = train_es(shard_rows[sh], 1000 * s + int(sh[1:])); torch.save(m.state_dict(), ck); inf["seconds"] = time.time() - ts
        save_json(meta, inf); info[sh] = inf; del m; free()
        log(f"s{s} shard {sh}: {inf['n_patients']} patients, {inf['n_rows']} admissions, best epoch {inf['best_epoch']} of {inf['epochs_run']} ({inf['seconds']:.0f}s)")
    Pm = np.zeros((len(test_idx), len(shards)), np.float32)
    for j, sh in enumerate(shards):
        m = load(f"{OUT}/ckpt/shard_{sh}_s{s}.pt", Xs.shape[1]); Pm[:, j] = probs(m, Xs[test_idx]); del m; free()
    orig = load(f"{V5}/ckpt/orig_s{s}.pt", Xs.shape[1]); pooled = probs(orig, Xs[test_idx]); del orig; free()
    w = np.array([info[sh]["n_rows"] for sh in shards], np.float64)
    np.savez_compressed(mat, P=Pm, pooled=pooled, w=w, y=y[test_idx], pid=sid[test_idx], cluster=cluster[test_idx])
    full = Pm.mean(1); log(f"s{s}: held-out AUROC pooled {auc(y[test_idx], pooled):.4f}, SISA uniform {auc(y[test_idx], full):.4f}, "
                          f"SISA weighted {auc(y[test_idx], Pm @ (w / w.sum())):.4f}")
if ONLY_SEED is not None and not DO_SUMMARY: raise SystemExit(0)

# ---------------------------------------------------------------- per group and seed arms
seeds = [s for s in CFG["seeds"] if os.path.exists(f"{OUT}/preds/heldout_matrix_s{s}.npz")]
if not seeds: raise SystemExit("no held-out matrices yet")
M = {s: dict(np.load(f"{OUT}/preds/heldout_matrix_s{s}.npz")) for s in seeds}
yte, pte, cte = M[seeds[0]]["y"], M[seeds[0]]["pid"], M[seeds[0]]["cluster"]
full_test = {s: dict(pooled=auc(yte, M[s]["pooled"]), sisa=auc(yte, M[s]["P"].mean(1)), sisa_w=auc(yte, M[s]["P"] @ (M[s]["w"] / M[s]["w"].sum()))) for s in seeds}
ARMS = ["pooled", "sisa_full", "sisa_minus", "sisa_full_w", "sisa_minus_w", "own_shard"]; CONF = ["chosen", "strict_base", "pooled_cf"]
have_conf = True
for g in groups:
    tag = g["key"].replace("/", "__"); te = cte == g["c"]; others = ~te
    for s in seeds:
        path = f"{OUT}/runs/{tag}__s{s}.json"
        Pm, w = M[s]["P"].astype(np.float64), M[s]["w"]; j = g["c"]; keep = np.arange(Pm.shape[1]) != j
        full, minus = Pm.mean(1), Pm[:, keep].mean(1)
        full_w, minus_w = Pm @ (w / w.sum()), Pm[:, keep] @ (w[keep] / w[keep].sum())
        p = dict(pooled=M[s]["pooled"][te], sisa_full=full[te], sisa_minus=minus[te], sisa_full_w=full_w[te], sisa_minus_w=minus_w[te], own_shard=Pm[te, j])
        f19 = f"{V19}/preds/{tag}__s{s}.npz"; k_flag = int(round(0.1 * te.sum()))
        if os.path.exists(f19):
            Z = np.load(f19); assert np.array_equal(Z["pid"], pte[te]), f"{tag}: v19 held-out order differs"
            if not np.allclose(Z["pooled"], p["pooled"], atol=1e-3): log(f"  warning {tag} s{s}: pooled differs from v19 by up to {np.abs(Z['pooled'] - p['pooled']).max():.4f}")
            for a in CONF: p[a] = Z[a]
            k_flag = int(Z["k_flag"])
        else: have_conf = False
        yy = yte[te]
        np.savez_compressed(f"{OUT}/preds/{tag}__s{s}.npz", y=yy, pid=pte[te], k_flag=k_flag, **p)
        rec = dict(group=g["key"], seed=s, n_train_patients=g["n_tr_p"], n_test_patients=g["n_te_p"],
                   arms={a: dict(auroc=auc(yy, v), ece=ece(yy, v)) for a, v in p.items()},
                   others=dict(n=int(others.sum()), auroc_sisa_full=auc(yte[others], full[others]), auroc_sisa_minus=auc(yte[others], minus[others]),
                               auroc_pooled=auc(yte[others], M[s]["pooled"][others]), mean_abs_change=float(np.abs(full[others] - minus[others]).mean()),
                               max_abs_change=float(np.abs(full[others] - minus[others]).max()), mean_abs_change_w=float(np.abs(full_w[others] - minus_w[others]).mean())),
                   full_test=full_test[s], cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_json(path, rec)
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
rows = [dict(group=r["group"], seed=r["seed"], arm=a, **m) for r in recs for a, m in r["arms"].items()]
S = pd.DataFrame(rows).groupby(["group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="group", columns="arm", values="auroc"); show = [a for a in ARMS + CONF if a in W]
print("\nAUROC on each group's held-out patients, seed means"); print(W[show].round(4).to_string())
O = pd.DataFrame([dict(group=r["group"], seed=r["seed"], **r["others"]) for r in recs]).groupby("group").mean(numeric_only=True)
print("\nOther patients when the group's shard is deleted (seed means)"); print(O[["auroc_pooled", "auroc_sisa_full", "auroc_sisa_minus", "mean_abs_change", "max_abs_change"]].round(5).to_string())
print("\nfull held-out AUROC by seed:", {s: {k: round(v, 4) for k, v in d.items()} for s, d in full_test.items()})

# ---------------------------------------------------------------- two-stage bootstrap (groups, then patients), as v19
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
P = {}
for f in glob.glob(f"{OUT}/preds/community_trainfit__*.npz"):
    Z = np.load(f); g, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(g.replace("__", "/"), int(s_))] = {k: Z[k] for k in Z.files}
names = sorted(set(g for g, _ in P if all((g, s_) in P for s_ in seeds)))
arms_b = ARMS + (CONF if have_conf else [])
pairs = {"SISA full minus pooled (state 1 under SISA)": ("sisa_full", "pooled"),
         "SISA shard deleted minus SISA full (cost of withdrawal under SISA)": ("sisa_minus", "sisa_full"),
         "SISA shard deleted minus pooled": ("sisa_minus", "pooled"),
         "SISA weighted full minus pooled": ("sisa_full_w", "pooled"),
         "SISA weighted shard deleted minus pooled": ("sisa_minus_w", "pooled"),
         "own shard alone minus pooled": ("own_shard", "pooled")}
if have_conf:
    pairs.update({"confined (chosen component) minus SISA full": ("chosen", "sisa_full"),
                  "confinement base minus SISA shard deleted (state 3 vs state 3)": ("strict_base", "sisa_minus"),
                  "best pooled system minus SISA full": ("pooled_cf", "sisa_full")})
def stats(g, idx):
    o = {}
    for a in arms_b:
        o[a] = (np.nanmean([fast_auc(P[(g, s_)]["y"][idx], P[(g, s_)][a][idx]) for s_ in seeds]),
                np.mean([missed_at_k(P[(g, s_)]["y"][idx], P[(g, s_)][a][idx], max(1, int(round(int(P[(g, s_)]["k_flag"]) * len(idx) / len(P[(g, s_)]["y"]))))) for s_ in seeds]))
    return o
def contrasts(st):
    d = {k: np.mean([o[a][0] - o[b][0] for o in st]) for k, (a, b) in pairs.items()}
    d["events unflagged per 1,000 at matched flag rate, SISA full minus pooled"] = np.mean([o["sisa_full"][1] - o["pooled"][1] for o in st])
    if have_conf: d["events unflagged per 1,000 at matched flag rate, confined minus SISA full"] = np.mean([o["chosen"][1] - o["sisa_full"][1] for o in st])
    return d
if names and DO_SUMMARY:
    pt = contrasts([stats(g, np.arange(len(P[(g, seeds[0])]["y"]))) for g in names]); pat_idx = {}
    for g in names:
        u, inv = np.unique(P[(g, seeds[0])]["pid"], return_inverse=True); pat_idx[g] = (u, [np.where(inv == i)[0] for i in range(len(u))])
    r = np.random.RandomState(0); boots = {k: [] for k in pt}
    for b in range(CFG["n_boot"]):
        st = []
        for g in r.choice(names, len(names), True):
            u, gidx = pat_idx[g]; idx = np.concatenate([gidx[j] for j in r.choice(len(u), len(u), True)]); st.append(stats(g, idx))
        for k, v in contrasts(st).items(): boots[k].append(v)
        if (b + 1) % 200 == 0: log(f"  bootstrap draw {b + 1}")
    res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(names), n_boot=CFG["n_boot"]) for k in pt}
    lw = {k: int(sum((W.loc[g, a] - W.loc[g, b]) < 0 for g in names)) for k, (a, b) in pairs.items()}
    print(f"\n== {len(names)} clustered groups, two-stage bootstrap ({CFG['n_boot']} draws)")
    for k, v in res.items(): print(f"  {k:90s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" + (f"  lower in {lw[k]} of {len(names)}" if k in lw else ""))
    oth = dict(mean_abs_change_mean=float(O.mean_abs_change.mean()), max_abs_change_max=float(O.max_abs_change.max()),
               others_auroc_full_minus_minus_mean=float((O.auroc_sisa_full - O.auroc_sisa_minus).mean()))
    print("\nother patients:", {k: round(v, 5) for k, v in oth.items()})
    save_json(f"{OUT}/intervals.json", dict(results=res, lower_in=lw, others=oth, full_test=full_test, confinement_arms=have_conf, date=time.strftime("%Y-%m-%d %H:%M:%S")))
log(f"done, results in {OUT}")
