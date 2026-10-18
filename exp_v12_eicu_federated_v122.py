"""
exp_v12_eicu_federated.py. Federated learning does not solve collective withdrawal; confinement inside a federation does.

v12.2 (training budget, matches v7.2): 10% of training patients are a validation set that no client trains on. FedAvg runs
up to 60 rounds and keeps the round with the lowest validation log loss (patience 5); the federation without a consortium
is stopped on validation stays outside that consortium. Best round is recorded per model. Nothing from v12.0/12.1 is reused.

eICU-CRD, same cohort, split, scaling and consortium partitions as exp_v7 (v7.1). Hospitals are the clients of a
FedAvg federation. For each consortium (all hospitals in a region, of a teaching status, or in a bed-size band):
  fl_global          FedAvg over every hospital; the federation's shared model before withdrawal
  fl_without         FedAvg over the hospitals outside the consortium; federated exact retraining after withdrawal
  fl_random          FedAvg with the same number of stays removed uniformly at random across the whole federation
  fl_resid           fl_without + residual logistic adapter on [logit, features] trained on the consortium's own stays,
                     held by the consortium and applied only to its patients (confinement inside the federation)
  fl_rep             fl_without + logistic head on the 128-d penultimate representation, same data, same custody
  fl_resid_random    fl_without + the same residual adapter trained on other hospitals' held-out stays (patient-matched to the
                     consortium's training patients, capped by availability); the control for "any adapter helps"
All arms are scored on the consortium's own held-out patients with the metrics of exp_v7; the top-decile threshold
is taken from the deployed system's composite full-test prediction. The adapters use the consortium's training stays,
which the consortium holds anyway under FL; the shared model fl_without never saw them (SHA-256 recorded per arm).
The data never leave the hospitals in either design; the difference is where the consortium's influence lives.

Env: ROOT (default ~/mimic), QUICK=1, NBOOT, DEVICE (cpu|mps|cuda override), ONLY_GROUPS (comma list of family/name). Reads ROOT/derived/eicu_cohort.parquet
and, if present, ROOT/results/v7_eicu_consortium_wholectrl/summary.csv for the centralised comparison in the summary.
Writes ROOT/results/v12_eicu_federated[_quick]/{runs,ckpt,results.csv,summary.csv,paired.json,zip}; resumes by skipping finished JSONs.
"""
import os, json, time, glob, zipfile, platform, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
COHORT = f"{ROOT}/derived/eicu_cohort.parquet"
V71 = f"{ROOT}/results/v7_eicu_consortium_es/summary.csv"
OUT = f"{ROOT}/results/v12_eicu_federated_es" + ("_quick" if QUICK else "")
RUNS, CKPT = f"{OUT}/runs", f"{OUT}/ckpt"; os.makedirs(RUNS, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], rounds=4 if QUICK else 60, patience=5, val_frac=0.1, local_epochs=1, lr=1e-3, batch=64,
           n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 200)), min_stays=800, fixed_cut=0.1, nb_thresholds=[0.05, 0.1, 0.15, 0.2, 0.3],
           adapter_C_grid=[0.01, 0.1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, version="v12.2", early_stopping="best round by validation log loss")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ---------- data, identical to exp_v7
df = pd.read_parquet(COHORT)
META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
feat = [c for c in df.columns if c not in META]
Xdf = df[feat].astype(float)
for c in Xdf.columns[Xdf.isna().any()]:
    Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32)
pid = df.uniquepid.values; hosp_id = df.hospitalid.values
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups)
test_p = set(ups[: int(0.2 * len(ups))]); is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
test_idx = np.where(is_test)[0]
rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
is_val = df.uniquepid.isin(set(tr_p[: int(CFG["val_frac"] * len(tr_p))])).values & is_train   # same split rule as v7.2
is_fit = is_train & ~is_val
log(f"{len(df)} stays, {X.shape[1]} features, mortality {y.mean():.3f}, train {is_train.sum()} (fit {is_fit.sum()}, validation {is_val.sum()}), test {is_test.sum()}")

# ---------- consortium partitions, identical to exp_v7
hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first()
labels = {}
for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
    vals = hattr[col].dropna().unique()
    for v in sorted(vals, key=str):
        hs = set(hattr.index[hattr[col] == v]); labels[(fam, str(v).replace(" ", "_").replace("/", "_"))] = hs
groups = []
for (fam, name), hs in labels.items():
    mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
    if f_tr.sum() < CFG["min_stays"] or f_te.sum() < 100: log(f"skip {fam}/{name}: {f_tr.sum()} training stays"); continue
    groups.append(dict(family=fam, name=name, hs=hs, f_tr=f_tr, f_te=f_te, mem=mem, n_hosp=len(hs), n_tr=int(f_tr.sum()), n_te=int(f_te.sum())))
if QUICK: groups = groups[:2]
only = os.environ.get("ONLY_GROUPS")
if only: groups = [g for g in groups if f"{g['family']}/{g['name']}" in only.split(",")]
log(f"{len(groups)} consortia: " + ", ".join(f"{g['family']}/{g['name']}({g['n_hosp']} hospitals, {g['n_tr']} stays)" for g in groups))

# ---------- clients: every hospital with training stays
train_idx = np.where(is_fit)[0]
clients = {h: train_idx[hosp_id[train_idx] == h] for h in np.unique(hosp_id[train_idx])}
log(f"{len(clients)} hospital clients, median {int(np.median([len(v) for v in clients.values()]))} training stays, "
    f"smallest {min(len(v) for v in clients.values())}, largest {max(len(v) for v in clients.values())}")

# ---------- model (as exp_v7) and FedAvg
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
Xt_all, yt_all = to_t(X), to_t(y)
ROUNDS_USED = {}
def fedavg(client_rows, seed, val_mask, tag=None):
    """FedAvg: each round every client starts from the global weights, trains local_epochs with Adam (batch 64, so a
    median hospital takes several steps per round), and the server averages the client weights in proportion to their
    training stays. Up to CFG["rounds"] rounds; the round with the lowest validation log loss on val_mask is kept."""
    torch.manual_seed(seed); np.random.seed(seed)
    g = MLP(X.shape[1]).to(DEVICE); bce = nn.BCEWithLogitsLoss()
    sizes = {h: len(r) for h, r in client_rows.items() if len(r) > 0}; total = sum(sizes.values()); B = CFG["batch"]
    rows_t = {h: to_t(r) for h, r in client_rows.items() if len(r) > 0}
    Xv, yv = to_t(X[val_mask]), to_t(y[val_mask]); best, best_state, best_rd, bad = float("inf"), None, 0, 0
    for rd in range(1, CFG["rounds"] + 1):
        acc = {k: torch.zeros_like(v) for k, v in g.state_dict().items()}; gstate = g.state_dict()
        for h, rt in rows_t.items():
            m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(gstate); m.train(); opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"])
            n = len(rt)
            for ep in range(CFG["local_epochs"]):
                perm = rt[torch.randperm(n, device=DEVICE)]
                for i in range(0, n, B):
                    idx = perm[i:i + B]; loss = bce(m(Xt_all[idx]), yt_all[idx]); opt.zero_grad(); loss.backward(); opt.step()
            w = sizes[h] / total
            for k, v in m.state_dict().items(): acc[k] += w * v
        g.load_state_dict(acc); g.eval()
        with torch.no_grad(): v = float(bce(g(Xv), yv))
        if v < best - 1e-5: best, best_state, best_rd, bad = v, {k: t.detach().clone() for k, t in g.state_dict().items()}, rd, 0
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    g.load_state_dict(best_state); g.eval()
    if tag: ROUNDS_USED[tag] = dict(best_round=best_rd, val_logloss=best, rounds_run=rd)
    return g
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def cached(key, fn):
    path = f"{CKPT}/{key}.pt"
    if os.path.exists(path):
        m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
        if os.path.exists(f"{CKPT}/{key}.json"): ROUNDS_USED[key] = json.load(open(f"{CKPT}/{key}.json"))
    else:
        m = fn(); torch.save(m.state_dict(), path + ".tmp"); os.replace(path + ".tmp", path)
        if key in ROUNDS_USED: json.dump(ROUNDS_USED[key], open(f"{CKPT}/{key}.json", "w"))
    return m, sha256(path)
def batched(fn, Xa, bs=65536):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), bs): out.append(fn(to_t(Xa[i:i + bs])).cpu().numpy())
    return np.concatenate(out)
def logits(m, Xa): m.eval(); return batched(lambda t: m(t), Xa).astype(np.float64)
def penult(m, Xa): m.eval(); return batched(lambda t: m.net[:4](t), Xa).astype(np.float32)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def fit_logistic_adapter(Z, yy, pp, seed):
    n_p = len(np.unique(pp)); grid = CFG["adapter_C_grid"]; chosen, cv = grid[len(grid) // 2], None
    if n_p >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pp)); cv = {}
        for C in grid:
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], yy[tr]); ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[C] = float(np.mean(ll))
        if cv: chosen = min(cv, key=cv.get)
    m = LogisticRegression(C=chosen, max_iter=2000).fit(Z, yy)
    return m, dict(C=chosen, cv_logloss=cv, n_params=int(Z.shape[1] + 1))

# ---------- metrics (as exp_v7, plus event counts)
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return float(e)
def auc(yy, pp): return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")
def cal_slope_intercept(yy, pp):
    pc = np.clip(pp, 1e-6, 1 - 1e-6); lg = np.log(pc / (1 - pc)).reshape(-1, 1)
    if len(np.unique(yy)) < 2: return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg, yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def threshold_metrics(yy, pp, thr):
    flag = pp >= thr; tp = float((flag & (yy == 1)).sum()); fp = float((flag & (yy == 0)).sum()); n = len(yy); ev = float((yy == 1).sum())
    return dict(flag_rate=float(flag.mean()), sensitivity=float(tp / max(ev, 1)), ppv=float(tp / max(flag.sum(), 1)), tp=tp, fp=fp, events=ev, n=n,
                missed_per_1000=float((ev - tp) / n * 1000), flagged_per_1000=float(flag.mean() * 1000)), flag
def net_benefit(yy, pp, pt):
    flag = pp >= pt; n = len(yy); tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum()
    return float(tp / n - fp / n * pt / (1 - pt))
def boot_patient(fn, yy, pp, pp_id):
    r = np.random.RandomState(1); ups_ = np.unique(pp_id); gi = {p: np.where(pp_id == p)[0] for p in ups_}; vals = []
    for _ in range(CFG["n_boot"]):
        i = np.concatenate([gi[p] for p in r.choice(ups_, len(ups_), True)]); vals.append(fn(yy[i], pp[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
def evaluate(p, p_full, g, p_ref_group, thr_ref):
    te = np.where(g["f_te"])[0]; yy, pp_id = y[te], pid[te]
    thr10 = float(np.percentile(p_full, 90))
    m10, flag10 = threshold_metrics(yy, p, thr10); mfix, _ = threshold_metrics(yy, p, CFG["fixed_cut"])
    slope, intercept = cal_slope_intercept(yy, p); alo, ahi = boot_patient(auc, yy, p, pp_id)
    out = dict(auroc=auc(yy, p), auroc_lo=alo, auroc_hi=ahi, ece=ece(yy, p), brier=float(np.mean((p - yy) ** 2)), cal_slope=slope,
               cal_intercept=intercept, mean_pred=float(p.mean()), observed=float(yy.mean()), full_test_auroc=auc(y[is_test], p_full),
               thr_top_decile=thr10, top_decile=m10, fixed_cut=mfix, net_benefit={str(pt): net_benefit(yy, p, pt) for pt in CFG["nb_thresholds"]})
    if p_ref_group is not None:
        _, flag_o = threshold_metrics(yy, p_ref_group, thr_ref)
        out["reclassified_top_decile"] = float((flag10 != flag_o).mean()); out["mean_abs_change"] = float(np.abs(p - p_ref_group).mean())
    return out
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

# ---------- run
globals_ = {}
cent = None
if os.path.exists(V71):
    V = pd.read_csv(V71); cent = float(V[V.model == "original"].full_test_auroc.mean())
for s in CFG["seeds"]:
    tg = time.time(); globals_[s] = cached(f"fl_global_s{s}", lambda s=s: fedavg(clients, s, is_val, tag=f"fl_global_s{s}"))
    log(f"fl_global seed {s} ready, full-test AUROC {auc(y[is_test], sigmoid(logits(globals_[s][0], X[is_test]))):.4f}"
        + (f" (centralised v7.2 original {cent:.4f})" if cent else "") + f" {ROUNDS_USED.get(f'fl_global_s{s}')} ({time.time()-tg:.0f}s)")
for gi_, g in enumerate(groups):
    key = f"{g['family']}_{g['name']}"; te = np.where(g["f_te"])[0]; pos = np.searchsorted(test_idx, te)
    tr_rows = np.where(g["f_tr"])[0]
    clients_without = {h: r for h, r in clients.items() if h not in g["hs"]}
    nong_te = test_idx[~np.isin(test_idx, te)]                                 # other hospitals' held-out stays: never seen by any shared model
    for s in CFG["seeds"]:
        path = f"{RUNS}/{g['family']}__{g['name']}__s{s}.json"; rec_old = None
        if os.path.exists(path):
            try: rec_old = json.load(open(path))
            except Exception: rec_old = None
            if rec_old is not None and "fl_resid_random" in rec_old["arms"]: log(f"skip {g['family']}/{g['name']} seed {s} (done)"); continue
        tg = time.time()
        fl_g, sha_g = globals_[s]
        fl_w, sha_w = cached(f"fl_without_{key}_s{s}", lambda: fedavg(clients_without, s, is_val & ~g["mem"], tag=f"fl_without_{key}_s{s}"))
        p_g_full = sigmoid(logits(fl_g, X[is_test])); thr_ref = float(np.percentile(p_g_full, 90)); p_ref = p_g_full[pos]
        z_w_full = logits(fl_w, X[is_test]); p_w_full = sigmoid(z_w_full); z_w_g = z_w_full[pos]
        arms, meta = {}, {}
        if rec_old is None:
            r = np.random.RandomState(300 + gi_); drop = set(r.choice(train_idx, g["n_tr"], False).tolist())   # same count, whole federation
            clients_random = {h: np.array([i for i in rows if i not in drop]) for h, rows in clients.items()}
            fl_r, sha_r = cached(f"fl_random_{key}_s{s}", lambda: fedavg(clients_random, s, is_val, tag=f"fl_random_{key}_s{s}"))
            p_r_full = sigmoid(logits(fl_r, X[is_test]))
            arms = {"fl_global": (p_ref, p_g_full, sha_g), "fl_without": (sigmoid(z_w_g), p_w_full, sha_w), "fl_random": (p_r_full[pos], p_r_full, sha_r)}
            meta = {k: {} for k in arms}
            z_tr = logits(fl_w, X[tr_rows]); h_tr = penult(fl_w, X[tr_rows]); h_g = penult(fl_w, X[te]); yr, pr = y[tr_rows], pid[tr_rows]
            ad, info = fit_logistic_adapter(np.column_stack([z_tr, X[tr_rows]]), yr, pr, s)
            p_a = ad.predict_proba(np.column_stack([z_w_g, X[te]]))[:, 1]; p_full = p_w_full.copy(); p_full[pos] = p_a
            arms["fl_resid"] = (p_a, p_full, sha_w); meta["fl_resid"] = dict(n_rows=int(len(tr_rows)), n_patients=int(len(np.unique(pr))), **info)
            ad, info = fit_logistic_adapter(h_tr, yr, pr, s)
            p_a = ad.predict_proba(h_g)[:, 1]; p_full = p_w_full.copy(); p_full[pos] = p_a
            arms["fl_rep"] = (p_a, p_full, sha_w); meta["fl_rep"] = dict(n_rows=int(len(tr_rows)), n_patients=int(len(np.unique(pr))), **info)
        # control: the same residual adapter trained on other hospitals' held-out stays, patient-matched to the consortium, applied to the consortium
        rc = np.random.RandomState(500 + gi_ + 31 * s); ng_p = np.unique(pid[nong_te]); rc.shuffle(ng_p)
        n_want = int(len(np.unique(pid[tr_rows]))); ctrl_rows = nong_te[np.isin(pid[nong_te], ng_p[: min(n_want, len(ng_p))])]
        Zc = np.column_stack([z_w_full[np.searchsorted(test_idx, ctrl_rows)], X[ctrl_rows]])
        ad, info = fit_logistic_adapter(Zc, y[ctrl_rows], pid[ctrl_rows], s)
        p_a = ad.predict_proba(np.column_stack([z_w_g, X[te]]))[:, 1]; p_full = p_w_full.copy(); p_full[pos] = p_a
        arms["fl_resid_random"] = (p_a, p_full, sha_w)
        meta["fl_resid_random"] = dict(n_rows=int(len(ctrl_rows)), n_patients=int(len(np.unique(pid[ctrl_rows]))), n_patients_wanted=n_want, **info)
        res = {} if rec_old is None else rec_old["arms"]
        for name, (p, p_full, sha) in arms.items():
            res[name] = evaluate(p, p_full, g, None if name == "fl_global" else p_ref, thr_ref); res[name].update(meta[name]); res[name]["shared_model_sha256"] = sha
        if rec_old is None:
            rec = dict(family=g["family"], group=g["name"], seed=s, n_hospitals=g["n_hosp"], n_train_rows=g["n_tr"], n_test_rows=g["n_te"],
                       n_test_patients=int(len(set(pid[te]))), frac_training=float(g["n_tr"] / is_fit.sum()), n_clients_total=len(clients),
                       n_val_rows=int(is_val.sum()), rounds=dict(fl_global=ROUNDS_USED.get(f"fl_global_s{s}"), fl_without=ROUNDS_USED.get(f"fl_without_{key}_s{s}"), fl_random=ROUNDS_USED.get(f"fl_random_{key}_s{s}")),
                       n_clients_without=len(clients_without), n_random_rows_removed=len(drop), arms=res, cfg=CFG, device=DEVICE,
                       torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
        else:
            rec = rec_old; rec["arms"] = res; rec["control_added"] = time.strftime("%Y-%m-%d %H:%M:%S"); rec["cfg_control"] = CFG
        save_json(path, rec)
        a0 = res["fl_global"]["auroc"]
        log(f"{g['family']}/{g['name']} seed {s}: fl_global {a0:.4f} | vs global: without {res['fl_without']['auroc']-a0:+.4f} random {res['fl_random']['auroc']-a0:+.4f} "
            f"resid {res['fl_resid']['auroc']-a0:+.4f} rep {res['fl_rep']['auroc']-a0:+.4f} ctrl {res['fl_resid_random']['auroc']-a0:+.4f}; "
            f"best rounds {dict((k, (v or {}).get('best_round')) for k, v in rec.get('rounds', {}).items())} ({time.time()-tg:.0f}s)")

# ---------- summary from disk
def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs yet"); return None
    rows = []
    for r in recs:
        a0, e0 = r["arms"]["fl_global"]["auroc"], r["arms"]["fl_global"]["ece"]
        for name, m in r["arms"].items():
            rows.append(dict(family=r["family"], group=r["group"], seed=r["seed"], model=name, n_hospitals=r["n_hospitals"], n_train_rows=r["n_train_rows"],
                             frac_training=r["frac_training"], auroc=m["auroc"], d_auroc=m["auroc"] - a0, ece=m["ece"], d_ece=m["ece"] - e0, brier=m["brier"],
                             cal_slope=m["cal_slope"], full_test_auroc=m["full_test_auroc"], sens_top_decile=m["top_decile"]["sensitivity"],
                             ppv_top_decile=m["top_decile"]["ppv"], missed_per_1000_top_decile=m["top_decile"]["missed_per_1000"],
                             sens_fixed=m["fixed_cut"]["sensitivity"], ppv_fixed=m["fixed_cut"]["ppv"], shared_model_sha256=m["shared_model_sha256"]))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results.csv", index=False)
    S = R.groupby(["family", "group", "model"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 220)
    W = S.pivot_table(index=["family", "group"], columns="model", values="d_auroc")
    info = S[S.model == "fl_global"].set_index(["family", "group"])[["n_hospitals", "n_train_rows", "frac_training", "full_test_auroc"]]
    print("\nchange in AUROC on the consortium's own held-out patients vs the federation's shared model (fl_global), seed means")
    cols = [c for c in ["fl_without", "fl_random", "fl_resid", "fl_rep", "fl_resid_random"] if c in W.columns]
    print(pd.concat([info.round(3), W[cols].round(4)], axis=1).to_string())
    out = {}; r = np.random.RandomState(0)
    def paired(d):
        d = np.asarray(d, float); d = d[~np.isnan(d)]
        if len(d) < 2: return dict(d=float(d.mean()) if len(d) else float("nan"), ci=[float("nan")] * 2, frac_worse=float("nan"), n=int(len(d)))
        bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
        return dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), n=int(len(d)))
    print("\npaired over consortia (AUROC on the consortium's held-out patients)")
    for a, b, label in [("fl_without", "fl_random", "federated withdrawal minus matched random removal (the FL penalty)"),
                        ("fl_resid", "fl_without", "confinement (resid) minus federated withdrawal (recovered)"),
                        ("fl_rep", "fl_without", "confinement (rep) minus federated withdrawal (recovered)"),
                        ("fl_resid", "fl_global", "confinement (resid) minus the shared model before withdrawal (sharing premium)"),
                        ("fl_rep", "fl_global", "confinement (rep) minus the shared model before withdrawal (sharing premium)"),
                        ("fl_resid_random", "fl_without", "control adapter (other hospitals' data) minus federated withdrawal"),
                        ("fl_resid", "fl_resid_random", "confinement (resid) minus the control adapter (consortium-specific gain)")]:
        if a in W.columns and b in W.columns:
            p = paired(W[a] - W[b]); out[f"{a}_vs_{b}"] = p
            print(f"  {label:78s} {p['d']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] worse {100*p['frac_worse']:.0f}% (n={p['n']})")
    E = S.pivot_table(index=["family", "group"], columns="model", values="d_ece"); D = S.pivot_table(index=["family", "group"], columns="model", values="missed_per_1000_top_decile")
    print("  ECE change vs fl_global: " + ", ".join(f"{c} {paired(E[c])['d']:+.4f}" for c in cols))
    print("  missed events per 1,000 consortium stays at the system top-decile threshold, change vs fl_global: " + ", ".join(f"{c} {paired(D[c]-D['fl_global'])['d']:+.1f}" for c in cols))
    if os.path.exists(V71):
        V = pd.read_csv(V71); VW = V.pivot_table(index=["family", "group"], columns="model", values="d_auroc")
        both = W.join(VW[["exact_group", "exact_random"]], how="inner")
        if len(both):
            pc, pf = paired(both["exact_group"] - both["exact_random"]), paired(both["fl_without"] - both["fl_random"])
            print(f"\ncentralised (v7.2) vs federated penalty on the same consortia (n={len(both)}): centralised {pc['d']:+.4f} [{pc['ci'][0]:+.4f},{pc['ci'][1]:+.4f}], "
                  f"federated {pf['d']:+.4f} [{pf['ci'][0]:+.4f},{pf['ci'][1]:+.4f}]")
            out["centralised_penalty_v71"] = pc; out["federated_penalty"] = pf
    with open(f"{OUT}/paired.json", "w") as f: json.dump(out, f, indent=1)
    return S
def export():
    z = f"{OUT}/v12_eicu_federated_es_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv", f"{OUT}/paired.json"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
