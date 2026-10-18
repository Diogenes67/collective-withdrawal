"""
exp_v11_confinement.py. Contribution confinement on MIMIC-IV.

The shared model is trained without the group (the exact retrain from exp_v5). The group's contribution lives in a
component the group holds and that is applied only to its own patients. Withdrawal is deletion of that component,
so nothing needs unlearning and nothing needs a reference-model audit; the base checkpoint's SHA-256 is recorded with
every arm. The experiment asks whether a group-held component recovers what exact retraining cost the group, and what
the group gives up by not pooling.

Arms, all scored on the group's own held-out patients (same cohort, split, scaling and partitions as exp_v5):
  original          pooled model trained on everything (v5 orig_s{seed})
  base              exact retrain without the group (v5 exact_{group}_s{seed}); the shared model under confinement
  recal_retained    base + Platt recalibration fitted on non-group held-out patients; no group data; applied system-wide
  resid_random      base + residual logistic adapter trained on non-group held-out patients (patient-matched); control
  platt_group@f     base + Platt scaling fitted on the group's own training admissions (2 parameters)
  resid_group@f     base + residual logistic adapter on [base logit, features], group training admissions
  rep_group@f       base + logistic head on the base's 128-d penultimate representation, group training admissions
  full_group@f      base copied and fine-tuned on the group's training admissions, early-stopped on 20% of its patients
  f in CFG["fractions"]: the fraction of the group's training patients the group holds (nested subsets per seed).
Group arms are applied to group patients only; the deployed system is the base for everyone else, so the top-decile
threshold comes from the composite full-test prediction.

Env: ROOT (default ~/mimic), QUICK=1, NBOOT (override bootstrap count), DEVICE (cpu|mps|cuda override), ONLY_GROUPS (comma list of family/name to run; GROUPS is a bash builtin).
Reads ROOT/derived/cohort_latest.parquet, ROOT/results/v5_partitions/{ckpt,runs}. Checks every group's membership
against the v5 run JSONs before using a v5 checkpoint. Writes ROOT/results/v11_confinement[_quick]/{runs,results.csv,
summary.csv,paired.json,zip}; one JSON per (group, seed), written atomically; resumes by skipping finished JSONs.
"""
import os, json, time, glob, zipfile, platform, hashlib, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
COHORT = os.environ.get("COHORT", f"{ROOT}/derived/cohort_latest.parquet")
V5 = f"{ROOT}/results/v5_partitions"; V5_CKPT, V5_RUNS = f"{V5}/ckpt", f"{V5}/runs"
OUT = f"{ROOT}/results/v11_confinement_1" + ("_quick" if QUICK else "")
RUNS = f"{OUT}/runs"; os.makedirs(RUNS, exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 200)),
           k_communities=25, min_patients=300, fractions=[1.0] if QUICK else [0.1, 0.25, 0.5, 1.0],
           adapter_C_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], adapter_cv_folds=3, adapter_min_patients_cv=30, logit_scale=100.0,
           full_lr=1e-4, full_batch=256, full_max_epochs=30, full_patience=5, full_val_frac=0.2,
           fixed_cut=0.3, nb_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4], version="v11.1")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"device {DEVICE}; quick={QUICK}; out {OUT}")

# ============================================================== data (identical to exp_v5)
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
test_idx = np.where(is_test)[0]
log(f"{len(df)} admissions, {len(pids)} patients, {X.shape[1]} features, train {is_train.sum()}, test {is_test.sum()}")

def modal_label(col):
    cnt = df.groupby(["subject_id", col]).size().reset_index(name="n")
    cnt = cnt.sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id")
    return cnt.set_index("subject_id")[col]
labels = {}
race = modal_label("race_group")
if (race == "native").any(): labels[("race", "native")] = set(race.index[race == "native"])
if "english" in df.columns:
    eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
if not QUICK:
    pat_mean = pd.DataFrame(X).groupby(sid).mean()
    train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=CFG["k_communities"], n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values)
    assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    for c in range(CFG["k_communities"]):
        labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])

groups = []
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype))
    f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p = len(set(sid[f_tr])); n_te_p = len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50:
        log(f"skip {fam}/{name}: {n_tr_p} training patients, {n_te_p} held-out patients"); continue
    groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p, n_te_p=n_te_p))
only = os.environ.get("ONLY_GROUPS")
if only: groups = [g for g in groups if f"{g['family']}/{g['name']}" in only.split(",")]
log(f"{len(groups)} groups: " + ", ".join(f"{g['family']}/{g['name']}({g['n_tr_p']})" for g in groups))

# ---------- membership check against the v5 run records (same clustering, same split) before any v5 checkpoint is used
for g in groups:
    ref = f"{V5_RUNS}/{g['family']}__{g['name']}__s0.json"
    if not os.path.exists(ref): raise SystemExit(f"no v5 run record {ref}; the v5 checkpoints cannot be matched to this group")
    r = json.load(open(ref))
    if r["n_train_patients"] != g["n_tr_p"] or r["n_test_patients"] != g["n_te_p"] or r["n_train_rows"] != int(g["f_tr"].sum()):
        raise SystemExit(f"{g['family']}/{g['name']}: membership differs from v5 (train patients {g['n_tr_p']} vs {r['n_train_patients']}, "
                         f"test patients {g['n_te_p']} vs {r['n_test_patients']}, train rows {int(g['f_tr'].sum())} vs {r['n_train_rows']}); "
                         f"re-cluster drift, do not reuse the v5 checkpoints")
log("membership of every group matches the v5 run records")

# ============================================================== model (as exp_v5)
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def load_v5(key):
    path = f"{V5_CKPT}/{key}.pt"
    if not os.path.exists(path): raise SystemExit(f"missing v5 checkpoint {path}")
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
    return m, sha256(path)
def batched(fn, Xa, bs=65536):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), bs): out.append(fn(to_t(Xa[i:i + bs])).cpu().numpy())
    return np.concatenate(out)
def logits(m, Xa): m.eval(); return batched(lambda t: m(t), Xa).astype(np.float64)
def penult(m, Xa): m.eval(); return batched(lambda t: m.net[:4](t), Xa).astype(np.float32)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

# ============================================================== group-held components
def fit_platt(z, yy):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), yy)
    return dict(a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))
def apply_platt(p, z): return sigmoid(p["a"] * z + p["b"])
class Platt:
    """Logistic on the first column only (the shared logit); monotone, so it leaves the ranking unchanged."""
    def __init__(self, Z, yy): self.m = LogisticRegression(C=1e6, max_iter=1000).fit(Z[:, :1], yy)
    def predict_proba(self, Z): return self.m.predict_proba(Z[:, :1])
def fit_logistic_adapter(Z, yy, pid, seed, offset=False):
    """Ridge logistic adapter, C chosen by patient-grouped CV log loss. offset=True: the first column is the shared logit, scaled by
    logit_scale so the L2 penalty barely touches its coefficient, and a logit-only (Platt) candidate competes in the CV, so the adapter
    cannot be chosen when leaving the shared ranking alone does better in CV. (v14.1 adapter; the earlier version penalised the logit.)"""
    Z = Z.copy()
    if offset: Z[:, 0] *= CFG["logit_scale"]
    n_p = len(np.unique(pid)); grid = CFG["adapter_C_grid"]; chosen, cv = "platt" if offset else grid[len(grid) // 2], None
    if n_p >= CFG["adapter_min_patients_cv"] and len(np.unique(yy)) > 1:
        folds = list(GroupKFold(n_splits=CFG["adapter_cv_folds"]).split(Z, yy, pid)); cv = {}
        for C in (["platt"] if offset else []) + list(grid):
            ll = []
            for tr, va in folds:
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[va])) < 2: continue
                m = Platt(Z[tr], yy[tr]) if C == "platt" else LogisticRegression(C=C, max_iter=3000).fit(Z[tr], yy[tr])
                ll.append(log_loss(yy[va], m.predict_proba(Z[va])[:, 1], labels=[0, 1]))
            if ll: cv[str(C)] = float(np.mean(ll))
        if cv: chosen = min(cv, key=cv.get); chosen = "platt" if chosen == "platt" else float(chosen)
    m = Platt(Z, yy) if chosen == "platt" else LogisticRegression(C=chosen, max_iter=3000).fit(Z, yy)
    class Wrapped:
        def predict_proba(self, Zn):
            Zn = Zn.copy()
            if offset: Zn[:, 0] *= CFG["logit_scale"]
            return m.predict_proba(Zn)
    return Wrapped(), dict(C=chosen, cv_logloss=cv, n_params=int(Z.shape[1] + 1) if chosen != "platt" else 2)
def finetune_copy(base, rows, pid, seed):
    """Full copy of the base fine-tuned on the group's rows; early-stopped on a 20% patient split of those rows."""
    r = np.random.RandomState(seed); ups = np.unique(pid); r.shuffle(ups)
    val_p = set(ups[: max(1, int(CFG["full_val_frac"] * len(ups)))]); va = np.isin(pid, list(val_p)); tr = ~va
    Xtr, ytr, Xva, yva = X[rows][tr], y[rows][tr], X[rows][va], y[rows][va]
    torch.manual_seed(seed); m = copy.deepcopy(base).to(DEVICE); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=CFG["full_lr"]); bce = nn.BCEWithLogitsLoss()
    Xt, yt, Xv, yv = to_t(Xtr), to_t(ytr), to_t(Xva), to_t(yva); n, B = len(Xtr), CFG["full_batch"]
    def val_loss():
        m.eval()
        with torch.no_grad(): v = float(bce(m(Xv), yv))
        m.train(); return v
    best, best_state, best_ep, bad = val_loss(), copy.deepcopy(m.state_dict()), 0, 0
    for ep in range(1, CFG["full_max_epochs"] + 1):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        v = val_loss()
        if v < best - 1e-5: best, best_state, best_ep, bad = v, copy.deepcopy(m.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= CFG["full_patience"]: break
    m.load_state_dict(best_state); m.eval()
    return m, dict(best_epoch=best_ep, val_logloss=float(best), n_train_rows=int(tr.sum()), n_val_rows=int(va.sum()),
                   n_params=int(sum(p.numel() for p in m.parameters())))

# ============================================================== metrics (as exp_v5, plus event counts)
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
    return dict(flag_rate=float(flag.mean()), sensitivity=float(tp / max(ev, 1)), ppv=float(tp / max(flag.sum(), 1)),
                tp=tp, fp=fp, events=ev, n=n, missed_per_1000=float((ev - tp) / n * 1000), flagged_per_1000=float(flag.mean() * 1000)), flag
def net_benefit(yy, pp, pt):
    flag = pp >= pt; n = len(yy); tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum()
    return float(tp / n - fp / n * pt / (1 - pt))
def boot_patient(fn, yy, pp, pid):
    r = np.random.RandomState(1); ups = np.unique(pid); gi = {p: np.where(pid == p)[0] for p in ups}; vals = []
    for _ in range(CFG["n_boot"]):
        i = np.concatenate([gi[p] for p in r.choice(ups, len(ups), True)]); vals.append(fn(yy[i], pp[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
def evaluate(p, p_full, g, p_orig_group, thr_orig):
    """p: predictions on the group's held-out rows; p_full: the deployed system's predictions on the whole test set."""
    te = np.where(g["f_te"])[0]; yy, pid = y[te], sid[te]
    thr10 = float(np.percentile(p_full, 90))
    m10, flag10 = threshold_metrics(yy, p, thr10); mfix, _ = threshold_metrics(yy, p, CFG["fixed_cut"])
    slope, intercept = cal_slope_intercept(yy, p); alo, ahi = boot_patient(auc, yy, p, pid)
    out = dict(auroc=auc(yy, p), auroc_lo=alo, auroc_hi=ahi, ece=ece(yy, p), brier=float(np.mean((p - yy) ** 2)),
               cal_slope=slope, cal_intercept=intercept, mean_pred=float(p.mean()), observed=float(yy.mean()),
               full_test_auroc=auc(y[is_test], p_full), thr_top_decile=thr10, top_decile=m10, fixed_cut=mfix,
               net_benefit={str(pt): net_benefit(yy, p, pt) for pt in CFG["nb_thresholds"]})
    if p_orig_group is not None:
        _, flag_o = threshold_metrics(yy, p_orig_group, thr_orig)
        out["reclassified_top_decile"] = float((flag10 != flag_o).mean()); out["mean_abs_change"] = float(np.abs(p - p_orig_group).mean())
    return out
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

# ============================================================== run
originals = {s: load_v5(f"orig_s{s}") for s in CFG["seeds"]}
log("original models ready")
for gi_, g in enumerate(groups):
    key = f"{g['family']}_{g['name']}"
    te = np.where(g["f_te"])[0]; pos = np.searchsorted(test_idx, te)          # positions of the group's held-out rows inside the test set
    tr_rows = np.where(g["f_tr"])[0]; tr_pid = sid[tr_rows]
    nong_te = test_idx[~np.isin(test_idx, te)]                                 # non-group held-out rows: never seen by the base
    for s in CFG["seeds"]:
        path = f"{RUNS}/{g['family']}__{g['name']}__s{s}.json"
        if os.path.exists(path):
            try: json.load(open(path)); log(f"skip {g['family']}/{g['name']} seed {s} (done)"); continue
            except Exception: pass
        tg = time.time()
        orig, orig_sha = originals[s]; base, base_sha = load_v5(f"exact_{key}_s{s}")
        # shared-model predictions
        p_orig_full = sigmoid(logits(orig, X[is_test])); p_orig_g = p_orig_full[pos]; thr_orig = float(np.percentile(p_orig_full, 90))
        z_base_full = logits(base, X[is_test]); p_base_full = sigmoid(z_base_full); z_base_g = z_base_full[pos]
        z_base_tr = logits(base, X[tr_rows]); h_base_tr = penult(base, X[tr_rows]); h_base_g = penult(base, X[te])
        arms, meta = {}, {}
        arms["original"] = (p_orig_g, p_orig_full); meta["original"] = dict(frac=None, n_patients=None)
        arms["base"] = (sigmoid(z_base_g), p_base_full); meta["base"] = dict(frac=None, n_patients=None)
        # system-wide recalibration on non-group held-out patients (no group data)
        pl = fit_platt(z_base_full[~np.isin(test_idx, te)], y[nong_te])
        p_full = apply_platt(pl, z_base_full); arms["recal_retained"] = (p_full[pos], p_full)
        meta["recal_retained"] = dict(frac=None, n_patients=int(len(np.unique(sid[nong_te]))), n_rows=int(len(nong_te)), platt=pl, n_params=2)
        # control: a residual adapter trained on non-group held-out patients, patient-matched to the group, applied to the group
        r = np.random.RandomState(500 + gi_ + 31 * s); ng_p = np.unique(sid[nong_te]); r.shuffle(ng_p)
        ctrl_rows = nong_te[np.isin(sid[nong_te], ng_p[: min(g["n_tr_p"], len(ng_p))])]
        Zc = np.column_stack([z_base_full[np.searchsorted(test_idx, ctrl_rows)], X[ctrl_rows]])
        ad, info = fit_logistic_adapter(Zc, y[ctrl_rows], sid[ctrl_rows], s, offset=True)
        p_g = ad.predict_proba(np.column_stack([z_base_g, X[te]]))[:, 1]; p_full = p_base_full.copy(); p_full[pos] = p_g
        arms["resid_random"] = (p_g, p_full)
        meta["resid_random"] = dict(frac=1.0, n_patients=int(len(np.unique(sid[ctrl_rows]))), n_rows=int(len(ctrl_rows)), **info)
        # group-held components at each fraction of the group's training patients (nested subsets)
        r = np.random.RandomState(1000 + gi_ + 31 * s); ups = np.unique(tr_pid); r.shuffle(ups)
        for f in CFG["fractions"]:
            keep = set(ups[: max(2, int(round(f * len(ups))))]); sub = np.isin(tr_pid, list(keep))
            rows, zr, hr, yr, pr = tr_rows[sub], z_base_tr[sub], h_base_tr[sub], y[tr_rows][sub], tr_pid[sub]
            tag = f"@{f:g}"; n_p, n_r = int(len(keep)), int(sub.sum())
            pl = fit_platt(zr, yr); p_g = apply_platt(pl, z_base_g); p_full = p_base_full.copy(); p_full[pos] = p_g
            arms["platt_group" + tag] = (p_g, p_full); meta["platt_group" + tag] = dict(frac=f, n_patients=n_p, n_rows=n_r, platt=pl, n_params=2)
            ad, info = fit_logistic_adapter(np.column_stack([zr, X[rows]]), yr, pr, s, offset=True)
            p_g = ad.predict_proba(np.column_stack([z_base_g, X[te]]))[:, 1]; p_full = p_base_full.copy(); p_full[pos] = p_g
            arms["resid_group" + tag] = (p_g, p_full); meta["resid_group" + tag] = dict(frac=f, n_patients=n_p, n_rows=n_r, **info)
            ad, info = fit_logistic_adapter(hr, yr, pr, s)
            p_g = ad.predict_proba(h_base_g)[:, 1]; p_full = p_base_full.copy(); p_full[pos] = p_g
            arms["rep_group" + tag] = (p_g, p_full); meta["rep_group" + tag] = dict(frac=f, n_patients=n_p, n_rows=n_r, **info)
            fm, info = finetune_copy(base, rows, pr, s)
            p_g = sigmoid(logits(fm, X[te])); p_full = p_base_full.copy(); p_full[pos] = p_g
            arms["full_group" + tag] = (p_g, p_full); meta["full_group" + tag] = dict(frac=f, n_patients=n_p, n_rows=n_r, **info)
            del fm
        res = {}
        for name, (p_g, p_full) in arms.items():
            res[name] = evaluate(p_g, p_full, g, None if name == "original" else p_orig_g, thr_orig); res[name].update(meta[name])
        rec = dict(family=g["family"], group=g["name"], seed=s, n_train_patients=g["n_tr_p"], n_train_rows=int(g["f_tr"].sum()),
                   n_test_patients=g["n_te_p"], n_test_rows=int(g["f_te"].sum()), prevalence_group_train=float(y[g["f_tr"]].mean()),
                   base_checkpoint=f"exact_{key}_s{s}.pt", base_sha256=base_sha, original_sha256=orig_sha, arms=res, cfg=CFG, device=DEVICE,
                   torch=torch.__version__, python=platform.python_version(), date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg)
        save_json(path, rec)
        a0, ab = res["original"]["auroc"], res["base"]["auroc"]; fmax = f"@{max(CFG['fractions']):g}"
        log(f"{g['family']}/{g['name']} seed {s}: original {a0:.4f} base {ab:.4f} ({ab-a0:+.4f}) | vs original: resid {res['resid_group'+fmax]['auroc']-a0:+.4f} "
            f"rep {res['rep_group'+fmax]['auroc']-a0:+.4f} full {res['full_group'+fmax]['auroc']-a0:+.4f} platt {res['platt_group'+fmax]['auroc']-a0:+.4f} "
            f"recal {res['recal_retained']['auroc']-a0:+.4f} ctrl {res['resid_random']['auroc']-a0:+.4f} ({time.time()-tg:.0f}s)")

# ============================================================== summary from disk
def summary():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/*.json"))]
    if not recs: print("no runs yet"); return None
    rows = []
    for r in recs:
        a0, e0, ab = r["arms"]["original"]["auroc"], r["arms"]["original"]["ece"], r["arms"]["base"]["auroc"]
        for name, m in r["arms"].items():
            arm = name.split("@")[0]
            rows.append(dict(family=r["family"], group=r["group"], seed=r["seed"], arm=arm, frac=m.get("frac"), name=name,
                             n_train_patients=r["n_train_patients"], n_test_patients=r["n_test_patients"], n_patients_used=m.get("n_patients"),
                             auroc=m["auroc"], d_auroc_vs_original=m["auroc"] - a0, d_auroc_vs_base=m["auroc"] - ab, ece=m["ece"], d_ece_vs_original=m["ece"] - e0,
                             brier=m["brier"], cal_slope=m["cal_slope"], cal_intercept=m["cal_intercept"], full_test_auroc=m["full_test_auroc"],
                             sens_top_decile=m["top_decile"]["sensitivity"], ppv_top_decile=m["top_decile"]["ppv"], flag_rate_top_decile=m["top_decile"]["flag_rate"],
                             missed_per_1000_top_decile=m["top_decile"]["missed_per_1000"], flagged_per_1000_top_decile=m["top_decile"]["flagged_per_1000"],
                             sens_fixed=m["fixed_cut"]["sensitivity"], ppv_fixed=m["fixed_cut"]["ppv"], missed_per_1000_fixed=m["fixed_cut"]["missed_per_1000"],
                             flagged_per_1000_fixed=m["fixed_cut"]["flagged_per_1000"], **{"nb_0.2": m["net_benefit"]["0.2"], "nb_0.3": m["net_benefit"]["0.3"]},
                             reclassified_top_decile=m.get("reclassified_top_decile", 0.0), adapter_C=m.get("C"), best_epoch=m.get("best_epoch"),
                             base_sha256=r["base_sha256"]))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results.csv", index=False)
    S = R.groupby(["family", "group", "name", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 250)
    # provenance: base and original must reproduce v5 within floating-point noise (same checkpoints, same held-out rows)
    drift = []
    for r in recs:
        p = f"{V5_RUNS}/{r['family']}__{r['group']}__s{r['seed']}.json"
        if os.path.exists(p):
            v = json.load(open(p))["models"]
            drift.append(max(abs(v["original"]["auroc"] - r["arms"]["original"]["auroc"]), abs(v["exact_group"]["auroc"] - r["arms"]["base"]["auroc"])))
    if drift: print(f"\nprovenance: max |AUROC difference| between this run's original/base and the v5 records = {max(drift):.2e} over {len(drift)} records")
    shas = R.groupby(["family", "group", "seed"]).base_sha256.nunique().max()
    print(f"base checkpoint identity: {int(shas)} distinct SHA-256 per (group, seed) across arms (must be 1)")
    out = {}
    def paired(d):
        d = np.asarray(d, float); d = d[~np.isnan(d)]
        if len(d) < 2: return dict(d=float(d.mean()) if len(d) else float("nan"), ci=[float("nan")] * 2, frac_worse=float("nan"), n=int(len(d)))
        r = np.random.RandomState(0); bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
        return dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_worse=float((d < 0).mean()), n=int(len(d)))
    for fam in S.family.unique():
        Sf = S[S.family == fam]; W = Sf.pivot_table(index="group", columns="name", values="auroc")
        print(f"\n== {fam}: AUROC on the group's own held-out patients, seed means, change vs original (pooled) and vs base (exact retrain)")
        fmax = f"@{max(CFG['fractions']):g}"; arms_g = ["platt_group", "resid_group", "rep_group", "full_group"]
        names = [n for n in ["base", "recal_retained", "resid_random"] + [a + fmax for a in arms_g] if n in W.columns]
        for n in [n for n in ["base", "recal_retained", "resid_random"] + [f"{a}@{f:g}" for a in arms_g for f in CFG["fractions"]] if n in W.columns]:
            po, pb = paired(W[n] - W["original"]), paired(W[n] - W["base"]); out[f"{fam}|{n}"] = dict(vs_original=po, vs_base=pb)
            if n in names:
                print(f"  {n:18s} n={po['n']:2d}  vs original {po['d']:+.4f} [{po['ci'][0]:+.4f},{po['ci'][1]:+.4f}] worse {100*po['frac_worse']:3.0f}%   "
                      f"vs base {pb['d']:+.4f} [{pb['ci'][0]:+.4f},{pb['ci'][1]:+.4f}] better {100*(1-pb['frac_worse']):3.0f}%")
        if W.shape[0] > 1:
            loss = float((W["original"] - W["base"]).mean())
            if abs(loss) > 1e-6:
                print(f"  mean exact-retraining loss {loss:+.4f}; share recovered (mean recovery / mean loss): " +
                      ", ".join(f"{n} {float((W[n] - W['base']).mean()) / loss:.2f}" for n in names if n.endswith(fmax)))
        if len(CFG["fractions"]) > 1:
            print("  data-size curve, change vs original by fraction of the group's training patients held (mean over groups):")
            for a in arms_g:
                cols = [f"{a}@{f:g}" for f in CFG["fractions"] if f"{a}@{f:g}" in W.columns]
                print(f"    {a:12s} " + "  ".join(f"{f:g}: {float((W[c] - W['original']).mean()):+.4f}" for f, c in zip(CFG["fractions"], cols)))
        E = Sf.pivot_table(index="group", columns="name", values="ece"); print("  ECE change vs original: " + ", ".join(f"{n} {paired(E[n]-E['original'])['d']:+.4f}" for n in names))
        D = Sf.pivot_table(index="group", columns="name", values="missed_per_1000_top_decile")
        print("  missed events per 1,000 group admissions at the system top-decile threshold, change vs original: " +
              ", ".join(f"{n} {paired(D[n]-D['original'])['d']:+.1f}" for n in names))
    with open(f"{OUT}/paired.json", "w") as f: json.dump(out, f, indent=1)
    return S
def export():
    z = f"{OUT}/v11_confinement_results.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(f"{RUNS}/*.json") + [f"{OUT}/results.csv", f"{OUT}/summary.csv", f"{OUT}/paired.json"]:
            if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT))
    log(f"export: {z} ({os.path.getsize(z)/1e3:.0f} kB)")
if __name__ == "__main__":
    summary(); export(); log(f"done, results in {OUT}")
