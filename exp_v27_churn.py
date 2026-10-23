"""
exp_v27_churn.py. What a withdrawal does to everyone else's predictions, from cached checkpoints. No training.

Every remedy in Table 1 that retrains the shared model changes every patient's predicted risk, not only the withdrawing
group's; the manuscript says "all change (new model)" without a number. This script measures it on the held-out patients
OUTSIDE the withdrawing group, relative to the pooled model: the mean, 99th percentile and maximum absolute change in
predicted risk; patients per 1,000 whose flag changes when each model flags its own top decile of the held-out set (a
fixed-capacity programme), and how many of those were dropped; and patients per 1,000 whose flag changes at the pooled
model's threshold (a fixed risk cut). Arms: exact retraining without the group (also the one retraining confinement needs
at adoption); exact retraining without the matched random set (the churn retraining causes on its own); the seed floor
(the pooled model trained on the same data with another seed); the common base trained without five groups (v14.1,
MIMIC-IV); the federation retrained without the consortium and without a random set (v12.2, eICU). Under contribution
confinement a withdrawal deletes the group's component and leaves the base untouched, so other patients' predictions
change by zero by construction; nothing is computed for it.

Resampling follows v13. MIMIC-IV draws groups with replacement and, once per draw, held-out patients with replacement (a
patient's admissions move together); each drawn group is evaluated on the drawn patients outside it. eICU draws held-out
hospitals with replacement; each consortium is evaluated on the drawn stays outside it. Thresholds are set once on the
full held-out set and held fixed. The 99th percentile and maximum are point estimates (mean over groups and seeds of each
run's value).

Env: ROOT (default ~/mimic), PART=mimic|eicu|both (default both), NBOOT (default 1000; QUICK=1 gives 20, two seeds and a
few groups), DEVICE. Reads the v5, v14.1, model_cache_eicu_es, v7.2 and v12.2 checkpoints; missing optional arms are
skipped with a note. Writes ROOT/results/v27_churn[_quick]/{rows_mimic.npz, rows_eicu.npz, per_group_*.csv, churn.json,
churn.csv, summary.txt, log.txt}. Per-row arrays are cached, so the bootstrap can be rerun without touching a model.
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic"))
PART = os.environ.get("PART", "both")
NBOOT = int(os.environ.get("NBOOT", 20 if QUICK else 1000))
OUT = f"{ROOT}/results/v27_churn" + ("_quick" if QUICK else ""); os.makedirs(OUT, exist_ok=True)
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
SEEDS = [0, 1] if QUICK else [0, 1, 2]
CFG = dict(seeds=SEEDS, n_boot=NBOOT, quick=QUICK, capacity=0.10, version="v27.0")
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; part {PART}; seeds {SEEDS}; n_boot {NBOOT}; out {OUT}")

# ---------- model (as v5 / v7.2), inference only
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def load(path, d, required=True):
    if not os.path.exists(path):
        if required: raise SystemExit(f"missing checkpoint {path}")
        return None
    m = MLP(d).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def probs(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(torch.tensor(Xa[i:i + 65536], device=DEVICE))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)

# ---------- churn of one model against a reference, per held-out row
METRICS = ["mean_abs", "reclass_per_1000", "dropped_per_1000", "reclass_fixed_per_1000"]
SCALE = dict(mean_abs=1.0, reclass_per_1000=1000.0, dropped_per_1000=1000.0, reclass_fixed_per_1000=1000.0)
def churn_rows(p_ref, p_new, thr_ref):
    """p_ref: the reference model's predictions on the whole held-out set; thr_ref: its top-decile threshold.
    Capacity flags use each model's own top decile of the held-out set; the fixed cut applies thr_ref to both."""
    thr_new = float(np.percentile(p_new, 100 * (1 - CFG["capacity"])))
    f_ref, f_new = p_ref >= thr_ref, p_new >= thr_new
    return dict(mean_abs=np.abs(p_new - p_ref), reclass_per_1000=(f_ref != f_new), dropped_per_1000=(f_ref & ~f_new),
                reclass_fixed_per_1000=(f_ref != (p_new >= thr_ref)))
def stack(dicts):
    """list over seeds of churn_rows dicts -> dict metric -> float32 array (n_seeds, n_held_out_rows)"""
    return {m: np.stack([d[m] for d in dicts]).astype(np.float32) for m in METRICS}
def floor_rows(p_ref, thr):
    """seed floor: for each seed, the mean churn of the other seeds' reference models against it"""
    out = []
    for s in SEEDS:
        others = [churn_rows(p_ref[s], p_ref[s2], thr[s]) for s2 in SEEDS if s2 != s]
        out.append({m: np.mean([o[m] for o in others], axis=0) for m in METRICS})
    return stack(out)
def level(A, rows): return {m: float(A[m][:, rows].mean(dtype=np.float64) * SCALE[m]) for m in METRICS}
def extremes(A, rows):
    a = A["mean_abs"][:, rows]; return float(np.percentile(a, 99, axis=1).mean()), float(a.max(axis=1).mean())
def pct(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
results = {}
def record(key, point, boots, extra=None):
    results[key] = dict(d=float(point), ci=pct(boots), n_boot=int(len(boots)), **(extra or {}))
    log(f"  {key:78s} {point:+9.4f} [{results[key]['ci'][0]:+9.4f}, {results[key]['ci'][1]:+9.4f}]")
def point_only(key, value, extra=None): results[key] = dict(d=float(value), ci=[float("nan")] * 2, n_boot=0, **(extra or {}))
def save():
    json.dump(dict(cfg=CFG, results=results, date=time.strftime("%Y-%m-%d %H:%M:%S"), device=DEVICE, torch=torch.__version__),
              open(f"{OUT}/churn.json", "w"), indent=1)
    pd.DataFrame([dict(key=k, d=v["d"], lo=v["ci"][0], hi=v["ci"][1], n_boot=v["n_boot"]) for k, v in results.items()]).to_csv(f"{OUT}/churn.csv", index=False)

# ===================================================================================================== MIMIC-IV
def run_mimic():
    V5, V14 = f"{ROOT}/results/v5_partitions", f"{ROOT}/results/v14_commonbase_1"
    log("MIMIC: loading cohort")
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location",
            "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]; X = df[feat].astype(float)
    for c in [c for c in feat if c.startswith("lab_")]:
        X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
    X = X.values.astype(np.float32); sid = df.subject_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    test_idx = np.where(is_test)[0]; Xte = X[test_idx]; sid_te = sid[test_idx]; d = X.shape[1]
    pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    names, in_te = [], {}
    for c in range(25):
        members = set(assign.index[assign == c]); mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
        ref = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
        if ref["n_train_patients"] != len(set(sid[f_tr])) or ref["n_test_patients"] != len(set(sid[f_te])): raise SystemExit(f"c{c:02d}: membership differs from v5")
        names.append(f"c{c:02d}"); in_te[f"c{c:02d}"] = mem[test_idx]
    log(f"MIMIC: 25 groups match the v5 records; {len(test_idx)} held-out admissions, {len(np.unique(sid_te))} patients")
    if QUICK: names = names[:3]
    cache = f"{OUT}/rows_mimic.npz"
    if os.path.exists(cache):
        R = np.load(cache, allow_pickle=True)["R"].item(); log("MIMIC: per-row churn loaded from cache")
    else:
        p_orig = {s: probs(load(f"{V5}/ckpt/orig_s{s}.pt", d), Xte) for s in SEEDS}; thr = {s: float(np.percentile(p_orig[s], 90)) for s in SEEDS}
        R = {"seed_floor": floor_rows(p_orig, thr)}
        for n in names:
            for arm, pre in [("exact_group", "exact"), ("exact_random", "exactrp")]:
                R[(n, arm)] = stack([churn_rows(p_orig[s], probs(load(f"{V5}/ckpt/{pre}_community_trainfit_{n}_s{s}.pt", d), Xte), thr[s]) for s in SEEDS])
            lg, lr = level(R[(n, "exact_group")], ~in_te[n]), level(R[(n, "exact_random")], ~in_te[n])
            log(f"MIMIC {n}: others' mean |dp| exact {lg['mean_abs']:.4f} random {lr['mean_abs']:.4f}; reclassified per 1,000 exact {lg['reclass_per_1000']:.1f} random {lr['reclass_per_1000']:.1f}")
        sets = json.load(open(f"{V14}/sets_B.json")) if os.path.exists(f"{V14}/sets_B.json") else {}
        for sname, members in sets.items():
            ms = [load(f"{V14}/ckpt/base_{sname}_s{s}.pt", d, required=False) for s in SEEDS]
            if any(m is None for m in ms): log(f"MIMIC: v14.1 common base for {sname} missing, skipped"); continue
            R[("five", sname)] = stack([churn_rows(p_orig[s], probs(m, Xte), thr[s]) for s, m in zip(SEEDS, ms)])
            R[("five_mask", sname)] = ~np.any([in_te[n] for n in members], axis=0)
        np.savez_compressed(cache, R=np.array(R, dtype=object)); log(f"MIMIC: per-row churn cached at {cache}")
    masks = {n: ~in_te[n] for n in names}; arms = ["exact_group", "exact_random", "seed_floor"]
    def arr(n, arm): return R["seed_floor"] if arm == "seed_floor" else R[(n, arm)]
    fives = [k[1] for k in R if isinstance(k, tuple) and k[0] == "five"]
    # ---- per-group table (seed means)
    tab = []
    for n in names:
        for arm in arms:
            p99, mx = extremes(arr(n, arm), masks[n]); tab.append(dict(group=n, arm=arm, n_rows_outside=int(masks[n].sum()), p99_abs=p99, max_abs=mx, **level(arr(n, arm), masks[n])))
    pd.DataFrame(tab).to_csv(f"{OUT}/per_group_mimic.csv", index=False)
    # ---- point estimates and the group-then-patient bootstrap
    def stats(drawn, rows):
        out = {}
        for arm in arms:
            for m in METRICS: out[(arm, m)] = float(np.mean([arr(n, arm)[m][:, rows[masks[n][rows]]].mean(dtype=np.float64) * SCALE[m] for n in drawn]))
        for m in METRICS:
            out[("exact_group minus exact_random", m)] = out[("exact_group", m)] - out[("exact_random", m)]
            out[("exact_group minus seed_floor", m)] = out[("exact_group", m)] - out[("seed_floor", m)]
            out[("exact_random minus seed_floor", m)] = out[("exact_random", m)] - out[("seed_floor", m)]
        for sname in fives:
            sel = rows[R[("five_mask", sname)][rows]]
            for m in METRICS: out[(f"five groups ({sname}) common_base", m)] = float(R[("five", sname)][m][:, sel].mean(dtype=np.float64) * SCALE[m])
        return out
    rows_all = np.arange(len(test_idx)); point = stats(names, rows_all)
    order = np.argsort(sid_te, kind="stable"); ups, starts, counts = np.unique(sid_te[order], return_index=True, return_counts=True)
    def draw_rows(r):
        j = r.randint(0, len(ups), len(ups)); st, ct = starts[j], counts[j]; cum = np.cumsum(ct)
        return order[np.repeat(st - (cum - ct), ct) + np.arange(cum[-1])]
    boots = {k: [] for k in point}; r = np.random.RandomState(0)
    log(f"MIMIC: bootstrap {NBOOT} draws (groups, then held-out patients)")
    for b in range(NBOOT):
        for k, v in stats(list(r.choice(names, len(names), True)), draw_rows(r)).items(): boots[k].append(v)
        if (b + 1) % 100 == 0: log(f"  draw {b + 1}")
    for (arm, m), v in point.items(): record(f"MIMIC | {arm} | {m}", v, boots[(arm, m)], dict(n_groups=len(names), unit="group, then patient"))
    for arm in arms:
        p99, mx = np.mean([extremes(arr(n, arm), masks[n]) for n in names], axis=0)
        point_only(f"MIMIC | {arm} | p99_abs", p99); point_only(f"MIMIC | {arm} | max_abs", mx)
    for sname in fives:
        p99, mx = extremes(R[("five", sname)], R[("five_mask", sname)])
        point_only(f"MIMIC | five groups ({sname}) common_base | p99_abs", p99); point_only(f"MIMIC | five groups ({sname}) common_base | max_abs", mx)
    point_only("MIMIC | rows outside the group | min", min(int(masks[n].sum()) for n in names)); point_only("MIMIC | rows outside the group | max", max(int(masks[n].sum()) for n in names))
    point_only("MIMIC | n_groups", len(names)); save()

# ===================================================================================================== eICU
def run_eicu():
    V72, V122, ORIG = f"{ROOT}/results/v7_eicu_consortium_es", f"{ROOT}/results/v12_eicu_federated_es", f"{ROOT}/results/model_cache_eicu_es"
    log("eICU: loading cohort")
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
    META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
    feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
    for c in Xdf.columns[Xdf.isna().any()]:
        Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    X = Xdf.values.astype(np.float32); hosp = df.hospitalid.values
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    test_idx = np.where(is_test)[0]; Xte = X[test_idx]; hosp_te = hosp[test_idx]; d = X.shape[1]
    hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); keys, fam_of, cons_hosp = [], {}, {}
    for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
        for v in sorted(hattr[col].dropna().unique(), key=str):
            hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp, np.fromiter(hs, dtype=hosp.dtype))
            if (is_fit & mem).sum() < 800 or (is_test & mem).sum() < 100: continue
            key = f"{fam}_{str(v).replace(' ', '_').replace('/', '_')}"; keys.append(key); fam_of[key] = fam; cons_hosp[key] = hs
    log(f"eICU: {len(keys)} consortia; {len(test_idx)} held-out stays in {len(np.unique(hosp_te))} hospitals")
    if QUICK: keys = keys[:2]
    masks = {k: ~np.isin(hosp_te, np.fromiter(cons_hosp[k], dtype=hosp.dtype)) for k in keys}
    cache = f"{OUT}/rows_eicu.npz"
    if os.path.exists(cache):
        R = np.load(cache, allow_pickle=True)["R"].item(); log("eICU: per-row churn loaded from cache")
    else:
        p_orig = {s: probs(load(f"{ORIG}/orig_s{s}.pt", d), Xte) for s in SEEDS}; thr = {s: float(np.percentile(p_orig[s], 90)) for s in SEEDS}
        R = {"seed_floor": floor_rows(p_orig, thr)}
        fl = [load(f"{V122}/ckpt/fl_global_s{s}.pt", d, required=False) for s in SEEDS]
        if all(m is not None for m in fl):
            p_fl = {s: probs(m, Xte) for s, m in zip(SEEDS, fl)}; thr_fl = {s: float(np.percentile(p_fl[s], 90)) for s in SEEDS}; R["fl_seed_floor"] = floor_rows(p_fl, thr_fl)
        else: p_fl = None; log("eICU: v12.2 federation checkpoints missing, federated arms skipped")
        for k in keys:
            for arm, pre in [("exact_group", "exact"), ("exact_random", "exactr")]:
                R[(k, arm)] = stack([churn_rows(p_orig[s], probs(load(f"{V72}/ckpt/{pre}_{k}_s{s}.pt", d), Xte), thr[s]) for s in SEEDS])
            if p_fl is not None:
                for arm, pre in [("fl_without", "fl_without"), ("fl_random", "fl_random")]:
                    ms = [load(f"{V122}/ckpt/{pre}_{k}_s{s}.pt", d, required=False) for s in SEEDS]
                    if all(m is not None for m in ms): R[(k, arm)] = stack([churn_rows(p_fl[s], probs(m, Xte), thr_fl[s]) for s, m in zip(SEEDS, ms)])
                    else: log(f"eICU {k}: {pre} checkpoints missing, skipped")
            lg, lr = level(R[(k, "exact_group")], masks[k]), level(R[(k, "exact_random")], masks[k])
            log(f"eICU {k}: others' mean |dp| exact {lg['mean_abs']:.4f} random {lr['mean_abs']:.4f}; reclassified per 1,000 exact {lg['reclass_per_1000']:.1f} random {lr['reclass_per_1000']:.1f}")
        np.savez_compressed(cache, R=np.array(R, dtype=object)); log(f"eICU: per-row churn cached at {cache}")
    fl_ok = "fl_seed_floor" in R and all((k, "fl_without") in R and (k, "fl_random") in R for k in keys)
    arms = ["exact_group", "exact_random", "seed_floor"] + (["fl_without", "fl_random", "fl_seed_floor"] if fl_ok else [])
    pairs = [("exact_group", "exact_random"), ("exact_group", "seed_floor"), ("exact_random", "seed_floor")] + ([("fl_without", "fl_random"), ("fl_without", "fl_seed_floor")] if fl_ok else [])
    def arr(k, arm): return R[arm] if arm.endswith("seed_floor") else R[(k, arm)]
    tab = []
    for k in keys:
        for arm in arms:
            p99, mx = extremes(arr(k, arm), masks[k]); tab.append(dict(consortium=k, family=fam_of[k], arm=arm, n_rows_outside=int(masks[k].sum()), p99_abs=p99, max_abs=mx, **level(arr(k, arm), masks[k])))
    pd.DataFrame(tab).to_csv(f"{OUT}/per_group_eicu.csv", index=False)
    scopes = {"all": keys, **{fam: [k for k in keys if fam_of[k] == fam] for fam in ["region", "teaching", "bedsize"]}}
    def stats(rows, hs):
        out, val = {}, {}
        for k in keys:
            sel = rows[~np.isin(hs, np.fromiter(cons_hosp[k], dtype=hosp.dtype))]
            for arm in arms:
                for m in METRICS: val[(k, arm, m)] = float(arr(k, arm)[m][:, sel].mean(dtype=np.float64) * SCALE[m])
        for scope, ks in scopes.items():
            if not ks: continue
            for arm in arms:
                for m in METRICS: out[(scope, arm, m)] = float(np.mean([val[(k, arm, m)] for k in ks]))
            for a, b in pairs:
                for m in METRICS: out[(scope, f"{a} minus {b}", m)] = out[(scope, a, m)] - out[(scope, b, m)]
        return out
    point = stats(np.arange(len(test_idx)), hosp_te)
    hospitals = np.unique(hosp_te); h_rows = {h: np.where(hosp_te == h)[0] for h in hospitals}
    boots = {k: [] for k in point}; r = np.random.RandomState(0)
    log(f"eICU: bootstrap {NBOOT} draws over {len(hospitals)} held-out hospitals")
    for b in range(NBOOT):
        drawn = r.choice(hospitals, len(hospitals), True); rows = np.concatenate([h_rows[h] for h in drawn]); hs = np.concatenate([np.full(len(h_rows[h]), h) for h in drawn])
        for k, v in stats(rows, hs).items(): boots[k].append(v)
        if (b + 1) % 100 == 0: log(f"  draw {b + 1}")
    for (scope, arm, m), v in point.items(): record(f"eICU | {scope} | {arm} | {m}", v, boots[(scope, arm, m)], dict(unit="hospital cluster"))
    for arm in arms:
        p99, mx = np.mean([extremes(arr(k, arm), masks[k]) for k in keys], axis=0); point_only(f"eICU | all | {arm} | p99_abs", p99); point_only(f"eICU | all | {arm} | max_abs", mx)
    point_only("eICU | rows outside the consortium | min", min(int(masks[k].sum()) for k in keys)); point_only("eICU | rows outside the consortium | max", max(int(masks[k].sum()) for k in keys))
    point_only("eICU | n_consortia", len(keys)); save()

# ===================================================================================================== summary
def fmt(key, digits, width=26):
    v = results.get(key)
    if v is None: return "n/a".rjust(width)
    s = f"{v['d']:.{digits}f}"
    return (s + (f" [{v['ci'][0]:.{digits}f}, {v['ci'][1]:.{digits}f}]" if v["n_boot"] else "")).rjust(width)
def block(title, prefix, arms, pairs):
    lines = [title, f"{'arm':44s}{'mean |dp|':>26s}{'p99 |dp|':>10s}{'max |dp|':>10s}{'reclassified/1000':>26s}{'dropped/1000':>26s}{'fixed cut, reclassified/1000':>30s}"]
    def row(arm, p99="", mx=""):
        return (f"{arm:44s}{fmt(f'{prefix}{arm} | mean_abs', 4)}{p99:>10s}{mx:>10s}{fmt(f'{prefix}{arm} | reclass_per_1000', 1)}"
                f"{fmt(f'{prefix}{arm} | dropped_per_1000', 1)}{fmt(f'{prefix}{arm} | reclass_fixed_per_1000', 1, 30)}")
    for arm in arms:
        lines.append(row(arm, f"{results.get(f'{prefix}{arm} | p99_abs', {}).get('d', float('nan')):.4f}", f"{results.get(f'{prefix}{arm} | max_abs', {}).get('d', float('nan')):.4f}"))
    lines.append(f"{'confinement, deletion of the component':44s}{'0 by construction':>26s}{'':>20s}{'0':>26s}{'0':>26s}{'0':>30s}")
    for a, b in pairs: lines.append(row(f"{a} minus {b}"))
    return lines
def summary():
    lines = [f"v27 churn: other patients' predictions after a withdrawal ({time.strftime('%Y-%m-%d %H:%M')}; seeds {SEEDS}; {NBOOT} bootstrap draws; capacity = top decile of the held-out set)"]
    if any(k.startswith("MIMIC") for k in results):
        fives = sorted({k.split(" | ")[1] for k in results if k.startswith("MIMIC | five groups")})
        lines += block(f"\nMIMIC-IV, held-out admissions outside the withdrawing group ({results['MIMIC | rows outside the group | min']['d']:.0f} to {results['MIMIC | rows outside the group | max']['d']:.0f} rows), {results['MIMIC | n_groups']['d']:.0f} groups", "MIMIC | ",
                       ["exact_group", "exact_random", "seed_floor"] + fives, [("exact_group", "exact_random"), ("exact_group", "seed_floor"), ("exact_random", "seed_floor")])
    if any(k.startswith("eICU") for k in results):
        fl = "eICU | all | fl_without | mean_abs" in results
        lines += block(f"\neICU, held-out stays outside the withdrawing consortium ({results['eICU | rows outside the consortium | min']['d']:.0f} to {results['eICU | rows outside the consortium | max']['d']:.0f} rows), {results['eICU | n_consortia']['d']:.0f} consortia", "eICU | all | ",
                       ["exact_group", "exact_random", "seed_floor"] + (["fl_without", "fl_random", "fl_seed_floor"] if fl else []),
                       [("exact_group", "exact_random"), ("exact_group", "seed_floor"), ("exact_random", "seed_floor")] + ([("fl_without", "fl_random"), ("fl_without", "fl_seed_floor")] if fl else []))
        for fam in ["region", "teaching", "bedsize"]:
            if f"eICU | {fam} | exact_group | mean_abs" in results and not np.isnan(results[f"eICU | {fam} | exact_group | mean_abs"]["d"]):
                lines.append(f"  {fam:12s} exact_group mean |dp| {fmt(f'eICU | {fam} | exact_group | mean_abs', 4).strip()}; reclassified/1000 {fmt(f'eICU | {fam} | exact_group | reclass_per_1000', 1).strip()}; "
                             f"exact minus random, reclassified/1000 {fmt(f'eICU | {fam} | exact_group minus exact_random | reclass_per_1000', 1).strip()}")
    lines.append("\n|dp| is the absolute change in predicted risk against the pooled model (federated arms: against the full federation). Intervals: 95% bootstrap.")
    lines.append("exact_group also gives the one-off cost of adopting confinement for a model already in use (B2 then B3).")
    text = "\n".join(lines); print("\n" + text); open(f"{OUT}/summary.txt", "w").write(text + "\n")

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()
summary(); log(f"done, {len(results)} entries in {OUT}/churn.json")
