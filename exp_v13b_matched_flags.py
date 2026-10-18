"""
exp_v13b_matched_flags.py. Decision metrics at a matched flag rate, from the v13 prediction caches (no models touched).

A fixed risk cut or a system-wide capacity threshold mixes two things: how well a model ranks the group's patients and
how high it places them on the probability scale. A group-held adapter that is better calibrated for a lower-risk group
flags fewer of its patients at the same threshold, so "missed events per 1,000" can rise while discrimination and
calibration improve. This script holds the group's flag rate fixed at what the original model flagged at the system
top-decile threshold, flags the same number of the group's patients under every arm by that arm's own ranking, and
reports missed events and false alarms per 1,000 under the same resampling as v13 (hospital clusters on eICU;
communities then patients on MIMIC). It also reports sensitivity at the group's own top decile and net benefit at the
fixed clinical threshold.

Env: ROOT, PART, NBOOT (default 1000; QUICK=1 gives 20). Reads results/v13_intervals/preds_{mimic,eicu}.npz and the
cohorts (for labels and ids). Writes results/v13_intervals/matched_flags.{json,csv}.
"""
import os, json, time
import numpy as np
import pandas as pd

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); PART = os.environ.get("PART", "both")
NBOOT = int(os.environ.get("NBOOT", 20 if QUICK else 1000))
IN = f"{ROOT}/results/v13_intervals_1" + ("_quick" if QUICK else ""); OUT = IN
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
SEEDS = [0, 1, 2]; results = {}
def pct(v): v = np.asarray(v, float); v = v[~np.isnan(v)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
def record(key, point, boots):
    results[key] = dict(d=float(point), ci=pct(boots), n_boot=int(len(boots))); log(f"  {key:70s} {point:+.3f} [{results[key]['ci'][0]:+.3f}, {results[key]['ci'][1]:+.3f}]")
def flag_topk(p, k):
    """Flag the k highest-scoring rows (ties broken by order); returns boolean mask."""
    if k <= 0: return np.zeros(len(p), bool)
    idx = np.argpartition(-p, min(k, len(p)) - 1)[:k]; m = np.zeros(len(p), bool); m[idx] = True; return m
def per1000(yy, flag):
    ev = (yy == 1).sum(); tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum(); n = len(yy)
    return dict(missed=float((ev - tp) / n * 1000), false_alarms=float(fp / n * 1000), flagged=float(flag.mean() * 1000), sens=float(tp / max(ev, 1)))
def net_benefit(yy, pp, pt):
    flag = pp >= pt; n = len(yy); tp = (flag & (yy == 1)).sum(); fp = (flag & (yy == 0)).sum(); return float(tp / n - fp / n * pt / (1 - pt))

def run_mimic():
    log("MIMIC: loading cohort ids and labels")
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]; X = df[feat].astype(float)
    for c in [c for c in feat if c.startswith("lab_")]:
        X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
    X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    mp = f"{IN}/membership_mimic.json"
    if not os.path.exists(mp): raise SystemExit(f"{mp} missing: run exp_v13_intervals.py with MEMBERSHIP_ONLY=1 PART=mimic first")
    te_rows = {k: np.array(v) for k, v in json.load(open(mp)).items()}; names = sorted(te_rows)
    P = np.load(f"{IN}/preds_mimic.npz", allow_pickle=True)["P"].item(); log("MIMIC: predictions loaded")
    for n in names:
        if len(P[(n, 0, "original")]) != len(te_rows[n]): raise SystemExit(f"{n}: membership ({len(te_rows[n])}) does not match cached predictions ({len(P[(n, 0, 'original')])})")
    arms = ["base", "random", "resid", "rep", "resid_random"]; pt_fixed = 0.3
    thr_orig = {s: float(np.percentile(P[("c00", s, "full_original")], 90)) for s in SEEDS}
    pat_index = {n: (np.unique(sid[te_rows[n]]), {p: np.where(sid[te_rows[n]] == p)[0] for p in np.unique(sid[te_rows[n]])}) for n in names}
    def stats(n, idx):
        rows = te_rows[n][idx]; yy = y[rows]; out = {}
        for s in SEEDS:
            po = P[(n, s, "original")][idx]; k = int((po >= thr_orig[s]).sum()); k10 = max(1, int(round(0.1 * len(idx))))
            o = per1000(yy, po >= thr_orig[s]); out.setdefault("original", []).append(dict(o, sens10=per1000(yy, flag_topk(po, k10))["sens"], nb=net_benefit(yy, po, pt_fixed)))
            for a in arms:
                p = P[(n, s, a)][idx]; r = per1000(yy, flag_topk(p, k)); r["sens10"] = per1000(yy, flag_topk(p, k10))["sens"]; r["nb"] = net_benefit(yy, p, pt_fixed); out.setdefault(a, []).append(r)
        return {a: {m: np.mean([d[m] for d in v]) for m in v[0]} for a, v in out.items()}
    def contrasts(S):
        d = {}
        for a in ["base", "resid", "rep", "resid_random"]:
            for m, lab in [("missed", "missed events per 1,000 at matched flag rate"), ("false_alarms", "false alarms per 1,000 at matched flag rate"), ("sens10", "sensitivity at the group's own top decile"), ("nb", "net benefit at p=0.3")]:
                d[f"{lab} | {a} minus original"] = np.mean([S[n][a][m] - S[n]["original"][m] for n in S])
        d["missed events per 1,000 at matched flag rate | base minus random"] = np.mean([S[n]["base"]["missed"] - S[n]["random"]["missed"] for n in S])
        d["flagged per 1,000 by the original at the system top decile (level)"] = np.mean([S[n]["original"]["flagged"] for n in S])
        return d
    point = contrasts({n: stats(n, np.arange(len(te_rows[n]))) for n in names}); boots = {k: [] for k in point}; r = np.random.RandomState(0)
    log(f"MIMIC: bootstrap {NBOOT}")
    for b in range(NBOOT):
        S = {}
        for j, n in enumerate(r.choice(names, len(names), True)):
            ups, gi = pat_index[n]; idx = np.concatenate([gi[p] for p in r.choice(ups, len(ups), True)]); S[f"{n}#{j}"] = stats(n, idx)
        for k, v in contrasts(S).items(): boots[k].append(v)
    for k in point: record(f"MIMIC communities | {k}", point[k], boots[k])

def run_eicu():
    log("eICU: loading cohort ids and labels")
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet"); y = df.died.values.astype(np.float32); pid = df.uniquepid.values; hosp = df.hospitalid.values
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test; test_idx = np.where(is_test)[0]
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    mp = f"{IN}/membership_eicu.json"
    if not os.path.exists(mp): raise SystemExit(f"{mp} missing: run exp_v13_intervals.py with MEMBERSHIP_ONLY=1 PART=eicu first")
    cons = {k: set(v) for k, v in json.load(open(mp)).items()}
    P = np.load(f"{IN}/preds_eicu.npz", allow_pickle=True)["P"].item(); log(f"eICU: predictions loaded, {len(cons)} consortia")
    keys = list(cons); hosp_te = hosp[test_idx]; y_te = y[test_idx]; hospitals = np.unique(hosp_te); h_rows = {h: np.where(hosp_te == h)[0] for h in hospitals}
    pt_fixed = 0.1; pairs = [("exact_group", "original"), ("exact_random", "original"), ("fl_without", "fl_global"), ("fl_random", "fl_global"), ("fl_resid", "fl_global"), ("fl_resid_random", "fl_global"), ("fl_resid", "fl_without")]
    thr = {(s, ref): float(np.percentile(P[(keys[0], s, ref)], 90)) for s in SEEDS for ref in ["original", "fl_global"]}
    def stats(rows_all, hs_all):
        out = {}
        for k in keys:
            rows = rows_all[np.isin(hs_all, list(cons[k]))]
            if len(rows) < 50 or len(np.unique(y_te[rows])) < 2: out[k] = None; continue
            yy = y_te[rows]; o = {}
            for arm, ref in pairs:
                vals = []
                for s in SEEDS:
                    pr = P[(k, s, ref)][rows]; kflag = int((pr >= thr[(s, ref)]).sum()) if ref in ("original", "fl_global") else int((pr >= float(np.percentile(P[(k, s, ref)], 90))).sum())
                    pa = P[(k, s, arm)][rows]; ra, rr = per1000(yy, flag_topk(pa, kflag)), per1000(yy, flag_topk(pr, kflag))
                    k10 = max(1, int(round(0.1 * len(rows))))
                    vals.append(dict(missed=ra["missed"] - rr["missed"], false_alarms=ra["false_alarms"] - rr["false_alarms"],
                                     sens10=per1000(yy, flag_topk(pa, k10))["sens"] - per1000(yy, flag_topk(pr, k10))["sens"], nb=net_benefit(yy, pa, pt_fixed) - net_benefit(yy, pr, pt_fixed)))
                o[(arm, ref)] = {m: np.mean([v[m] for v in vals]) for m in vals[0]}
            out[k] = o
        return out
    def contrasts(S):
        d = {}
        for arm, ref in pairs:
            for m, lab in [("missed", "missed events per 1,000 at matched flag rate"), ("false_alarms", "false alarms per 1,000 at matched flag rate"), ("sens10", "sensitivity at the consortium's own top decile"), ("nb", "net benefit at p=0.1")]:
                v = [S[k][(arm, ref)][m] for k in keys if S[k] is not None]; d[f"{lab} | {arm} minus {ref}"] = np.mean(v) if v else np.nan
        return d
    all_rows = np.arange(len(test_idx)); point = contrasts(stats(all_rows, hosp_te)); boots = {k: [] for k in point}; r = np.random.RandomState(0)
    log(f"eICU: bootstrap {NBOOT} over {len(hospitals)} hospitals")
    for b in range(NBOOT):
        drawn = r.choice(hospitals, len(hospitals), True); rows = np.concatenate([h_rows[h] for h in drawn]); hs = np.concatenate([np.full(len(h_rows[h]), h) for h in drawn])
        for k, v in contrasts(stats(rows, hs)).items(): boots[k].append(v)
    for k in point: record(f"eICU | all | {k}", point[k], boots[k])

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()
json.dump(dict(results=results, n_boot=NBOOT, date=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{OUT}/matched_flags.json", "w"), indent=1)
pd.DataFrame([dict(key=k, d=v["d"], lo=v["ci"][0], hi=v["ci"][1]) for k, v in results.items()]).to_csv(f"{OUT}/matched_flags.csv", index=False)
log(f"done, {len(results)} intervals in {OUT}/matched_flags.json")
