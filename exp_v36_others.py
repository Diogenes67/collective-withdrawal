"""
exp_v36_others.py. Other patients' discrimination under each single-group base (review item M7): AUROC on the held-out
patients outside the withdrawing group under the retained-only base, against the pooled model, with intervals, on both
datasets; and the same for the standard-pipeline exact retrain and, on eICU, the federation without the consortium.
No models are loaded. Replaces the "other patients' AUROC unchanged" claim of Table 3 A1 with measured ranges.

Sources. MIMIC-IV: results/v13_intervals_1/preds_mimic.npz (full-test predictions of the pooled model and the standard-pipeline
exact retrain per group and seed: arms full_original, full_base) and results/v16_strict/preds_others (strict base and pooled
model on the other held-out patients, written by exp_v16_strict.py with OTHERS=1); when preds_others is absent the strict-base
row uses the per-run others_auroc values of the v16 run records with a bootstrap over groups only. eICU: results/v13_intervals_1/
preds_eicu.npz (whole-test predictions: original, exact_group, fl_global, fl_without) and results/v23_eicu_strict/preds_others
(exp_v23_eicu_strict.py with OTHERS=1), on all stays and at the 24-hour landmark.
Bootstrap. MIMIC-IV: groups, then other patients within each drawn group (patients as units); eICU: hospitals, one draw feeding
all consortia. NBOOT draws, RandomState(0), draws saved, exact percentile p-values. Per-group values are seed means; the
per-group intervals resample patients (MIMIC-IV) or hospitals (eICU) within the group's complement.

Env: ROOT, NBOOT (default 1000; QUICK=1 gives 50), PART=mimic|eicu|both. Writes results/v36_others/{per_group.csv, intervals.json,
draws_*.npz, summary.txt, log.txt}.
"""
import os, json, time, glob
import numpy as np
import pandas as pd

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; V13 = f"{R}/v13_intervals_1"; PART = os.environ.get("PART", "both")
OUT = f"{R}/v36_others" + ("_quick" if QUICK else ""); os.makedirs(OUT, exist_ok=True)
NBOOT = int(os.environ.get("NBOOT", 50 if QUICK else 1000)); SEEDS = [0, 1, 2]
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"part {PART}; n_boot {NBOOT}; out {OUT}")
def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    o = np.argsort(p, kind="mergesort"); ps = p[o]; _, idx, cnt = np.unique(ps, return_index=True, return_counts=True)
    r = np.empty(len(p)); r[o] = np.repeat(idx + (cnt + 1) / 2.0, cnt); return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
def summarise(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; nb = len(v)
    return dict(ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], n_boot=int(nb), p_exact=float(min(1.0, max(2.0 / nb, 2 * min((v <= 0).mean(), (v >= 0).mean())))) if nb else float("nan"))
per_rows, results, draws = [], {}, {}

# ============================================================================== MIMIC-IV
def run_mimic():
    P13 = np.load(f"{V13}/preds_mimic.npz", allow_pickle=True)["P"].item(); memb = json.load(open(f"{V13}/membership_mimic.json"))
    names = sorted(n for n in memb if n.startswith("c"))
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet", columns=["subject_id", "readmit30"])
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; test_idx = np.where(is_test)[0]; y_te = df.readmit30.values[is_test].astype(np.float64); sid_te = df.subject_id.values[is_test]
    assert len(P13[(names[0], 0, "full_original")]) == len(y_te), "v13 MIMIC cache does not match the held-out set"
    G = {}
    for n in names:
        pos = np.searchsorted(test_idx, np.asarray(memb[n])); assert np.array_equal(test_idx[pos], np.asarray(memb[n])), f"{n}: membership rows are not held-out rows"
        others = np.setdiff1d(np.arange(len(test_idx)), pos)   # held-out rows outside the group, in held-out order (as v16 OTHERS=1 writes them)
        fo = {s: f"{R}/v16_strict/preds_others/community_trainfit__{n}__s{s}.npz" for s in SEEDS}
        rec = {s: json.load(open(f"{R}/v16_strict/runs/community_trainfit__{n}__s{s}.json")) for s in SEEDS if os.path.exists(f"{R}/v16_strict/runs/community_trainfit__{n}__s{s}.json")}
        G[n] = dict(others=others, y=y_te[others], pid=sid_te[others], arms={}, strict_point={})
        for s in SEEDS:
            G[n]["arms"][(s, "pooled")] = P13[(n, s, "full_original")][others]; G[n]["arms"][(s, "exact_std")] = P13[(n, s, "full_base")][others]
            if os.path.exists(fo[s]):
                Z = np.load(fo[s]); assert np.array_equal(Z["pid"], G[n]["pid"]), f"{n} s{s}: preds_others rows differ"; G[n]["arms"][(s, "strict_base")] = Z["strict_base"]; G[n]["arms"][(s, "pooled_v16")] = Z["pooled"]
            if s in rec: G[n]["strict_point"][s] = rec[s]["others_auroc"]["strict_base"] - rec[s]["others_auroc"]["pooled"]
    names = list(G); have_strict = all((s, "strict_base") in G[n]["arms"] for n in names for s in SEEDS)
    log(f"MIMIC: {len(names)} groups; strict-base predictions on other patients {'present' if have_strict else 'absent (run exp_v16_strict.py with OTHERS=1 for patient-level intervals)'}")
    CON = [("exact retrain, standard pipeline, minus pooled", "exact_std", "pooled")] + ([("strict base minus pooled", "strict_base", "pooled_v16")] if have_strict else [])
    def group_vals(n, idx):
        return {c[0]: float(np.nanmean([auc(G[n]["y"][idx], G[n]["arms"][(s, c[1])][idx]) - auc(G[n]["y"][idx], G[n]["arms"][(s, c[2])][idx]) for s in SEEDS])) for c in CON}
    full = {n: group_vals(n, np.arange(len(G[n]["y"]))) for n in names}
    pat = {}
    for n in names:
        order = np.argsort(G[n]["pid"], kind="stable"); u, st, ct = np.unique(G[n]["pid"][order], return_index=True, return_counts=True); pat[n] = (order, st, ct)
    def draw_rows(n, r):
        order, st, ct = pat[n]; j = r.randint(0, len(st), len(st)); s_, c_ = st[j], ct[j]; cum = np.cumsum(c_); return order[np.repeat(s_ - (cum - c_), c_) + np.arange(cum[-1])]
    for c in CON:
        per_rows.extend(dict(dataset="MIMIC-IV", group=n, population="all", contrast=c[0], n_other_patients=int(len(pat[n][1])), value=full[n][c[0]]) for n in names)
    # per-group patient-level intervals (NBOOT // 4 draws each) and the mean over groups (two-stage)
    cache = f"{OUT}/draws_mimic.npz"
    if os.path.exists(cache): D = np.load(cache, allow_pickle=True)["D"].item(); log("MIMIC: draws loaded")
    else:
        D = {"mean": {c[0]: [] for c in CON}, "group": {(n, c[0]): [] for n in names for c in CON}}; r = np.random.RandomState(0)
        for n in names:
            for b in range(max(50, NBOOT // 4)):
                v = group_vals(n, draw_rows(n, r))
                for c in CON: D["group"][(n, c[0])].append(v[c[0]])
        log("MIMIC: per-group draws done")
        for b in range(NBOOT):
            drawn = r.choice(names, len(names), True); vals = [group_vals(n, draw_rows(n, r)) for n in drawn]
            for c in CON: D["mean"][c[0]].append(float(np.mean([v[c[0]] for v in vals])))
            if (b + 1) % 200 == 0: log(f"  MIMIC draw {b + 1}")
        np.savez_compressed(cache, D=np.array(D, dtype=object))
    for c in CON:
        vals = np.array([full[n][c[0]] for n in names]); key = f"MIMIC-IV | all | other patients' AUROC, {c[0]}"
        results[key] = dict(d=float(vals.mean()), min=float(vals.min()), max=float(vals.max()), n_groups=len(names), unit="group, then patient", **summarise(D["mean"][c[0]]))
        for n in names:
            s_ = summarise(D["group"][(n, c[0])]); results[f"MIMIC-IV | all | {n} | other patients' AUROC, {c[0]}"] = dict(d=full[n][c[0]], unit="patient", **s_)
    if not have_strict and all(G[n]["strict_point"] for n in names):
        vals = np.array([np.mean(list(G[n]["strict_point"].values())) for n in names]); r = np.random.RandomState(1); bs = [r.choice(vals, len(vals), True).mean() for _ in range(5000)]
        results["MIMIC-IV | all | other patients' AUROC, strict base minus pooled (run records; bootstrap over groups only)"] = dict(d=float(vals.mean()), min=float(vals.min()), max=float(vals.max()), n_groups=len(names), unit="group", **summarise(bs))
        per_rows.extend(dict(dataset="MIMIC-IV", group=n, population="all", contrast="strict base minus pooled (run records)", n_other_patients=int(len(pat[n][1])), value=float(np.mean(list(G[n]["strict_point"].values())))) for n in names)

# ============================================================================== eICU
def run_eicu():
    P13 = np.load(f"{V13}/preds_eicu.npz", allow_pickle=True)["P"].item(); mem = json.load(open(f"{V13}/membership_eicu.json")); keys = sorted(mem)
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["patientunitstayid", "uniquepid", "hospitalid", "died"])
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; y_te = df.died.values[is_test].astype(np.float64); h_te = df.hospitalid.values[is_test]; stay = df.patientunitstayid.values[is_test]
    pt = pd.read_csv(f"{ROOT}/eicu/patient.csv.gz", usecols=["patientunitstayid", "unitdischargeoffset"]).set_index("patientunitstayid").unitdischargeoffset; lm = (pt.reindex(stay).values >= 1440)
    assert len(P13[(keys[0], 0, "original")]) == len(y_te)
    G = {}
    for k in keys:
        others = np.where(~np.isin(h_te, mem[k]))[0]; arms = {}
        for s in SEEDS:
            for a in ["original", "exact_group", "fl_global", "fl_without"]: arms[(s, a)] = P13[(k, s, a)][others]
            fo = f"{R}/v23_eicu_strict/preds_others/{k}__s{s}.npz"
            if os.path.exists(fo):
                Z = np.load(fo); assert np.array_equal(Z["hosp"], h_te[others]), f"{k} s{s}: preds_others rows differ"; arms[(s, "strict_base")] = Z["strict_base"]; arms[(s, "pooled_v23")] = Z["pooled"]
        G[k] = dict(others=others, y=y_te[others], hosp=h_te[others], lm=lm[others], arms=arms)
    have_strict = all((s, "strict_base") in G[k]["arms"] for k in keys for s in SEEDS)
    log(f"eICU: {len(keys)} consortia; strict-base predictions on other patients {'present' if have_strict else 'absent (run exp_v23_eicu_strict.py with OTHERS=1)'}")
    CON = [("exact retrain, standard pipeline, minus pooled", "exact_group", "original"), ("federation without the consortium minus full federation", "fl_without", "fl_global")] + \
          ([("strict base minus pooled", "strict_base", "pooled_v23")] if have_strict else [])
    hospitals = np.unique(h_te)
    def vals(pop, w=None):
        out = {}
        for k in keys:
            g = G[k]; idx = np.arange(len(g["y"])) if w is None else np.concatenate([np.repeat(np.where(g["hosp"] == h)[0], n) for h, n in w.items() if (g["hosp"] == h).any()] or [np.zeros(0, int)])
            if pop == "landmark": idx = idx[g["lm"][idx]]
            if len(idx) < 50 or len(np.unique(g["y"][idx])) < 2: out[k] = None; continue
            out[k] = {c[0]: float(np.nanmean([auc(g["y"][idx], g["arms"][(s, c[1])][idx]) - auc(g["y"][idx], g["arms"][(s, c[2])][idx]) for s in SEEDS])) for c in CON}
        return out
    for pop in ["all", "landmark"]:
        full = vals(pop)
        for c in CON: per_rows.extend(dict(dataset="eICU", group=k, population=pop, contrast=c[0], n_other_stays=int(len(G[k]["others"]) if pop == "all" else int(G[k]["lm"].sum())), value=full[k][c[0]]) for k in keys if full[k])
        cache = f"{OUT}/draws_eicu_{pop}.npz"
        if os.path.exists(cache): D = np.load(cache, allow_pickle=True)["D"].item(); log(f"eICU {pop}: draws loaded")
        else:
            D = {"mean": {c[0]: [] for c in CON}, "group": {(k, c[0]): [] for k in keys for c in CON}}; r = np.random.RandomState(0)
            for b in range(NBOOT):
                drawn = r.choice(hospitals, len(hospitals), True); w = dict(zip(*np.unique(drawn, return_counts=True))); v = vals(pop, w)
                for c in CON:
                    D["mean"][c[0]].append(float(np.nanmean([v[k][c[0]] for k in keys if v[k]])))
                    for k in keys: D["group"][(k, c[0])].append(v[k][c[0]] if v[k] else np.nan)
                if (b + 1) % 200 == 0: log(f"  eICU {pop} draw {b + 1}")
            np.savez_compressed(cache, D=np.array(D, dtype=object))
        for c in CON:
            vs = np.array([full[k][c[0]] for k in keys if full[k]]); key = f"eICU | {pop} | other patients' AUROC, {c[0]}"
            results[key] = dict(d=float(vs.mean()), min=float(vs.min()), max=float(vs.max()), n_groups=int(len(vs)), unit="hospital cluster", **summarise(D["mean"][c[0]]))
            for k in keys:
                if full[k]: results[f"eICU | {pop} | {k} | other patients' AUROC, {c[0]}"] = dict(d=full[k][c[0]], unit="hospital cluster", **summarise(D["group"][(k, c[0])]))

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()
pd.DataFrame(per_rows).to_csv(f"{OUT}/per_group.csv", index=False)
json.dump(dict(results=results, n_boot=NBOOT, date=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{OUT}/intervals.json", "w"), indent=1)
L = [f"v36 other patients' AUROC ({time.strftime('%Y-%m-%d %H:%M')}; {NBOOT} draws)"]
for k, v in results.items():
    if " | c" in k or (" | region_" in k) or (" | teaching_" in k) or (" | bedsize_" in k): continue
    L.append(f"  {k:95s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p_exact']:.4f}  per-group range {v['min']:+.4f} to {v['max']:+.4f} (n={v['n_groups']})")
L.append("largest per-group changes:")
PR = pd.DataFrame(per_rows)
if len(PR):
    for (ds, pop, con), sub in PR.groupby(["dataset", "population", "contrast"]):
        w = sub.loc[sub.value.idxmin()]; L.append(f"  {ds} {pop:8s} {con:60s} most negative {w.group} {w.value:+.4f}")
open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); log(f"done, results in {OUT}")
