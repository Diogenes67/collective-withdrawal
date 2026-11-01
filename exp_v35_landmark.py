"""
exp_v35_landmark.py. The 24-hour landmark cohort as the primary eICU analysis (review item M5): every eICU contrast of the
main text recomputed on held-out stays still in the unit at 24 hours, with per-consortium values, from cached predictions only
(no models, no torch). Supersedes the landmark rows of v18 and v26 as the source of the manuscript's eICU numbers; the
all-stays cohort is recomputed alongside as the sensitivity analysis and as a check against the published values.

Arms. Centralised and federated arms from the v13.1 cache (predictions on the whole held-out set): original, exact_group,
exact_random, fl_global, fl_without, fl_random, fl_resid, fl_resid_random. Strict-pipeline arms from the v25 and v23 caches
(predictions on the consortium's held-out stays): pooled, adapted pooled comparator (pooled_cf_chosen), confined system
(confined_strict_chosen), strict base, control component.
Landmark. Held-out stays with unitdischargeoffset >= 1440 minutes (as v18 and v23); models unchanged.
Matched flag rate. k = number of the consortium's stays in the population that the reference model (original for centralised
arms, full federation for federated arms, pooled for strict arms) flags at its system-wide top-decile threshold; every arm
flags its own top k; deaths unflagged per 1,000 stays. The reference flags are fixed once on the full held-out set and counted
within each bootstrap sample, as v18.
Bootstrap. Hospital-cluster bootstrap of the held-out set, one hospital draw feeding all ten consortia; NBOOT draws in NCHUNK
chunks with RandomState(1000 + chunk) (all stays) and RandomState(2000 + chunk) (landmark), every draw saved; 95% percentile
intervals, exact two-sided percentile p-values (2 x the smaller tail at zero, floored at 2/NBOOT), and for the two confinement
contrasts the two one-sided tests at a 0.005 AUROC margin from the 90% interval.

Env: ROOT (default ~/mimic), NBOOT (default 1000; QUICK=1 gives 50), NCHUNK (default 5), POP=landmark|all|both (default both).
Reads results/v13_intervals_1/{preds_eicu.npz, membership_eicu.json}, results/v25_symmetric_eicu/preds, results/v23_eicu_strict/preds,
derived/eicu_cohort.parquet, eicu/patient.csv.gz. Writes results/v35_landmark/{per_consortium.csv, contrasts_{pop}.json,
draws_{pop}_chunk{c}.npz, table2_eicu_{pop}.md, summary.txt, log.txt}.
"""
import os, json, time, glob
import numpy as np
import pandas as pd

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; V13 = f"{R}/v13_intervals_1"
OUT = f"{R}/v35_landmark" + ("_quick" if QUICK else ""); os.makedirs(OUT, exist_ok=True)
NBOOT = int(os.environ.get("NBOOT", 50 if QUICK else 1000)); NCHUNK = int(os.environ.get("NCHUNK", 5)); POPS = ["landmark", "all"] if os.environ.get("POP", "both") == "both" else [os.environ["POP"]]
SEEDS = [0, 1, 2]; MARGIN = 0.005
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"quick={QUICK}; n_boot {NBOOT} in {NCHUNK} chunks; populations {POPS}; out {OUT}")

def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    o = np.argsort(p, kind="mergesort"); ps = p[o]; _, idx, cnt = np.unique(ps, return_index=True, return_counts=True)
    r = np.empty(len(p)); r[o] = np.repeat(idx + (cnt + 1) / 2.0, cnt); return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
def unflagged(y, p, k):
    if k <= 0: return 1000.0 * y.sum() / len(y)
    top = np.argpartition(-p, min(k, len(p)) - 1)[:k]; return 1000.0 * (y.sum() - y[top].sum()) / len(y)

# ------------------------------------------------------------------ held-out set, landmark flag and consortium membership (as v18)
df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["patientunitstayid", "uniquepid", "hospitalid", "died"])
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values; y_all = df.died.values[is_test].astype(np.float64); h_all = df.hospitalid.values[is_test]; stay = df.patientunitstayid.values[is_test]
pt = pd.read_csv(f"{ROOT}/eicu/patient.csv.gz", usecols=["patientunitstayid", "unitdischargeoffset"]).set_index("patientunitstayid").unitdischargeoffset
lm_all = (pt.reindex(stay).values >= 1440)
P13 = np.load(f"{V13}/preds_eicu.npz", allow_pickle=True)["P"].item(); mem = json.load(open(f"{V13}/membership_eicu.json")); keys = sorted(mem)
assert len(P13[next(iter(P13))]) == len(y_all), "v13 cache does not match the held-out set"
CENT = ["original", "exact_group", "exact_random"]; FED = ["fl_global", "fl_without", "fl_random", "fl_resid", "fl_resid_random"]; STRICT = ["pooled", "sym", "confined", "base", "control"]
REF = {**{a: "original" for a in CENT}, **{a: "fl_global" for a in FED}, **{a: "pooled" for a in STRICT}}
ARMS = CENT + FED + STRICT
thr13 = {(k, s, a): float(np.percentile(P13[(k, s, a)], 90)) for k in keys for s in SEEDS for a in ["original", "fl_global"]}
C = {}   # per consortium: rows (indices into the held-out set), y, hosp, landmark, preds[(s, arm)], flags[(s, ref)]
for k in keys:
    rows = np.where(np.isin(h_all, mem[k]))[0]; c = dict(rows=rows, y=y_all[rows], hosp=h_all[rows], lm=lm_all[rows], pred={}, flag={})
    for s in SEEDS:
        for a in CENT + FED: c["pred"][(s, a)] = P13[(k, s, a)][rows]
        Z = np.load(f"{R}/v25_symmetric_eicu/preds/{k}__s{s}.npz"); Y = np.load(f"{R}/v23_eicu_strict/preds/{k}__s{s}.npz")
        assert np.array_equal(Z["y"], c["y"]) and np.array_equal(Y["y"], c["y"]) and np.array_equal(Z["hosp"], c["hosp"]), f"{k} s{s}: strict caches not aligned with the held-out rows"
        assert np.array_equal(Z["landmark"].astype(bool), c["lm"]), f"{k} s{s}: landmark flags differ between v25 and patient.csv"
        c["pred"][(s, "pooled")] = Z["pooled"]; c["pred"][(s, "sym")] = Z["pooled_cf_chosen"]; c["pred"][(s, "confined")] = Z["confined_strict_chosen"]
        c["pred"][(s, "base")] = Y["base"]; c["pred"][(s, "control")] = Y["control"]
        c["flag"][(s, "original")] = c["pred"][(s, "original")] >= thr13[(k, s, "original")]; c["flag"][(s, "fl_global")] = c["pred"][(s, "fl_global")] >= thr13[(k, s, "fl_global")]
        kf = int(Z["k_flag"]); fl = np.zeros(len(rows), bool); fl[np.argsort(-c["pred"][(s, "pooled")], kind="mergesort")[:kf]] = True; c["flag"][(s, "pooled")] = fl
    C[k] = c
hospitals = np.unique(h_all); h_rows = {h: np.where(h_all == h)[0] for h in hospitals}
log(f"{len(keys)} consortia; {len(hospitals)} held-out hospitals; {int(lm_all.sum())} of {len(lm_all)} held-out stays at the landmark (mortality {y_all[lm_all].mean():.4f}, {len(np.unique(h_all[lm_all]))} hospitals)")

AUC_PAIRS = {"centralised penalty: retrained without the consortium minus random removal": ("exact_group", "exact_random"),
             "retrained without the consortium minus pooled model": ("exact_group", "original"),
             "random removal minus pooled model": ("exact_random", "original"),
             "federated penalty: federation without the consortium minus federation after random removal": ("fl_without", "fl_random"),
             "federation without the consortium minus full federation": ("fl_without", "fl_global"),
             "full federation minus pooled model": ("fl_global", "original"),
             "consortium component in the federation minus full federation": ("fl_resid", "fl_global"),
             "consortium component minus federation without the consortium (recovered)": ("fl_resid", "fl_without"),
             "control component minus federation without the consortium": ("fl_resid_random", "fl_without"),
             "consortium-specific gain in the federation (component minus control)": ("fl_resid", "fl_resid_random"),
             "strict base minus pooled model (centralised)": ("base", "pooled"),
             "confined (strict) minus pooled model": ("confined", "pooled"),
             "confined (strict) minus adapted pooled comparator": ("confined", "sym"),
             "adapted pooled comparator minus pooled model": ("sym", "pooled"),
             "control component minus strict base": ("control", "base"),
             "deletion: confined minus strict base": ("confined", "base")}
MF_PAIRS = {"retrained without the consortium minus pooled model": ("exact_group", "original"), "random removal minus pooled model": ("exact_random", "original"),
            "retrained without the consortium minus random removal": ("exact_group", "exact_random"),
            "federation without the consortium minus full federation": ("fl_without", "fl_global"), "federation without the consortium minus federation after random removal": ("fl_without", "fl_random"),
            "consortium component minus full federation": ("fl_resid", "fl_global"), "consortium component minus control component": ("fl_resid", "fl_resid_random"),
            "confined (strict) minus pooled model": ("confined", "pooled"), "confined (strict) minus adapted pooled comparator": ("confined", "sym"), "strict base minus pooled model": ("base", "pooled")}
TOST = ["confined (strict) minus adapted pooled comparator", "confined (strict) minus pooled model"]

def stats(pop, w=None):
    """Per consortium: {arm: (AUROC seed mean, unflagged per 1,000 seed mean)} on the (weighted) hospital sample."""
    out = {}
    for k in keys:
        c = C[k]
        idx = np.arange(len(c["rows"])) if w is None else np.concatenate([np.repeat(np.where(c["hosp"] == h)[0], n) for h, n in w.items() if (c["hosp"] == h).any()] or [np.zeros(0, int)])
        if pop == "landmark": idx = idx[c["lm"][idx]]
        if len(idx) < 50 or len(np.unique(c["y"][idx])) < 2: out[k] = None; continue
        yy = c["y"][idx]; o = {}
        for a in ARMS:
            aucs, miss = [], []
            for s in SEEDS:
                p = c["pred"][(s, a)][idx]; aucs.append(auc(yy, p)); miss.append(unflagged(yy, p, int(c["flag"][(s, REF[a])][idx].sum())))
            o[a] = (float(np.nanmean(aucs)), float(np.mean(miss)))
        out[k] = o
    return out
def contrasts(st):
    d = {}
    for n, (a, b) in AUC_PAIRS.items(): d[f"AUROC | {n}"] = float(np.mean([o[a][0] - o[b][0] for o in st.values() if o is not None]))
    for n, (a, b) in MF_PAIRS.items(): d[f"deaths unflagged per 1,000 | {n}"] = float(np.mean([o[a][1] - o[b][1] for o in st.values() if o is not None]))
    return d

# ------------------------------------------------------------------ per-consortium values (seed means), both populations
rows = []
for pop in ["all", "landmark"]:
    st = stats(pop)
    for k in keys:
        o = st[k]; c = C[k]; idx = np.arange(len(c["rows"])); idx = idx[c["lm"][idx]] if pop == "landmark" else idx
        row = dict(consortium=k, family=k.split("_")[0], group=k.split("_", 1)[1], population=pop, held_out_stays=int(len(idx)), deaths=int(c["y"][idx].sum()), hospitals=int(len(np.unique(c["hosp"][idx]))))
        for a in ARMS: row[f"auroc_{a}"] = o[a][0]; row[f"unflagged_{a}"] = o[a][1]
        row["penalty"] = o["exact_group"][0] - o["exact_random"][0]; row["fed_penalty"] = o["fl_without"][0] - o["fl_random"][0]
        row["confined_minus_sym"] = o["confined"][0] - o["sym"][0]; row["confined_minus_pooled"] = o["confined"][0] - o["pooled"][0]; row["base_minus_pooled"] = o["base"][0] - o["pooled"][0]
        rows.append(row)
PC = pd.DataFrame(rows)
wide = PC.pivot_table(index=["consortium", "family", "group"], columns="population", values=["penalty", "fed_penalty", "confined_minus_sym", "confined_minus_pooled", "base_minus_pooled"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]; wide = wide.reset_index()
PC.to_csv(f"{OUT}/per_consortium_long.csv", index=False); wide.to_csv(f"{OUT}/per_consortium.csv", index=False)
log("per-consortium values written (per_consortium.csv: one row per consortium; per_consortium_long.csv: one row per consortium and population)")

# ------------------------------------------------------------------ bootstrap, per population, chunked and resumable
summary_lines = [f"v35 landmark ({time.strftime('%Y-%m-%d %H:%M')}); {NBOOT} hospital-cluster draws in {NCHUNK} chunks"]
for pop in POPS:
    pt_ = contrasts(stats(pop)); per = max(1, NBOOT // NCHUNK); B = {k: [] for k in pt_}
    for ch in range(NCHUNK):
        f = f"{OUT}/draws_{pop}_chunk{ch}.npz"
        if os.path.exists(f): Z = np.load(f, allow_pickle=True)["B"].item(); log(f"{pop}: chunk {ch} loaded")
        else:
            r = np.random.RandomState((2000 if pop == "landmark" else 1000) + ch); Z = {k: [] for k in pt_}
            for b in range(per):
                drawn = r.choice(hospitals, len(hospitals), True); w = dict(zip(*np.unique(drawn, return_counts=True)))
                for k, v in contrasts(stats(pop, w)).items(): Z[k].append(v)
            np.savez_compressed(f, B=np.array(Z, dtype=object)); log(f"{pop}: chunk {ch} done ({per} draws)")
        for k in pt_: B[k] += list(Z[k])
    st_full = stats(pop); n_ok = sum(1 for o in st_full.values() if o is not None)
    lower = {f"AUROC | {n}": int(sum(st_full[k][a][0] < st_full[k][b][0] for k in keys if st_full[k] is not None)) for n, (a, b) in AUC_PAIRS.items()}
    res = {}
    for k, d in pt_.items():
        v = np.asarray(B[k], float); v = v[~np.isnan(v)]; nb = len(v)
        p_exact = min(1.0, max(2.0 / nb, 2 * min((v <= 0).mean(), (v >= 0).mean()))) if nb else float("nan")
        res[k] = dict(d=d, ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], n_boot=int(nb), p_exact=float(p_exact), n_consortia=n_ok)
        if k in lower: res[k]["lower_in"] = lower[k]
        if k.startswith("AUROC | ") and k[8:] in TOST:
            lo90, hi90 = float(np.percentile(v, 5)), float(np.percentile(v, 95)); mmin = max(abs(lo90), abs(hi90))
            res[k]["tost"] = dict(margin=MARGIN, ci90=[lo90, hi90], smallest_margin=mmin, equivalent=bool(mmin < MARGIN), p_tost=float(max((v <= -MARGIN).mean(), (v >= MARGIN).mean())))
    info = dict(population=pop, held_out_stays=int(lm_all.sum()) if pop == "landmark" else int(len(lm_all)), of_total=int(len(lm_all)), mortality=float(y_all[lm_all].mean() if pop == "landmark" else y_all.mean()),
                hospitals=int(len(np.unique(h_all[lm_all]))) if pop == "landmark" else int(len(hospitals)), n_boot=NBOOT, seeds=SEEDS, chunk_rng="RandomState((2000 if landmark else 1000) + chunk)")
    json.dump(dict(population=info, results=res, date=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{OUT}/contrasts_{pop}.json", "w"), indent=1)
    summary_lines.append(f"== {pop}: {info['held_out_stays']} of {info['of_total']} held-out stays, mortality {info['mortality']:.4f}, {info['hospitals']} hospitals")
    for k, v in res.items():
        line = f"  {k:110s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p_exact']:.4f}" + (f"  lower in {v['lower_in']} of {n_ok}" if "lower_in" in v else "")
        if "tost" in v: line += f"  TOST 90% [{v['tost']['ci90'][0]:+.4f}, {v['tost']['ci90'][1]:+.4f}] smallest margin {v['tost']['smallest_margin']:.4f} p {v['tost']['p_tost']:.4f}"
        summary_lines.append(line)
    def f4(x): return ("+" if x >= 0 else "−") + f"{abs(x):.4f}"
    def f2(x): return ("+" if x >= 0 else "−") + f"{abs(x):.2f}"
    def cell(key, fmt):
        v = res[key]; return f"{fmt(v['d'])} ({fmt(v['ci'][0])} to {fmt(v['ci'][1])})"
    T = [f"Table 2, eICU rows, {pop} ({info['held_out_stays']} held-out stays).", "",
         "| Setting | State 1, pooled | State 2, base with the group's chosen component | State 2 against the adapted pooled comparator | State 3, base alone after deletion |", "|---|---|---|---|---|",
         f"| eICU, 10 consortia, centralised model, AUROC change relative to the pooled model | Reference | {cell('AUROC | confined (strict) minus pooled model', f4)} | {cell('AUROC | confined (strict) minus adapted pooled comparator', f4)} | {cell('AUROC | strict base minus pooled model (centralised)', f4)} |",
         f"| eICU, centralised, deaths unflagged per 1,000 stays at a matched flag rate | Reference | {cell('deaths unflagged per 1,000 | confined (strict) minus pooled model', f2)} | {cell('deaths unflagged per 1,000 | confined (strict) minus adapted pooled comparator', f2)} | {cell('deaths unflagged per 1,000 | strict base minus pooled model', f2)} |",
         f"| eICU, 10 consortia, inside the federation, AUROC change relative to the full federation | Reference | {cell('AUROC | consortium component in the federation minus full federation', f4)}; {f4(res['AUROC | consortium-specific gain in the federation (component minus control)']['d'])} attributable to the consortium's data | Not applicable | {cell('AUROC | federation without the consortium minus full federation', f4)} |",
         f"| eICU, federation, deaths unflagged per 1,000 stays at a matched flag rate | Reference | {cell('deaths unflagged per 1,000 | consortium component minus full federation', f2)} | Not applicable | {cell('deaths unflagged per 1,000 | federation without the consortium minus full federation', f2)} |", "",
         f"Table 1 and text: centralised penalty {cell('AUROC | centralised penalty: retrained without the consortium minus random removal', f4)}; retrained minus pooled {cell('AUROC | retrained without the consortium minus pooled model', f4)}; "
         f"deaths unflagged, retrained minus random removal {cell('deaths unflagged per 1,000 | retrained without the consortium minus random removal', f2)}, retrained minus pooled {cell('deaths unflagged per 1,000 | retrained without the consortium minus pooled model', f2)}; "
         f"federated penalty {cell('AUROC | federated penalty: federation without the consortium minus federation after random removal', f4)}; federation without the consortium minus full federation {cell('AUROC | federation without the consortium minus full federation', f4)}, "
         f"deaths unflagged {cell('deaths unflagged per 1,000 | federation without the consortium minus full federation', f2)}; consortium-specific gain {cell('AUROC | consortium-specific gain in the federation (component minus control)', f4)}; "
         f"adapted pooled comparator minus pooled {cell('AUROC | adapted pooled comparator minus pooled model', f4)}; full federation minus pooled {cell('AUROC | full federation minus pooled model', f4)}."]
    open(f"{OUT}/table2_eicu_{pop}.md", "w").write("\n".join(T) + "\n")
summary_lines.append("per consortium (landmark): penalty, federated penalty, confined minus adapted, confined minus pooled, base minus pooled")
for r_ in wide.itertuples():
    summary_lines.append(f"  {r_.consortium:22s} {r_.penalty_landmark:+.4f} {r_.fed_penalty_landmark:+.4f} {r_.confined_minus_sym_landmark:+.4f} {r_.confined_minus_pooled_landmark:+.4f} {r_.base_minus_pooled_landmark:+.4f}")
open(f"{OUT}/summary.txt", "w").write("\n".join(summary_lines) + "\n"); print("\n".join(summary_lines)); log(f"done, results in {OUT}")
