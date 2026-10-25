"""
exp_v29_seeds_equivalence.py. Two statistical checks on the main contrasts, from cached predictions only (no training).

1. Training randomness. The manuscript's intervals resample the evaluation data (groups then patients on MIMIC-IV, hospitals on
   eICU) and condition on the fitted models, so the randomness of training enters only through the mean over three seeds. Here
   each contrast is also estimated seed by seed on the full held-out data; the between-seed standard deviation s is reported,
   and a combined interval adds the seed component to the evaluation bootstrap: SE_total = sqrt(SE_boot^2 + s^2 / 3), where
   SE_boot is the standard deviation of the bootstrap draws of the seed-mean contrast. With three seeds s has two degrees of
   freedom, so the combined interval is indicative; the per-seed estimates are shown so the spread can be read directly.
2. Equivalence. An interval that includes zero is not evidence of equivalence. For the contrasts the manuscript describes as
   within 0.002 AUROC (confined system against the adapted pooled comparator; the confined system against the pooled model on
   eICU) the two one-sided tests procedure is applied post hoc with a margin of 0.005 AUROC, the gain of the multilayer
   perceptron over logistic regression on MIMIC-IV (0.704 against 0.699) and about half the MIMIC-IV withdrawal penalty:
   equivalence holds at the 5 percent level when the 90 percent bootstrap interval lies within (-0.005, +0.005). The smallest
   margin at which equivalence would hold (the wider of |5th| and |95th| percentile) is reported with the TOST p-value, with and
   without the seed component.

Env: ROOT (default ~/mimic), NBOOT (default 1000; QUICK=1 gives 20), PART=mimic|eicu|both, SEEDS (MIMIC-IV seeds, default 0,1,2; seeds beyond
the v13 cache are read from results/v30_seed_core/preds), TAG (suffix for the output folder). Reads results/v13_intervals_1/
{preds_mimic.npz, preds_eicu.npz, membership_eicu.json}, results/v24_symmetric_mimic/preds, results/v19_protocol/preds,
results/v25_symmetric_eicu/preds, results/v14_commonbase_1/sets_B.json and derived/eicu_cohort.parquet (hospital ids and the
test split only). Writes results/v29_seeds_equivalence/{draws_*.npz, seeds.csv, equivalence.csv, table_S20.md, table_S21.md,
summary.txt, log.txt}.
"""
import os, json, time, csv, math
import numpy as np

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; PART = os.environ.get("PART", "both")
NBOOT = int(os.environ.get("NBOOT", 20 if QUICK else 1000)); TAG = os.environ.get("TAG", "")
OUT = f"{R}/v29_seeds_equivalence{TAG}" + ("_quick" if QUICK else ""); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]   # MIMIC-IV seeds; seeds beyond the v13 cache are read from results/v30_seed_core/preds
ESEEDS = [0, 1, 2]; MARGIN = 0.005; Z95, Z90 = 1.959964, 1.644854
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"part {PART}; n_boot {NBOOT}; MIMIC seeds {SEEDS}; eICU seeds {ESEEDS}; out {OUT}")

def auc(y, p):
    """AUROC by ranks (ties get average rank)."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    o = np.argsort(p, kind="mergesort"); ps = p[o]
    _, idx, cnt = np.unique(ps, return_index=True, return_counts=True)
    r = np.empty(len(p)); r[o] = np.repeat(idx + (cnt + 1) / 2.0, cnt)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
def pct(v, q): v = np.asarray(v, float); v = v[~np.isnan(v)]; return float(np.percentile(v, q)) if len(v) else float("nan")
def sd(v): v = np.asarray(v, float); return float(v.std(ddof=1)) if len(v) > 1 else float("nan")

seed_rows, eq_rows = [], []
def summarise(name, unit, est_seed, draws_mean, draws_seed, equivalence=False):
    """est_seed: per-seed full-data estimates; draws_mean: bootstrap draws of the seed mean; draws_seed: list of per-seed draw arrays."""
    d = float(np.mean(est_seed)); s = sd(est_seed); se_boot = float(np.std(draws_mean, ddof=1)); se_tot = math.sqrt(se_boot ** 2 + (s ** 2) / len(est_seed))
    lo95, hi95 = pct(draws_mean, 2.5), pct(draws_mean, 97.5)
    row = dict(contrast=name, unit=unit, n_seeds=len(est_seed), estimate=d, lo95=lo95, hi95=hi95, seeds=" ".join(f"{v:+.4f}" for v in est_seed), seed_min=float(min(est_seed)), seed_max=float(max(est_seed)),
               seed_sd=s, se_boot=se_boot, se_total=se_tot, lo95_total=d - Z95 * se_tot, hi95_total=d + Z95 * se_tot,
               single_seed_lo95=min(pct(x, 2.5) for x in draws_seed), single_seed_hi95=max(pct(x, 97.5) for x in draws_seed))
    seed_rows.append(row)
    log(f"  {name:70s} {d:+.4f} [{lo95:+.4f}, {hi95:+.4f}]  seeds {row['seeds']}  sd {s:.4f}  combined [{row['lo95_total']:+.4f}, {row['hi95_total']:+.4f}]")
    if equivalence:
        lo90, hi90 = pct(draws_mean, 5), pct(draws_mean, 95); margin_min = max(abs(lo90), abs(hi90))
        F = np.asarray(draws_mean, float); p_tost = max(float((F <= -MARGIN).mean()), float((F >= MARGIN).mean()))
        lo90t, hi90t = d - Z90 * se_tot, d + Z90 * se_tot; margin_tot = max(abs(lo90t), abs(hi90t))
        p_tost_tot = max(0.5 * math.erfc((d + MARGIN) / se_tot / math.sqrt(2)), 0.5 * math.erfc((MARGIN - d) / se_tot / math.sqrt(2)))
        eq_rows.append(dict(contrast=name, estimate=d, lo90=lo90, hi90=hi90, margin_min=margin_min, equivalent_005=margin_min < MARGIN, p_tost=p_tost,
                            lo90_total=lo90t, hi90_total=hi90t, margin_min_total=margin_tot, equivalent_005_total=margin_tot < MARGIN, p_tost_total=p_tost_tot))
        log(f"    equivalence: 90% [{lo90:+.4f}, {hi90:+.4f}] smallest margin {margin_min:.4f} TOST p {p_tost:.4f} at 0.005; with seeds [{lo90t:+.4f}, {hi90t:+.4f}] margin {margin_tot:.4f} p {p_tost_tot:.4f}")

# ===================================================================================================== MIMIC-IV
def run_mimic():
    V13 = f"{R}/v13_intervals_1"; P13 = np.load(f"{V13}/preds_mimic.npz", allow_pickle=True)["P"].item()
    names = [f"c{c:02d}" for c in range(25)]; iso = json.load(open(f"{R}/v14_commonbase_1/sets_B.json"))["most_isolated"]
    A, Y, PID = {}, {}, {}
    for n in names:
        for s in SEEDS:
            Z24 = np.load(f"{R}/v24_symmetric_mimic/preds/community_trainfit__{n}__s{s}.npz"); Z19 = np.load(f"{R}/v19_protocol/preds/community_trainfit__{n}__s{s}.npz")
            if (n, s, "original") in P13: core = dict(original=P13[(n, s, "original")], base=P13[(n, s, "base")], random=P13[(n, s, "random")])
            else:
                Z30 = np.load(f"{R}/v30_seed_core/preds/community_trainfit__{n}__s{s}.npz"); assert np.array_equal(Z30["pid"], Z24["pid"]), f"{n} s{s}: v30 rows differ from v24"
                core = dict(original=Z30["original"], base=Z30["base"], random=Z30["random"])
            assert len(Z24["y"]) == len(core["original"]) and np.array_equal(Z24["y"], Z19["y"]), f"{n} s{s}: row sets differ between caches"
            gap = float(np.abs(Z24["pooled"] - core["original"]).max())
            if gap > 1e-5: raise SystemExit(f"{n} s{s}: pooled predictions differ between v24 and the core cache (max {gap:.2e}); row order not aligned")
            A[(n, s)] = dict(base=core["base"], random=core["random"], original=core["original"], confined=Z24["confined_chosen"], sym=Z24["pooled_cf_chosen"],
                             pooled=Z24["pooled"], chosen19=Z19["chosen"], strict_base=Z19["strict_base"], pooled_cf=Z19["pooled_cf"])
            Y[n] = Z24["y"].astype(np.float64); PID[n] = Z24["pid"]
    log(f"MIMIC: caches aligned for 25 groups x {len(SEEDS)} seeds; most isolated {iso}")
    CON = [("Withdrawal minus matched random removal (exact retraining)", "base", "random", False), ("Withdrawal minus pooled model", "base", "original", False),
           ("Confined system minus adapted pooled comparator", "confined", "sym", True), ("Confined system minus pooled model", "confined", "pooled", False),
           ("Confined system minus residual-only comparator", "chosen19", "pooled_cf", True), ("Deletion of the component: confined minus base", "chosen19", "strict_base", False)]
    arms = sorted({a for _, a, b, _ in CON for a in (a, b)})
    pat = {}
    for n in names:
        order = np.argsort(PID[n], kind="stable"); ups, starts, counts = np.unique(PID[n][order], return_index=True, return_counts=True); pat[n] = (order, starts, counts)
    def draw_rows(n, r):
        order, starts, counts = pat[n]; j = r.randint(0, len(starts), len(starts)); st, ct = starts[j], counts[j]; cum = np.cumsum(ct)
        return order[np.repeat(st - (cum - ct), ct) + np.arange(cum[-1])]
    def group_aucs(n, rows):
        y = Y[n][rows]; return {s: {a: auc(y, A[(n, s)][a][rows]) for a in arms} for s in SEEDS}
    for label, group_set in [("25 groups", names), ("five most isolated groups", iso)]:
        cache = f"{OUT}/draws_mimic_{'all' if group_set is names else 'isolated'}.npz"
        full = {n: group_aucs(n, np.arange(len(Y[n]))) for n in group_set}
        est = {c[0]: [float(np.mean([full[n][s][c[1]] - full[n][s][c[2]] for n in group_set])) for s in SEEDS] for c in CON}
        if os.path.exists(cache): D = np.load(cache, allow_pickle=True)["D"].item(); log(f"MIMIC {label}: draws loaded from cache")
        else:
            D = {c[0]: {"mean": [], "seed": [[] for _ in SEEDS]} for c in CON}; r = np.random.RandomState(0)
            log(f"MIMIC {label}: bootstrap {NBOOT} draws (groups, then patients)")
            for b in range(NBOOT):
                drawn = r.choice(group_set, len(group_set), True); G = {}
                for j, n in enumerate(drawn): G[j] = group_aucs(n, draw_rows(n, r))
                for c in CON:
                    per_seed = [float(np.mean([G[j][s][c[1]] - G[j][s][c[2]] for j in G])) for s in SEEDS]
                    D[c[0]]["mean"].append(float(np.mean(per_seed)))
                    for i, v in enumerate(per_seed): D[c[0]]["seed"][i].append(v)
                if (b + 1) % 100 == 0: log(f"  draw {b + 1}")
            np.savez_compressed(cache, D=np.array(D, dtype=object))
        for c in CON:
            summarise(f"MIMIC-IV, {label}: {c[0]}", "AUROC", est[c[0]], D[c[0]]["mean"], D[c[0]]["seed"], equivalence=c[3])

# ===================================================================================================== eICU
def run_eicu():
    import pandas as pd
    V13 = f"{R}/v13_intervals_1"; P13 = np.load(f"{V13}/preds_eicu.npz", allow_pickle=True)["P"].item(); cons = {k: set(v) for k, v in json.load(open(f"{V13}/membership_eicu.json")).items()}
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["uniquepid", "hospitalid", "died"])
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; hosp_te = df.hospitalid.values[is_test]; y_te = df.died.values[is_test].astype(np.float64)
    keys = sorted(cons); hospitals = np.unique(hosp_te); h_rows = {h: np.where(hosp_te == h)[0] for h in hospitals}
    V25 = {}
    for k in keys:
        for s in ESEEDS:
            Z = np.load(f"{R}/v25_symmetric_eicu/preds/{k}__s{s}.npz"); V25[(k, s)] = dict(y=Z["y"].astype(np.float64), hosp=Z["hosp"], pooled=Z["pooled"], sym=Z["pooled_cf_chosen"], strict=Z["confined_strict_chosen"])
        assert len(V25[(k, 0)]["y"]) == int(np.isin(hosp_te, list(cons[k])).sum()), f"{k}: v25 rows differ from the consortium's held-out stays"
    h_rows25 = {k: {h: np.where(V25[(k, 0)]["hosp"] == h)[0] for h in np.unique(V25[(k, 0)]["hosp"])} for k in keys}
    log(f"eICU: {len(keys)} consortia, {len(hospitals)} held-out hospitals; v25 caches aligned")
    CON13 = [("Consortium withdrawal minus matched random removal (exact retraining)", "exact_group", "exact_random", False), ("Consortium withdrawal minus pooled model", "exact_group", "original", False),
             ("Federation: consortium withdrawal minus random removal", "fl_without", "fl_random", False)]
    CON25 = [("Centralised model: confined (strict) minus adapted pooled comparator", "strict", "sym", True), ("Centralised model: confined (strict) minus pooled model", "strict", "pooled", True)]
    arms13 = sorted({a for _, a, b, _ in CON13 for a in (a, b)}); arms25 = sorted({a for _, a, b, _ in CON25 for a in (a, b)})
    def stats(rows13, rows25):
        out = {}
        for k in keys:
            r13 = rows13[k]; r25 = rows25[k]; y13 = y_te[r13]; y25 = V25[(k, 0)]["y"][r25]
            out[k] = {s: {**{a: auc(y13, P13[(k, s, a)][r13]) for a in arms13}, **{a: auc(y25, V25[(k, s)][a][r25]) for a in arms25}} for s in ESEEDS}
        return out
    full = stats({k: np.where(np.isin(hosp_te, list(cons[k])))[0] for k in keys}, {k: np.arange(len(V25[(k, 0)]["y"])) for k in keys})
    CON = CON13 + CON25
    est = {c[0]: [float(np.mean([full[k][s][c[1]] - full[k][s][c[2]] for k in keys])) for s in ESEEDS] for c in CON}
    cache = f"{OUT}/draws_eicu.npz"
    if os.path.exists(cache): D = np.load(cache, allow_pickle=True)["D"].item(); log("eICU: draws loaded from cache")
    else:
        D = {c[0]: {"mean": [], "seed": [[] for _ in ESEEDS]} for c in CON}; r = np.random.RandomState(0)
        log(f"eICU: bootstrap {NBOOT} draws over {len(hospitals)} held-out hospitals (one draw feeds all consortia)")
        for b in range(NBOOT):
            drawn = r.choice(hospitals, len(hospitals), True)
            rows13 = {k: np.concatenate([h_rows[h] for h in drawn if h in cons[k]] or [np.zeros(0, int)]) for k in keys}
            rows25 = {k: np.concatenate([h_rows25[k][h] for h in drawn if h in h_rows25[k]] or [np.zeros(0, int)]) for k in keys}
            S = stats(rows13, rows25)
            for c in CON:
                per_seed = [float(np.nanmean([S[k][s][c[1]] - S[k][s][c[2]] for k in keys])) for s in ESEEDS]
                D[c[0]]["mean"].append(float(np.mean(per_seed)))
                for i, v in enumerate(per_seed): D[c[0]]["seed"][i].append(v)
            if (b + 1) % 100 == 0: log(f"  draw {b + 1}")
        np.savez_compressed(cache, D=np.array(D, dtype=object))
    for c in CON: summarise(f"eICU, {len(keys)} consortia: {c[0]}", "AUROC", est[c[0]], D[c[0]]["mean"], D[c[0]]["seed"], equivalence=c[3])

if PART in ("mimic", "both"): run_mimic()
if PART in ("eicu", "both"): run_eicu()

# ===================================================================================================== tables and summary
with open(f"{OUT}/seeds.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys())); w.writeheader(); w.writerows(seed_rows)
if eq_rows:
    with open(f"{OUT}/equivalence.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(eq_rows[0].keys())); w.writeheader(); w.writerows(eq_rows)
def f4(x): return ("+" if x >= 0 else "−") + f"{abs(x):.4f}"
L = ["**Table S20.** Training randomness. Each contrast estimated seed by seed on the full held-out data, with the between-seed standard deviation and an interval that adds the seed component to the evaluation bootstrap (SE_total = sqrt(SE_boot² + s²/k), k the number of seeds; normal approximation). The evaluation-only interval is the manuscript's two-stage (MIMIC-IV) or hospital-cluster (eICU) percentile bootstrap, recomputed here from the cached predictions.", "",
     "| Contrast | Seeds | Seed mean (95% evaluation bootstrap) | Per-seed estimates, lowest to highest | Between-seed SD | 95% interval with the seed component |", "|---|---|---|---|---|---|"]
for r in seed_rows: L.append(f"| {r['contrast']} | {r['n_seeds']} | {f4(r['estimate'])} ({f4(r['lo95'])} to {f4(r['hi95'])}) | {f4(r['seed_min'])} to {f4(r['seed_max'])} | {r['seed_sd']:.4f} | {f4(r['lo95_total'])} to {f4(r['hi95_total'])} |")
open(f"{OUT}/table_S20.md", "w").write("\n".join(L) + "\n")
if eq_rows:
    E = [f"**Table S21.** Equivalence by two one-sided tests, post hoc, margin ±{MARGIN} AUROC. Equivalence holds at the 5 percent level when the 90 percent interval lies within the margin; the smallest margin at which it would hold is the wider of the two 90 percent bounds. First with the evaluation bootstrap as in the manuscript, then with the seed component added (Table S20).", "",
         "| Contrast | Estimate | 90% interval | Smallest margin | TOST p at ±0.005 | 90% interval with the seed component | Smallest margin | TOST p |", "|---|---|---|---|---|---|---|---|"]
    for r in eq_rows: E.append(f"| {r['contrast']} | {f4(r['estimate'])} | {f4(r['lo90'])} to {f4(r['hi90'])} | {r['margin_min']:.4f} | {r['p_tost']:.4f} | {f4(r['lo90_total'])} to {f4(r['hi90_total'])} | {r['margin_min_total']:.4f} | {r['p_tost_total']:.4f} |")
    open(f"{OUT}/table_S21.md", "w").write("\n".join(E) + "\n")
S = [f"v29 ({time.strftime('%Y-%m-%d %H:%M')}; {NBOOT} draws; MIMIC seeds {SEEDS})", "Training randomness (seed mean [evaluation 95%] | seeds, range | sd | combined 95%):"]
for r in seed_rows: S.append(f"  {r['contrast'][:72]:72s} {f4(r['estimate'])} [{f4(r['lo95'])}, {f4(r['hi95'])}] | n={r['n_seeds']} {f4(r['seed_min'])}..{f4(r['seed_max'])} | sd {r['seed_sd']:.4f} | [{f4(r['lo95_total'])}, {f4(r['hi95_total'])}]")
S.append("Equivalence at ±0.005 (90% interval, smallest margin, TOST p; then with the seed component):")
for r in eq_rows: S.append(f"  {r['contrast'][:72]:72s} [{f4(r['lo90'])}, {f4(r['hi90'])}] {r['margin_min']:.4f} p={r['p_tost']:.4f} | [{f4(r['lo90_total'])}, {f4(r['hi90_total'])}] {r['margin_min_total']:.4f} p={r['p_tost_total']:.4f}")
open(f"{OUT}/summary.txt", "w").write("\n".join(S) + "\n"); print("\n" + "\n".join(S)); log(f"done, results in {OUT}")
