"""
exp_v34_isolation.py. Withdrawal penalty against feature-space isolation for all 49 groups (review items M1c and m12).

Stage 1 (ISO=1; needs the cohorts, no models). Isolation ratios for every group under two definitions.
  v5 definition: for up to 2,000 sampled members (RandomState(3)), the mean distance to the nearest of up to 50,000 sampled
  retained training patients divided by the mean distance to the nearest other sampled member. MIMIC-IV values are read from
  results/v5_partitions/results.csv (the published Table S1 values); the 10 eICU consortia are computed here with the same code.
  Equal-pool definition (m12): both nearest-neighbour searches use pools of the same size. M = min(2,000, n_members) members
  are sampled; each member's within-group neighbour is the nearest of the other M - 1 sampled members, and its retained
  neighbour is the nearest of M - 1 sampled retained training patients. The ratio no longer depends on the retained pool
  being larger than the group. Patient mean vectors of the standardised training rows, as v5.
Stage 2 (from result files). Penalty = AUROC after withdrawal of the group minus AUROC after removal of its matched random
  set, seed means: MIMIC-IV from v5 (exact_group minus exact_random_pat, 39 partitions; the white group's control is capped at
  all other training patients and is flagged), eICU from v7.2 (all held-out stays) and, when results/v35_landmark/
  per_consortium.csv exists, from the 24-hour landmark cohort (POP=landmark, default when available). Ordinary least squares
  of penalty on ratio over all 49 groups, slope interval by bootstrap over groups (5,000 draws, RandomState(0), draws saved),
  Spearman correlation; the same with the 48 groups other than white, within MIMIC-IV, and with family intercepts, as checks.
  Figure data (figure_data.csv) and a print-width figure (figure_isolation.pdf/png) with 7-point text.

Env: ROOT (default ~/mimic), ISO=1 (run stage 1; default 1 if isolation.csv is absent), POP=all|landmark, NBOOT (default 5000),
QUICK=1 (200 members, 2,000 retained, 200 draws). Writes results/v34_isolation/{isolation.csv, regression.json, draws.npz,
figure_data.csv, figure_isolation.pdf, figure_isolation.png, summary.txt, log.txt}.
"""
import os, json, csv, time
import numpy as np
import pandas as pd

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"
OUT = f"{R}/v34_isolation" + ("_quick" if QUICK else ""); os.makedirs(OUT, exist_ok=True)
NBOOT = int(os.environ.get("NBOOT", 200 if QUICK else 5000)); N_MEM, N_RET = (200, 2000) if QUICK else (2000, 50000)
ISO_PATH = f"{OUT}/isolation.csv"; RUN_ISO = os.environ.get("ISO", "1" if not os.path.exists(ISO_PATH) else "0") == "1"
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"quick={QUICK}; stage 1 {'on' if RUN_ISO else 'off (isolation.csv present)'}; n_boot {NBOOT}; out {OUT}")

# ============================================================================== stage 1: isolation ratios
def nn_dist(Q, P, exclude_self=False):
    """Distance from each row of Q to its nearest row of P (Euclidean). exclude_self: Q is P and the point itself is skipped."""
    from sklearn.neighbors import NearestNeighbors
    k = 2 if exclude_self else 1
    d = NearestNeighbors(n_neighbors=k).fit(P).kneighbors(Q)[0]
    return d[:, 1] if exclude_self else d[:, 0]
def isolation_v5(G, Rr, seed=3):
    r = np.random.RandomState(seed)
    gs = G[r.choice(len(G), min(N_MEM, len(G)), False)]; rs = Rr[r.choice(len(Rr), min(N_RET, len(Rr)), False)]
    d_in = nn_dist(gs, gs, True); d_out = nn_dist(gs, rs)
    return dict(ratio_v5=float(d_out.mean() / max(d_in.mean(), 1e-9)), nn_within_v5=float(d_in.mean()), nn_retained_v5=float(d_out.mean()),
                n_members_sampled=int(len(gs)), n_retained_sampled=int(len(rs)))
def isolation_equal(G, Rr, seed=3):
    r = np.random.RandomState(seed); M = min(N_MEM, len(G))
    gs = G[r.choice(len(G), M, False)]; rs = Rr[r.choice(len(Rr), min(M - 1, len(Rr)), False)]
    d_in = nn_dist(gs, gs, True); d_out = nn_dist(gs, rs)
    return dict(ratio_equal=float(d_out.mean() / max(d_in.mean(), 1e-9)), nn_within_equal=float(d_in.mean()), nn_retained_equal=float(d_out.mean()), pool_size=int(len(rs)))

def stage1_mimic():
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
            "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
    X = df[feat].astype(float)
    for c in [c for c in feat if c.startswith("lab_")]:
        X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
    X = X.values.astype(np.float32); sid = df.subject_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    # partition labels in v5's order (race, language, insurance, anchor year, clustered groups)
    def modal_label(col):
        cnt = df.groupby(["subject_id", col]).size().reset_index(name="n").sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id")
        return cnt.set_index("subject_id")[col]
    labels = {}
    race = modal_label("race_group")
    for g in sorted(race.unique()): labels[("race", str(g))] = set(race.index[race == g])
    if "english" in df.columns:
        eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
    ins = modal_label("insurance")
    if (ins == "Medicaid").any(): labels[("insurance", "medicaid")] = set(ins.index[ins == "Medicaid"])
    ayg = modal_label("anchor_year_group")
    for g in sorted(ayg.unique()): labels[("anchor_year", str(g).replace(" ", "").replace("-", "_"))] = set(ayg.index[ayg == g])
    from sklearn.cluster import KMeans
    pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
    km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
    for c in range(25): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
    pat_mean_tr = pd.DataFrame(X[is_train]).groupby(sid[is_train]).mean()
    v5 = pd.read_csv(f"{R}/v5_partitions/results.csv"); v5 = v5[v5.model == "exact_group"].groupby(["family", "group"]).first().reset_index()
    rows = []
    for (fam, name), members in labels.items():
        f_tr = is_train & np.isin(sid, np.fromiter(members, dtype=sid.dtype)); gp = np.array(sorted(set(sid[f_tr])))
        if len(gp) < 300: continue
        rest = np.array(sorted(set(pat_mean_tr.index) - set(gp))); G, Rr = pat_mean_tr.loc[gp].values, pat_mean_tr.loc[rest].values
        ref = v5[(v5.family == fam) & (v5.group == name)]
        if not len(ref): log(f"  {fam}/{name}: not in v5 results.csv, skipped"); continue
        row = dict(dataset="MIMIC-IV", family=fam, group=name, n_members=int(len(gp)), n_retained=int(len(rest)))
        row.update(isolation_v5(G, Rr)); row["ratio_v5_published"] = float(ref.isolation_ratio.iloc[0]); row.update(isolation_equal(G, Rr)); rows.append(row)
        log(f"  {fam}/{name}: n {len(gp)}, v5 ratio {row['ratio_v5']:.3f} (published {row['ratio_v5_published']:.3f}), equal-pool ratio {row['ratio_equal']:.3f}")
    return rows
def stage1_eicu():
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
    META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
    feat = [c for c in df.columns if c not in META]; Xdf = df[feat].astype(float)
    for c in Xdf.columns[Xdf.isna().any()]:
        Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    X = Xdf.values.astype(np.float32); pid = df.uniquepid.values; hosp = df.hospitalid.values
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first()
    pat_mean_tr = pd.DataFrame(X[is_train]).groupby(pid[is_train]).mean()
    rows = []
    for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
        for v in sorted(hattr[col].dropna().unique(), key=str):
            hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp, np.fromiter(hs, dtype=hosp.dtype)); f_tr = is_fit & mem
            if f_tr.sum() < 800 or (is_test & mem).sum() < 100: continue
            name = str(v).replace(" ", "_").replace("/", "_"); gp = np.array(sorted(set(pid[f_tr])))
            rest = np.array(sorted(set(pat_mean_tr.index) - set(gp))); G, Rr = pat_mean_tr.loc[gp].values, pat_mean_tr.loc[rest].values
            row = dict(dataset="eICU", family=fam, group=name, n_members=int(len(gp)), n_retained=int(len(rest)))
            row.update(isolation_v5(G, Rr)); row["ratio_v5_published"] = float("nan"); row.update(isolation_equal(G, Rr)); rows.append(row)
            log(f"  {fam}/{name}: n {len(gp)}, v5 ratio {row['ratio_v5']:.3f}, equal-pool ratio {row['ratio_equal']:.3f}")
    return rows
if RUN_ISO:
    log("stage 1: MIMIC-IV"); rows = stage1_mimic(); log("stage 1: eICU"); rows += stage1_eicu()
    pd.DataFrame(rows).to_csv(ISO_PATH, index=False); log(f"isolation ratios written for {len(rows)} groups")
ISO = pd.read_csv(ISO_PATH)

# ============================================================================== stage 2: penalties and regression
def penalties_mimic():
    v5 = pd.read_csv(f"{R}/v5_partitions/results.csv"); v5 = v5[v5.model.isin(["exact_group", "exact_random_pat"])]
    S = v5.groupby(["family", "group", "model"]).d_auroc.mean().unstack()
    return {(f, g): float(S.loc[(f, g), "exact_group"] - S.loc[(f, g), "exact_random_pat"]) for f, g in S.index}
def penalties_eicu(pop):
    if pop == "landmark":
        pc = pd.read_csv(f"{R}/v35_landmark/per_consortium.csv")
        return {(r.family, r.group): float(r.penalty_landmark) for r in pc.itertuples()}
    S = pd.read_csv(f"{R}/v7_eicu_consortium_es/summary.csv"); S = S[S.model.isin(["exact_group", "exact_random"])].pivot_table(index=["family", "group"], columns="model", values="d_auroc")
    return {(f, g): float(S.loc[(f, g), "exact_group"] - S.loc[(f, g), "exact_random"]) for f, g in S.index}
POP = os.environ.get("POP", "landmark" if os.path.exists(f"{R}/v35_landmark/per_consortium.csv") else "all")
pm, pe = penalties_mimic(), penalties_eicu(POP)
ISO["penalty"] = [(pm if r.dataset == "MIMIC-IV" else pe).get((r.family, r.group), np.nan) for r in ISO.itertuples()]
ISO["control_capped"] = (ISO.family == "race") & (ISO.group == "white")
ISO["family_group"] = np.where(ISO.family == "community_trainfit", "clustered", np.where(ISO.dataset == "eICU", "consortium", "census or era"))
missing = ISO[ISO.penalty.isna()]
if len(missing): log("no penalty for: " + ", ".join(f"{r.family}/{r.group}" for r in missing.itertuples()))
D = ISO.dropna(subset=["penalty"]).reset_index(drop=True); log(f"eICU penalties from the {POP} cohort; {len(D)} groups with a penalty and an isolation ratio")

def ols(x, yv):
    x = np.asarray(x, float); yv = np.asarray(yv, float); A = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(A, yv, rcond=None)[0]; return float(b[1]), float(b[0])
def ols_family(x, yv, fam):
    x = np.asarray(x, float); yv = np.asarray(yv, float); F = pd.get_dummies(pd.Series(fam)).values.astype(float)
    A = np.column_stack([F, x]); b = np.linalg.lstsq(A, yv, rcond=None)[0]; return float(b[-1])
def spearman(x, yv):
    rx = pd.Series(x).rank().values; ry = pd.Series(yv).rank().values; return float(np.corrcoef(rx, ry)[0, 1])
def fit(sub, col, label, family=False):
    x, yv = sub[col].values, sub.penalty.values; n = len(sub)
    slope = ols_family(x, yv, sub.family_group.values) if family else ols(x, yv)[0]
    r = np.random.RandomState(0); draws = np.empty(NBOOT)
    for b in range(NBOOT):
        j = r.randint(0, n, n); sj = sub.iloc[j]
        draws[b] = ols_family(sj[col].values, sj.penalty.values, sj.family_group.values) if family else ols(sj[col].values, sj.penalty.values)[0]
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)); p_exact = float(min(1.0, max(2.0 / NBOOT, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))))
    res = dict(label=label, ratio=col, n_groups=int(n), slope=slope, ci=[lo, hi], p_exact=p_exact, spearman=spearman(x, yv), intercept=(None if family else ols(x, yv)[1]),
               penalty_at_ratio_1=(None if family else float(ols(x, yv)[1] + slope)), penalty_at_ratio_2=(None if family else float(ols(x, yv)[1] + 2 * slope)))
    log(f"  {label:60s} {col:12s} n={n:2d} slope {slope:+.4f} [{lo:+.4f}, {hi:+.4f}] p {p_exact:.4f} rho {res['spearman']:+.3f}")
    return res, draws
results, draws = [], {}
for col in ["ratio_equal", "ratio_v5"]:
    for label, sub, fam in [(f"all {len(D)} groups", D, False), (f"{int((~D.control_capped).sum())} groups, white group excluded (capped control)", D[~D.control_capped], False),
                            ("all groups, family intercepts", D, True), ("MIMIC-IV partitions only", D[D.dataset == "MIMIC-IV"], False),
                            ("clustered groups only", D[D.family == "community_trainfit"], False), ("eICU consortia only", D[D.dataset == "eICU"], False)]:
        if len(sub) < 4: continue
        res, dr = fit(sub, col, label, fam); results.append(res); draws[f"{col} | {label}"] = dr
np.savez_compressed(f"{OUT}/draws.npz", **{k.replace(" ", "_").replace("|", "_").replace(",", ""): v for k, v in draws.items()}, keys=np.array(list(draws), dtype=object))
json.dump(dict(population_eicu=POP, n_boot=NBOOT, bootstrap="over groups, RandomState(0), draws in draws.npz", results=results,
               groups=[dict(dataset=r.dataset, family=r.family, group=r.group, family_group=r.family_group, n_members=int(r.n_members), ratio_v5=float(r.ratio_v5),
                            ratio_equal=float(r.ratio_equal), penalty=float(r.penalty), control_capped=bool(r.control_capped)) for r in D.itertuples()],
               date=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{OUT}/regression.json", "w"), indent=1)
D[["dataset", "family", "group", "family_group", "n_members", "ratio_v5", "ratio_v5_published", "ratio_equal", "penalty", "control_capped"]].to_csv(f"{OUT}/figure_data.csv", index=False)

# ============================================================================== figure (print width, 7-point minimum text)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 70 / 25.4), dpi=300)
    mk = {"clustered": ("o", "#1f4e79"), "census or era": ("s", "#999999"), "consortium": ("^", "#c0504d")}
    for ax, col, title in [(axes[0], "ratio_equal", "Equal-pool isolation ratio"), (axes[1], "ratio_v5", "Isolation ratio (v5 definition, Table S1)")]:
        for fg, (m, c) in mk.items():
            s_ = D[D.family_group == fg]; ax.scatter(s_[col], s_.penalty, marker=m, s=12, facecolors="none" if fg == "census or era" else c, edgecolors=c, linewidths=0.6, label=fg, zorder=3)
        w = D[D.control_capped]; ax.scatter(w[col], w.penalty, marker="x", s=14, c="black", linewidths=0.6, label="white group (capped control)", zorder=4)
        res = next(r_ for r_ in results if r_["ratio"] == col and r_["label"] == f"all {len(D)} groups"); dr = draws[f"{col} | all {len(D)} groups"]
        xs = np.linspace(D[col].min(), D[col].max(), 50); b0 = res["intercept"]; ax.plot(xs, b0 + res["slope"] * xs, color="black", lw=0.8, zorder=2)
        # band: refit intercept per draw is not stored; show the slope interval through the centroid
        xc, yc = D[col].mean(), D.penalty.mean(); lo, hi = res["ci"]
        ax.fill_between(xs, yc + lo * (xs - xc), yc + hi * (xs - xc), color="black", alpha=0.08, lw=0, zorder=1)
        ax.axhline(0, color="black", lw=0.4, ls=":"); ax.axvline(1, color="black", lw=0.4, ls=":")
        ax.set_xlabel(title); ax.set_ylabel("Withdrawal penalty, AUROC")
        ax.set_title(f"slope {res['slope']:+.4f} per unit ratio ({res['ci'][0]:+.4f} to {res['ci'][1]:+.4f}); Spearman {res['spearman']:+.2f}", fontsize=7)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left"); fig.tight_layout()
    fig.savefig(f"{OUT}/figure_isolation.pdf"); fig.savefig(f"{OUT}/figure_isolation.png"); log("figure written")
except Exception as e: log(f"figure skipped: {e}")

L = [f"v34 isolation regression ({time.strftime('%Y-%m-%d %H:%M')}); eICU penalties from the {POP} cohort; {NBOOT} bootstrap draws over groups"]
for r_ in results: L.append(f"  {r_['ratio']:12s} {r_['label']:60s} n={r_['n_groups']:2d} slope {r_['slope']:+.4f} [{r_['ci'][0]:+.4f}, {r_['ci'][1]:+.4f}] p_exact {r_['p_exact']:.4f} Spearman {r_['spearman']:+.3f}")
L.append("groups (dataset family/group n ratio_v5 ratio_equal penalty):")
for r_ in D.sort_values("ratio_equal").itertuples(): L.append(f"  {r_.dataset:8s} {r_.family}/{r_.group:14s} {int(r_.n_members):6d} {r_.ratio_v5:.3f} {r_.ratio_equal:.3f} {r_.penalty:+.4f}" + (" (capped control)" if r_.control_capped else ""))
open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); log(f"done, results in {OUT}")
