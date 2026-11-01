"""
exp_v37_exact_p.py. Multiplicity adjustment with exact percentile p-values (review item M6), replacing exp_v28_multiplicity.py
for the v9 manuscript. Every contrast's bootstrap draws are read from the draws files that the patched scripts now write
(boot_draws.py: results/*/draws*.npz; the chunked scripts v13c, v18, v26 and v35 already kept theirs), and the two-sided
p-value is the percentile p-value at zero (2 x the smaller tail, floored at 2/n). Where a draws file is absent the normal
approximation of v28 is used and the contrast is marked, so the summary lists which script summaries still have to be rerun.
Holm (family-wise) and Benjamini-Hochberg (false discovery rate) adjustment over the family as in v28.

Family. FAMILY=v9 (default): the v8 family of 55 contrasts with the eICU landmark cohort promoted to primary: the landmark
versions of the eICU contrasts come from results/v35_landmark, and the all-stays versions stay in the family as the
sensitivity analysis. FAMILY=v8 reproduces the v28 family (55 contrasts) with exact p-values where draws exist.

Env: ROOT, FAMILY=v9|v8. Reads the result files of v5, v8, v13, v13b, v13c, v16, v18, v19, v21, v22, v24, v26, v32, v33 and v35.
Writes results/v37_exact_p/{multiplicity.csv, table_S19.md, summary.txt}.
"""
import os, json, csv, math, glob
import numpy as np
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; FAM = os.environ.get("FAMILY", "v9")
OUT = f"{R}/v37_exact_p"; os.makedirs(OUT, exist_ok=True); Z975 = 1.959964

def load(path): return json.load(open(f"{R}/{path}"))
def jl(p):
    try: return load(p)
    except FileNotFoundError: return None
J = {k: jl(p) for k, p in dict(v13="v13_intervals_1/intervals.json", v13b="v13_intervals_1/matched_flags.json", v13c="v13_intervals_1/extra_contrasts.json",
     v5="v5_partitions/paired.json", v8="v8_modelclass/paired.json", v16="v16_strict/intervals.json", v18a="v18_landmark/all_contrasts.json",
     v18l="v18_landmark/landmark_contrasts.json", v19="v19_protocol/intervals.json", v21="v21_sisa/intervals.json", v22="v22_unlearning_flags/intervals.json",
     v24="v24_symmetric_mimic/intervals.json", v26="v26_eicu_table2/all.json", v32="v32_pooled_full_adapt/intervals.json",
     v33="v33_strict_multigroup/intervals.json", v35l="v35_landmark/contrasts_landmark.json", v35a="v35_landmark/contrasts_all.json", v38e="v38_tuned_unlearning_ext/intervals.json", v38m="v38_tuned_unlearning_ext/matched.json").items()}

# ------------------------------------------------------------------ draws: one loader per source
def npz_draws(path):
    Z = np.load(path, allow_pickle=True); keys = list(Z["keys"]); return {k: np.asarray(Z[f"d{i}"], float) for i, k in enumerate(keys)}
def chunk_draws(pattern):
    B = {}
    for f in sorted(glob.glob(pattern)):
        Zb = np.load(f, allow_pickle=True)["B"].item()
        for k, v in Zb.items(): B.setdefault(k, []).extend(list(v))
    return {k: np.asarray(v, float) for k, v in B.items()}
_cache = {}
SLUG = {"all": "all_clustered_groups", "most_isolated": "five_most_isolated_groups", "mimic_all": "all_clustered_groups", "mimic_most_isolated": "five_most_isolated_groups",
        "eicu_all": "all_held-out_stays", "eicu_landmark": "stays_in_the_unit_at_24_h"}   # the labels the v19, v24 and v32 summaries pass to boot_draws.attach, spaces replaced
def draws_for(src, key, sub=None, field=None):
    """Bootstrap draws for one contrast, or None when the draws file is absent."""
    try:
        if src == "v13": D = _cache.setdefault("v13", npz_draws(f"{R}/v13_intervals_1/draws.npz")); return D.get(key)
        if src == "v13b": D = _cache.setdefault("v13b", npz_draws(f"{R}/v13_intervals_1/draws_matched_flags.npz")); return D.get(key)
        if src == "v13c": D = _cache.setdefault("v13c", chunk_draws(f"{R}/v13_intervals_1/extra_chunk*.npz")); return D.get(key.replace("eICU | all | ", ""))
        if src == "v5": D = _cache.setdefault("v5", npz_draws(f"{R}/v5_partitions/draws_paired.npz")); return D.get(f"{key}|{field or 'auroc'}")
        if src == "v5nw": return _cache["v5nw"]
        if src == "v8": D = _cache.setdefault("v8", npz_draws(f"{R}/v8_modelclass/draws_paired.npz")); return D.get(key)
        if src == "v16": D = _cache.setdefault("v16", npz_draws(f"{R}/v16_strict/draws.npz")); return D.get(key)
        if src in ("v18a", "v18l"): pop = "all" if src == "v18a" else "landmark"; D = _cache.setdefault(src, chunk_draws(f"{R}/v18_landmark/{pop}_chunk*.npz")); return D.get(key)
        if src == "v19": D = _cache.setdefault(("v19", sub), npz_draws(f"{R}/v19_protocol/draws_{SLUG[sub]}.npz")); return D.get(key)
        if src == "v21": D = _cache.setdefault("v21", npz_draws(f"{R}/v21_sisa/draws.npz")); return D.get(key)
        if src == "v22": D = _cache.setdefault("v22", npz_draws(f"{R}/v22_unlearning_flags/draws.npz")); return D.get(key)
        if src == "v38e": D = _cache.setdefault("v38e", npz_draws(f"{R}/v38_tuned_unlearning_ext/draws.npz")); return D.get(key)
        if src == "v38m": D = _cache.setdefault("v38m", npz_draws(f"{R}/v38_tuned_unlearning_ext/matched_draws.npz")); return D.get(key)
        if src == "v24": D = _cache.setdefault(("v24", sub), npz_draws(f"{R}/v24_symmetric_mimic/draws_{SLUG[sub]}.npz")); return D.get(key)
        if src == "v26": D = _cache.setdefault("v26", chunk_draws(f"{R}/v26_eicu_table2/all_chunk*.npz")); return D.get(key)
        if src == "v32": D = _cache.setdefault(("v32", sub), npz_draws(f"{R}/v32_pooled_full_adapt/draws_{SLUG[sub]}.npz")); return D.get(key)
        if src == "v33": D = _cache.setdefault(("v33", sub), npz_draws(f"{R}/v33_strict_multigroup/draws_{sub}.npz")); return D.get(key)
        if src in ("v35l", "v35a"): pop = "landmark" if src == "v35l" else "all"; D = _cache.setdefault(src, chunk_draws(f"{R}/v35_landmark/draws_{pop}_chunk*.npz")); return D.get(key)
    except FileNotFoundError: return None
    return None
def race_without_white():
    acc = {}
    for r in csv.DictReader(open(f"{R}/v5_partitions/results.csv")):
        if r["family"] == "race" and r["model"] in ("exact_group", "exact_random_pat") and r["d_auroc"] not in ("", "nan"):
            acc.setdefault((r["group"], r["model"]), []).append(float(r["d_auroc"]))
    groups = sorted(set(g for g, _ in acc) - {"white"})
    d = np.array([np.mean(acc[(g, "exact_group")]) - np.mean(acc[(g, "exact_random_pat")]) for g in groups])
    rr = np.random.RandomState(0); bs = np.array([rr.choice(d, len(d), True).mean() for _ in range(5000)]); _cache["v5nw"] = bs
    return dict(race_without_white=dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(d)))
J["v5nw"] = race_without_white()
def E(src, key, sub=None, field=None):
    j = J[src]
    if j is None: raise FileNotFoundError(src)
    j = j[sub]["results"] if sub else j.get("results", j); v = j[key]
    if field: v = v[field]
    d = v["d"] if "d" in v else (v["point"] if "point" in v else v["d_auroc"]); return float(d), float(v["ci"][0]), float(v["ci"][1])

V8_FAMILY = [
 ("Withdrawal cost, MIMIC-IV", "Group withdrawal minus matched random removal, 25 clustered groups", "AUROC", ("v13", "MIMIC communities | base_vs_random (structured minus random withdrawal)")),
 ("Withdrawal cost, MIMIC-IV", "Group withdrawal minus pooled model", "AUROC", ("v13", "MIMIC communities | base_vs_original")),
 ("Withdrawal cost, MIMIC-IV", "Group withdrawal minus random removal, readmissions unflagged at a matched flag rate", "per 1,000", ("v13b", "MIMIC communities | missed events per 1,000 at matched flag rate | base minus random")),
 ("Withdrawal cost, MIMIC-IV", "Race groups, withdrawal minus random removal (bootstrap over 7 groups)", "AUROC", ("v5", "race|exact_group_vs_exact_random_pat", None, "auroc")),
 ("Withdrawal cost, MIMIC-IV", "Race groups other than white, withdrawal minus random removal (bootstrap over 6 groups)", "AUROC", ("v5nw", "race_without_white")),
 ("Withdrawal cost, MIMIC-IV", "Gradient-boosted trees, withdrawal minus random removal (bootstrap over groups)", "AUROC", ("v8", "gbdt|exact_group_vs_exact_random_pat")),
 ("Withdrawal cost, MIMIC-IV", "Logistic regression, withdrawal minus random removal (bootstrap over groups)", "AUROC", ("v8", "logreg|exact_group_vs_exact_random_pat")),
 ("Withdrawal cost, eICU", "Consortium withdrawal minus matched random removal, 10 consortia", "AUROC", ("v18a", "AUROC | exact_group minus exact_random (centralised penalty)")),
 ("Withdrawal cost, eICU", "Consortium withdrawal minus pooled model", "AUROC", ("v18a", "AUROC | exact_group minus original")),
 ("Withdrawal cost, eICU", "Deaths unflagged at a matched flag rate, withdrawal minus pooled model", "per 1,000", ("v18a", "events unflagged per 1,000 at matched flag rate | exact_group minus original")),
 ("Withdrawal cost, eICU", "Deaths unflagged at a matched flag rate, withdrawal minus random removal", "per 1,000", ("v18a", "events unflagged per 1,000 at matched flag rate | exact_group minus exact_random")),
 ("Withdrawal cost, eICU", "24-hour landmark, withdrawal minus random removal", "AUROC", ("v18l", "AUROC | exact_group minus exact_random (centralised penalty)")),
 ("Remedies", "Tuned NegGrad+ on the group minus tuned NegGrad+ on random removal", "AUROC", ("v38e", "AUROC | neggrad group minus neggrad random")),
 ("Remedies", "Tuned Fair Unlearning on the group minus tuned Fair Unlearning on random removal", "AUROC", ("v38e", "AUROC | fu_community group minus fu_community random")),
 ("Remedies", "Tuned NegGrad+, readmissions unflagged, group minus random removal", "per 1,000", ("v38e", "unflagged per 1,000 | neggrad group minus neggrad random")),
 ("Remedies", "Tuned Fair Unlearning, readmissions unflagged, group minus random removal", "per 1,000", ("v38e", "unflagged per 1,000 | fu_community group minus fu_community random")),
 ("Remedies", "NegGrad+ at exact retraining's removal minus exact retraining (bootstrap over groups)", "AUROC", ("v38m", "AUROC | neggrad removal-matched (forget gap closest to zero) minus exact group")),
 ("Remedies", "Fair Unlearning at exact retraining's removal minus exact retraining (bootstrap over groups)", "AUROC", ("v38m", "AUROC | fu_community removal-matched (forget gap closest to zero) minus exact group")),
 ("Remedies", "Federation, consortium withdrawal minus random removal", "AUROC", ("v13", "eICU | all | fl_without_vs_fl_random (federated penalty)")),
 ("Remedies", "Federation without the consortium minus full federation", "AUROC", ("v13c", "eICU | all | fl_without minus fl_global (federated, state 3 vs state 1)")),
 ("Remedies", "Federation, deaths unflagged, without the consortium minus full federation", "per 1,000", ("v13b", "eICU | all | missed events per 1,000 at matched flag rate | fl_without minus fl_global")),
 ("Remedies", "Federated penalty, teaching-status consortia", "AUROC", ("v13", "eICU | teaching | fl_without_vs_fl_random (federated penalty)")),
 ("Remedies", "Federated penalty at the 24-hour landmark", "AUROC", ("v18l", "AUROC | fl_without minus fl_random (federated penalty)")),
 ("Confinement, MIMIC-IV", "Base alone (retained-only pipeline) minus pooled model", "AUROC", ("v19", "strict base minus pooled", "all")),
 ("Confinement, MIMIC-IV", "Base with the validation-chosen component minus pooled model", "AUROC", ("v19", "chosen component minus pooled", "all")),
 ("Confinement, MIMIC-IV", "Control component minus base", "AUROC", ("v16", "strict base + control minus strict base")),
 ("Confinement, MIMIC-IV", "Pooled model with a cross-fitted residual component minus pooled model", "AUROC", ("v24", "pooled + cross-fitted residual minus pooled", "all")),
 ("Confinement, MIMIC-IV", "Adapted pooled comparator minus pooled model", "AUROC", ("v24", "pooled with chosen cross-fitted component minus pooled", "all")),
 ("Confinement, MIMIC-IV", "Confined system minus adapted pooled comparator", "AUROC", ("v24", "confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)", "all")),
 ("Confinement, MIMIC-IV", "Confined system minus residual-only comparator", "AUROC", ("v24", "confined (chosen) minus pooled with cross-fitted residual (v17/v19 comparator)", "all")),
 ("Confinement, MIMIC-IV", "Full pooled comparator (component fitted on all of the group's data) minus pooled model", "AUROC", ("v32", "full comparator (chosen) minus pooled", "mimic_all")),
 ("Confinement, MIMIC-IV", "Confined system minus full pooled comparator", "AUROC", ("v32", "confined (chosen) minus full comparator", "mimic_all")),
 ("Confinement, MIMIC-IV", "Five most isolated groups, base minus pooled model", "AUROC", ("v19", "strict base minus pooled", "most_isolated")),
 ("Confinement, MIMIC-IV", "Five most isolated groups, confined minus residual-only comparator", "AUROC", ("v19", "chosen component minus pooled + cross-fitted component (protocol vs best pooled)", "most_isolated")),
 ("Confinement, MIMIC-IV", "Five most isolated groups, confined minus adapted pooled comparator", "AUROC", ("v24", "confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)", "most_isolated")),
 ("Confinement, MIMIC-IV", "Five most isolated groups, full pooled comparator minus pooled model", "AUROC", ("v32", "full comparator (chosen) minus pooled", "mimic_most_isolated")),
 ("Confinement, MIMIC-IV", "Deletion of the component: confined minus base", "AUROC", ("v19", "chosen component minus strict base", "all")),
 ("Confinement, MIMIC-IV", "Readmissions unflagged at a matched flag rate, confined minus pooled model", "per 1,000", ("v19", "events unflagged per 1,000 at matched flag rate, chosen minus pooled", "all")),
 ("Confinement, MIMIC-IV", "Readmissions unflagged, confined minus adapted pooled comparator", "per 1,000", ("v24", "events unflagged per 1,000 at matched flag rate, confined minus symmetric comparator", "all")),
 ("Confinement, MIMIC-IV", "Readmissions unflagged, base minus pooled model", "per 1,000", ("v16", "events unflagged per 1,000 at matched flag rate, strict base minus pooled")),
 ("Several groups withdrawn at once, MIMIC-IV", "Five random groups, confined on the common base minus single-exclusion confined system", "AUROC", ("v33", "common base + chosen component minus single-exclusion confined system", "random_5")),
 ("Several groups withdrawn at once, MIMIC-IV", "Ten random groups, confined on the common base minus single-exclusion confined system", "AUROC", ("v33", "common base + chosen component minus single-exclusion confined system", "random_10")),
 ("Several groups withdrawn at once, MIMIC-IV", "Five most isolated groups, confined on the common base minus single-exclusion confined system", "AUROC", ("v33", "common base + chosen component minus single-exclusion confined system", "most_isolated_5")),
 ("Several groups withdrawn at once, MIMIC-IV", "Fifteen random groups, confined on the common base minus single-exclusion confined system", "AUROC", ("v33", "common base + chosen component minus single-exclusion confined system", "random_15")),
 ("Confinement, eICU", "Federation: consortium component minus full federation", "AUROC", ("v13", "eICU | all | fl_resid_vs_fl_global (sharing premium)")),
 ("Confinement, eICU", "Federation: consortium-specific gain (component minus control component)", "AUROC", ("v13", "eICU | all | fl_resid_vs_fl_resid_random (consortium-specific gain)")),
 ("Confinement, eICU", "Federation: deaths unflagged, component minus full federation", "per 1,000", ("v13b", "eICU | all | missed events per 1,000 at matched flag rate | fl_resid minus fl_global")),
 ("Confinement, eICU", "Centralised model: base alone (retained-only pipeline) minus pooled model", "AUROC", ("v26", "AUROC | strict base minus pooled")),
 ("Confinement, eICU", "Centralised model: confined minus pooled model", "AUROC", ("v26", "AUROC | strict confined minus pooled")),
 ("Confinement, eICU", "Centralised model: confined minus adapted pooled comparator", "AUROC", ("v26", "AUROC | strict confined minus symmetric comparator")),
 ("Confinement, eICU", "Centralised model: full pooled comparator minus pooled model", "AUROC", ("v32", "full comparator (chosen) minus pooled", "eicu_all")),
 ("Confinement, eICU", "Centralised model: deaths unflagged, confined minus pooled model", "per 1,000", ("v26", "unflagged per 1,000 | strict confined minus pooled")),
 ("Confinement, eICU", "Centralised model: deaths unflagged, confined minus adapted pooled comparator", "per 1,000", ("v26", "unflagged per 1,000 | strict confined minus symmetric comparator")),
 ("Confinement, eICU", "Centralised model: deaths unflagged, base minus pooled model", "per 1,000", ("v26", "unflagged per 1,000 | strict base minus pooled")),
 ("Sharded training", "Group-aligned ensemble minus pooled model, before deletion", "AUROC", ("v21", "SISA full minus pooled (state 1 under SISA)")),
 ("Sharded training", "Group-aligned ensemble minus pooled model, after deleting the group's shard", "AUROC", ("v21", "SISA shard deleted minus pooled")),
 ("Sharded training", "Confined system minus group-aligned ensemble", "AUROC", ("v21", "confined (chosen component) minus SISA full")),
]
# v9: the eICU landmark cohort is primary (v35); the v8 eICU rows become the sensitivity analysis on all stays
LANDMARK = [
 ("Withdrawal cost, eICU (24-hour landmark, primary)", "Consortium withdrawal minus matched random removal, 10 consortia", "AUROC", ("v35l", "AUROC | centralised penalty: retrained without the consortium minus random removal")),
 ("Withdrawal cost, eICU (24-hour landmark, primary)", "Consortium withdrawal minus pooled model", "AUROC", ("v35l", "AUROC | retrained without the consortium minus pooled model")),
 ("Withdrawal cost, eICU (24-hour landmark, primary)", "Deaths unflagged at a matched flag rate, withdrawal minus random removal", "per 1,000", ("v35l", "deaths unflagged per 1,000 | retrained without the consortium minus random removal")),
 ("Withdrawal cost, eICU (24-hour landmark, primary)", "Deaths unflagged at a matched flag rate, withdrawal minus pooled model", "per 1,000", ("v35l", "deaths unflagged per 1,000 | retrained without the consortium minus pooled model")),
 ("Remedies (24-hour landmark, primary)", "Federation, consortium withdrawal minus random removal", "AUROC", ("v35l", "AUROC | federated penalty: federation without the consortium minus federation after random removal")),
 ("Remedies (24-hour landmark, primary)", "Federation without the consortium minus full federation", "AUROC", ("v35l", "AUROC | federation without the consortium minus full federation")),
 ("Remedies (24-hour landmark, primary)", "Federation, deaths unflagged, without the consortium minus full federation", "per 1,000", ("v35l", "deaths unflagged per 1,000 | federation without the consortium minus full federation")),
 ("Confinement, eICU (24-hour landmark, primary)", "Federation: consortium component minus full federation", "AUROC", ("v35l", "AUROC | consortium component in the federation minus full federation")),
 ("Confinement, eICU (24-hour landmark, primary)", "Federation: consortium-specific gain (component minus control component)", "AUROC", ("v35l", "AUROC | consortium-specific gain in the federation (component minus control)")),
 ("Confinement, eICU (24-hour landmark, primary)", "Federation: deaths unflagged, component minus full federation", "per 1,000", ("v35l", "deaths unflagged per 1,000 | consortium component minus full federation")),
 ("Confinement, eICU (24-hour landmark, primary)", "Centralised model: base alone (retained-only pipeline) minus pooled model", "AUROC", ("v35l", "AUROC | strict base minus pooled model (centralised)")),
 ("Confinement, eICU (24-hour landmark, primary)", "Centralised model: confined minus pooled model", "AUROC", ("v35l", "AUROC | confined (strict) minus pooled model")),
 ("Confinement, eICU (24-hour landmark, primary)", "Centralised model: confined minus adapted pooled comparator", "AUROC", ("v35l", "AUROC | confined (strict) minus adapted pooled comparator")),
 ("Confinement, eICU (24-hour landmark, primary)", "Centralised model: deaths unflagged, confined minus pooled model", "per 1,000", ("v35l", "deaths unflagged per 1,000 | confined (strict) minus pooled model")),
 ("Confinement, eICU (24-hour landmark, primary)", "Centralised model: deaths unflagged, confined minus adapted pooled comparator", "per 1,000", ("v35l", "deaths unflagged per 1,000 | confined (strict) minus adapted pooled comparator")),
 ("Confinement, eICU (24-hour landmark, primary)", "Centralised model: deaths unflagged, base minus pooled model", "per 1,000", ("v35l", "deaths unflagged per 1,000 | strict base minus pooled model")),
]
if FAM == "v9":
    drop = {"24-hour landmark, withdrawal minus random removal", "Federated penalty at the 24-hour landmark"}   # now in LANDMARK
    FAMILY = LANDMARK + [(sec + (" (all stays, sensitivity)" if "eICU" in sec or lab.startswith("Federat") else ""), lab, unit, src) for sec, lab, unit, src in V8_FAMILY if lab not in drop]
else: FAMILY = V8_FAMILY

def phi_two_sided(z): return math.erfc(abs(z) / math.sqrt(2))
V40 = {("v19", "events unflagged per 1,000 at matched flag rate, chosen minus pooled", "all"): "v19",
       ("v21", "confined (chosen component) minus SISA full"): "v21",
       ("v33", "common base + chosen component minus single-exclusion confined system", "random_15"): "v33"}
rows, missing = [], []
for sec, label, unit, src in FAMILY:
    try: d, lo, hi = E(*src)
    except (FileNotFoundError, KeyError, TypeError) as e: missing.append(f"{src[0]}: {src[1]} ({type(e).__name__})"); continue
    se = (hi - lo) / (2 * Z975); z = d / se if se > 0 else float("inf"); p_norm = phi_two_sided(z)
    dr = draws_for(*src)
    if dr is not None and len(dr) >= 100:
        v = np.asarray(dr, float); v = v[~np.isnan(v)]; nb = int(len(v)); p_ex = min(1.0, max(2.0 / nb, 2 * min((v <= 0).mean(), (v >= 0).mean())))
        # With n draws the percentile p cannot go below 2/n (0.002 for 1,000), which no contrast could survive under Holm over ~70 contrasts.
        # Where no draw crosses zero the exact p is only known to be below 2/n, so the normal-approximation p (itself below 2/n) is used there.
        if p_ex > 2.0 / nb: p, kind = p_ex, "exact"
        else: p, kind = min(p_norm, 2.0 / nb), f"beyond bootstrap resolution (<{2.0 / nb:.3f}); normal approximation"
    else: p = p_norm; kind = "normal approximation (draws not found)"; nb = 0; missing.append(f"draws for {src[0]}: {src[1]}")
    v40 = V40.get(tuple(src))   # review item 7: the three contrasts whose Holm decision depended on resolution, re-drawn 10,000 times by exp_v40_boot10k.py
    if v40 and kind.startswith("beyond") and os.path.exists(f"{R}/v40_boot10k/{v40}.json"):
        J40 = json.load(open(f"{R}/v40_boot10k/{v40}.json")); p, kind, nb = J40["p_exact"], f"exact, {J40['n_boot']:,} draws (v40)", J40["n_boot"]
    rows.append(dict(section=sec, contrast=label, unit=unit, estimate=d, lower=lo, upper=hi, excludes_zero=(lo > 0) or (hi < 0), p=p, p_kind=kind, n_draws=nb, p_normal=p_norm, source=src[0] + ": " + src[1]))
m = len(rows); order = sorted(range(m), key=lambda i: rows[i]["p"])
running = 0.0
for rank, i in enumerate(order, start=1):
    adj = min(1.0, (m - rank + 1) * rows[i]["p"]); running = max(running, adj); rows[i]["p_holm"] = running
q = 1.0
for rank in range(m, 0, -1):
    i = order[rank - 1]; q = min(q, min(1.0, m * rows[i]["p"] / rank)); rows[i]["q_bh"] = q
for r in rows: r["holm_05"] = r["p_holm"] < 0.05; r["bh_05"] = r["q_bh"] < 0.05
with open(f"{OUT}/multiplicity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def fmt(x, unit): return f"{x:+.4f}" if unit == "AUROC" else f"{x:+.2f}"
def fp(p): return "<0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"
n_exact = sum(r["p_kind"].startswith("exact") for r in rows); n_floor = sum(r["p_kind"].startswith("beyond") for r in rows)
lines = [f"**Table S19.** Multiplicity adjustment for the contrasts reported in the main text. Two-sided p-values are exact percentile p-values from the stored bootstrap draws ({n_exact} of {m} contrasts); where no draw crossed zero ({n_floor} contrasts) the p-value is below the resolution of 1,000 draws and the normal approximation to the interval is given, adjusted over all {m} contrasts by Holm's method (family-wise error rate) and by the Benjamini-Hochberg procedure (false discovery rate). Unadjusted intervals are those reported in the text and tables.", "",
         "| Contrast | Estimate (95% interval) | p | Holm-adjusted p | Benjamini-Hochberg q |", "|---|---|---|---|---|"]
sec = None
for r in rows:
    if r["section"] != sec: sec = r["section"]; lines.append(f"| **{sec}** | | | | |")
    lines.append(f"| {r['contrast']} | {fmt(r['estimate'], r['unit'])} ({fmt(r['lower'], r['unit'])} to {fmt(r['upper'], r['unit'])}) | {fp(r['p'])}{'' if r['p_kind'].startswith('exact') else ' (approx.)'} | {fp(r['p_holm'])} | {fp(r['q_bh'])} |")
open(f"{OUT}/table_S19.md", "w").write("\n".join(lines) + "\n")
claimed = [r for r in rows if r["excludes_zero"]]; lost_h = [r for r in claimed if not r["holm_05"]]; lost_b = [r for r in claimed if not r["bh_05"]]
s = [f"v37 exact-p multiplicity (family {FAM}): {m} contrasts; {n_exact} with exact percentile p-values; {len(claimed)} intervals exclude zero; after Holm {len(claimed) - len(lost_h)} remain below 0.05, after Benjamini-Hochberg {len(claimed) - len(lost_b)}.",
     "Lost under Holm:"] + [f"  {r['section']} | {r['contrast']}: {fmt(r['estimate'], r['unit'])} ({fmt(r['lower'], r['unit'])} to {fmt(r['upper'], r['unit'])}), p {fp(r['p'])} ({r['p_kind']}), Holm {fp(r['p_holm'])}, BH q {fp(r['q_bh'])}" for r in lost_h] + \
    ["Lost under Benjamini-Hochberg:"] + [f"  {r['section']} | {r['contrast']}: Holm {fp(r['p_holm'])}, BH q {fp(r['q_bh'])}" for r in lost_b] + \
    ["Exact against normal-approximation p-values, largest differences:"] + [f"  {r['contrast']}: exact {fp(r['p'])} normal {fp(r['p_normal'])}" for r in sorted([r for r in rows if r['p_kind'] == 'exact'], key=lambda r: -abs(math.log10(max(r['p'], 1e-6)) - math.log10(max(r['p_normal'], 1e-6))))[:8]] + \
    (["Missing (rerun the script's summary with the patched version, or exp_v35 for the landmark rows):"] + [f"  {x}" for x in missing] if missing else ["All contrasts and draws found."])
open(f"{OUT}/summary.txt", "w").write("\n".join(s) + "\n"); print("\n".join(s)); print(f"\nwritten {OUT}/multiplicity.csv, table_S19.md, summary.txt")
