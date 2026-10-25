"""
exp_v28_multiplicity.py. Multiplicity adjustment for every contrast reported in the main text (Results, Tables 1 and 2),
from the result files behind the manuscript. No models, no bootstrap reruns.

Each contrast is reported in the manuscript with a 95% percentile bootstrap interval. The bootstrap draws were not kept, so a
two-sided p-value is derived from the interval under a normal approximation (Altman and Bland, BMJ 2011;343:d2304):
standard error = (upper - lower) / (2 x 1.96), z = estimate / standard error, p = 2(1 - Phi(|z|)). The p-values are then
adjusted over the whole family with Holm's step-down method (family-wise error) and, as a secondary summary, with the
Benjamini-Hochberg procedure (false discovery rate). Adjustment bears on claims of a difference; a contrast whose interval
includes zero is reported in the manuscript as not detected, and adjustment does not change that reading.

Env: ROOT (default ~/mimic). Reads results/{v5_partitions, v8_modelclass, v13_intervals_1, v16_strict, v18_landmark,
v19_protocol, v21_sisa, v22_unlearning_flags, v24_symmetric_mimic, v26_eicu_table2, v32_pooled_full_adapt, v33_strict_multigroup}.
v28.1 (v8 manuscript): adds the full pooled comparator (v32), several groups withdrawn at once (v33) and the race contrast
without the white group, whose random control was capped at all other training patients (computed here from v5's
results.csv with v5's paired bootstrap over groups, 5,000 draws, seed 0). Writes
results/v28_multiplicity/{multiplicity.csv, table_S19.md, summary.txt}.
"""
import os, json, csv, math
import numpy as np
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"; OUT = f"{R}/v28_multiplicity"; os.makedirs(OUT, exist_ok=True)
Z975 = 1.959964

def load(path): return json.load(open(f"{R}/{path}"))
J = {k: load(p) for k, p in dict(v13="v13_intervals_1/intervals.json", v13b="v13_intervals_1/matched_flags.json", v13c="v13_intervals_1/extra_contrasts.json",
     v5="v5_partitions/paired.json", v8="v8_modelclass/paired.json", v16="v16_strict/intervals.json", v18a="v18_landmark/all_contrasts.json",
     v18l="v18_landmark/landmark_contrasts.json", v19="v19_protocol/intervals.json", v21="v21_sisa/intervals.json", v22="v22_unlearning_flags/intervals.json",
     v24="v24_symmetric_mimic/intervals.json", v26="v26_eicu_table2/all.json", v32="v32_pooled_full_adapt/intervals.json",
     v33="v33_strict_multigroup/intervals.json").items()}
def race_without_white():
    """v5 paired contrast over the six race groups other than white (seed means of exact_group minus exact_random_pat d_auroc)."""
    acc = {}
    for r in csv.DictReader(open(f"{R}/v5_partitions/results.csv")):
        if r["family"] == "race" and r["model"] in ("exact_group", "exact_random_pat") and r["d_auroc"] not in ("", "nan"):
            acc.setdefault((r["group"], r["model"]), []).append(float(r["d_auroc"]))
    groups = sorted(set(g for g, _ in acc) - {"white"})
    d = np.array([np.mean(acc[(g, "exact_group")]) - np.mean(acc[(g, "exact_random_pat")]) for g in groups])
    rr = np.random.RandomState(0); bs = [rr.choice(d, len(d), True).mean() for _ in range(5000)]
    return dict(race_without_white=dict(d=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(d)))
J["v5nw"] = race_without_white()
def E(src, key, sub=None, field=None):
    """(estimate, lower, upper) from one result file; sub picks a nested block, field a nested metric"""
    j = J[src]; j = j[sub]["results"] if sub else j.get("results", j); v = j[key]
    if field: v = v[field]
    d = v["d"] if "d" in v else (v["point"] if "point" in v else v["d_auroc"]); return float(d), float(v["ci"][0]), float(v["ci"][1])

FAMILY = [
 # section, label, unit, source
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
 ("Remedies", "NegGrad+ on the group minus NegGrad+ on random removal", "AUROC", ("v22", "AUROC | neggrad group minus neggrad random")),
 ("Remedies", "Fair Unlearning on the group minus Fair Unlearning on random removal", "AUROC", ("v22", "AUROC | fu_community group minus fu_community random")),
 ("Remedies", "NegGrad+, readmissions unflagged, group minus random removal", "per 1,000", ("v22", "unflagged per 1,000 | neggrad group minus neggrad random")),
 ("Remedies", "Fair Unlearning, readmissions unflagged, group minus random removal", "per 1,000", ("v22", "unflagged per 1,000 | fu_community group minus fu_community random")),
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

def phi_two_sided(z): return math.erfc(abs(z) / math.sqrt(2))
rows = []
for sec, label, unit, src in FAMILY:
    d, lo, hi = E(*src); se = (hi - lo) / (2 * Z975); z = d / se if se > 0 else float("inf"); p = phi_two_sided(z)
    rows.append(dict(section=sec, contrast=label, unit=unit, estimate=d, lower=lo, upper=hi, excludes_zero=(lo > 0) or (hi < 0), se=se, z=z, p=p, source=src[0] + ": " + src[1]))
m = len(rows); order = sorted(range(m), key=lambda i: rows[i]["p"])
# Holm step-down
running = 0.0
for rank, i in enumerate(order, start=1):
    adj = min(1.0, (m - rank + 1) * rows[i]["p"]); running = max(running, adj); rows[i]["p_holm"] = running
# Benjamini-Hochberg step-up
q = 1.0
for rank in range(m, 0, -1):
    i = order[rank - 1]; q = min(q, min(1.0, m * rows[i]["p"] / rank)); rows[i]["q_bh"] = q
for r in rows: r["holm_05"] = r["p_holm"] < 0.05; r["bh_05"] = r["q_bh"] < 0.05

with open(f"{OUT}/multiplicity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def fmt(x, unit): return f"{x:+.4f}" if unit == "AUROC" else f"{x:+.2f}"
def fp(p): return "<0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"
lines = [f"**Table S19.** Multiplicity adjustment for the contrasts reported in the main text. Two-sided p-values derived from each 95% bootstrap interval under a normal approximation, adjusted over all {m} contrasts by Holm's method (family-wise error rate) and by the Benjamini-Hochberg procedure (false discovery rate). Unadjusted intervals are those reported in the text and tables.", "",
         "| Contrast | Estimate (95% interval) | p | Holm-adjusted p | Benjamini-Hochberg q |", "|---|---|---|---|---|"]
sec = None
for r in rows:
    if r["section"] != sec: sec = r["section"]; lines.append(f"| **{sec}** | | | | |")
    lines.append(f"| {r['contrast']} | {fmt(r['estimate'], r['unit'])} ({fmt(r['lower'], r['unit'])} to {fmt(r['upper'], r['unit'])}) | {fp(r['p'])} | {fp(r['p_holm'])} | {fp(r['q_bh'])} |")
open(f"{OUT}/table_S19.md", "w").write("\n".join(lines) + "\n")
claimed = [r for r in rows if r["excludes_zero"]]
lost_h = [r for r in claimed if not r["holm_05"]]; lost_b = [r for r in claimed if not r["bh_05"]]
s = [f"v28 multiplicity: {m} contrasts; {len(claimed)} intervals exclude zero; after Holm {len(claimed) - len(lost_h)} remain below 0.05, after Benjamini-Hochberg {len(claimed) - len(lost_b)} remain below 0.05.",
     "Lost under Holm:"] + [f"  {r['section']} | {r['contrast']}: {fmt(r['estimate'], r['unit'])} ({fmt(r['lower'], r['unit'])} to {fmt(r['upper'], r['unit'])}), p {fp(r['p'])}, Holm {fp(r['p_holm'])}, BH q {fp(r['q_bh'])}" for r in lost_h] + \
    ["Lost under Benjamini-Hochberg:"] + [f"  {r['section']} | {r['contrast']}: Holm {fp(r['p_holm'])}, BH q {fp(r['q_bh'])}" for r in lost_b] + \
    ["Smallest Holm-adjusted p among the intervals that include zero: " + fp(min(r["p_holm"] for r in rows if not r["excludes_zero"]))]
open(f"{OUT}/summary.txt", "w").write("\n".join(s) + "\n"); print("\n".join(s)); print(f"\nwritten {OUT}/multiplicity.csv, table_S19.md, summary.txt")
