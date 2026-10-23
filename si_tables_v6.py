"""si_tables_v6.py. New Supplementary Appendix tables for draft v6, built only from result files (pandas/numpy).
Writes nejmai_build/si_tables_v6.md. Run: ROOT=~/mimic python3 si_tables_v6.py"""
import os, json, glob, numpy as np, pandas as pd
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); R = f"{ROOT}/results"; out = []
def iv(d, k): v = d[k]; return f"{v['d']:+.4f} ({v['ci'][0]:+.4f} to {v['ci'][1]:+.4f})"
def ivn(d, k, nd=2): v = d[k]; return f"{v['d']:+.{nd}f} ({v['ci'][0]:+.{nd}f} to {v['ci'][1]:+.{nd}f})"
def md(df): return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])
def choices(path_glob, key):
    c = {}
    for f in glob.glob(path_glob):
        r = json.load(open(f)); c.setdefault(r[key], []).append(r["selection"]["chosen"])
    return {k: ", ".join(sorted(v)) for k, v in c.items()}
ab = {"residual": "R", "head": "H", "finetuned": "F"}
def short(s): return "".join(ab[x] for x in s.split(", "))
# ---------------- provenance
p16 = pd.read_csv(f"{R}/v16_strict/provenance.csv"); p23 = pd.read_csv(f"{R}/v23_eicu_strict/provenance.csv")
rows = [["MIMIC-IV (25 clustered groups and 2 census groups)", len(p16), f"{int(p16.preprocessing_rows.min()):,} to {int(p16.preprocessing_rows.max()):,}", int(p16.group_patients_in_preprocessing.sum()), int(p16.heldout_rows_in_preprocessing.sum()),
         "fixed 12 epochs, no validation data", f"{int(p16.vocabulary_overlap_with_whole_cohort.min())} to {int(p16.vocabulary_overlap_with_whole_cohort.max())} of 100", "single checksum per run"],
        ["eICU centralised (10 consortia)", len(p23), f"{int(p23.preprocessing_rows.min()):,} to {int(p23.preprocessing_rows.max()):,}", int(p23.consortium_rows_in_preprocessing.sum()), int(p23.heldout_rows_in_preprocessing.sum()),
         f"early stopping on validation stays outside the consortium ({int(p23.consortium_rows_in_fit_or_val.sum())} consortium stays in fit or validation)", "not applicable", "unchanged in all runs" if p23.base_unchanged.all() else "CHANGED"]]
out += ["**Table S10.** Provenance of the retained-only shared pipeline. One record per group or consortium and seed; counts are summed over runs.", "",
        md(pd.DataFrame(rows, columns=["Setting", "Runs", "Rows fitting preprocessing", "Group rows in preprocessing", "Held-out rows in preprocessing", "Model selection", "Diagnosis vocabulary shared with whole-cohort vocabulary", "Base checksum before and after component fitting"])), ""]
# ---------------- MIMIC per group
S19 = pd.read_csv(f"{R}/v19_protocol/summary.csv").pivot_table(index="group", columns="arm", values="auroc")
S24 = pd.read_csv(f"{R}/v24_symmetric_mimic/summary.csv").pivot_table(index="group", columns="arm", values="auroc")
v5 = pd.read_csv(f"{R}/v5_partitions/results.csv"); v5 = v5[v5.family == "community_trainfit"]
iso = v5[v5.model == "original"].groupby("group").isolation_ratio.mean(); ntr = v5[v5.model == "original"].groupby("group").n_train_patients.first()
c19 = choices(f"{R}/v19_protocol/runs/community*.json", "group"); c24 = choices(f"{R}/v24_symmetric_mimic/runs/*.json", "group")
g = [x for x in S19.index if x.startswith("community")]
T = pd.DataFrame({"Group": [x.split("/")[1] for x in g], "Training patients": [f"{int(ntr[x.split('/')[1]]):,}" for x in g], "Isolation ratio": [f"{iso[x.split('/')[1]]:.2f}" for x in g],
                  "Pooled": [f"{S19.loc[x,'pooled']:.4f}" for x in g], "Strict base (state 3)": [f"{S19.loc[x,'strict_base']:.4f}" for x in g],
                  "Confined (state 2)": [f"{S19.loc[x,'chosen']:.4f}" for x in g], "Type chosen, confined": [short(c19[x]) for x in g],
                  "Adapted pooled comparator": [f"{S24.loc[x,'pooled_cf_chosen']:.4f}" for x in g], "Type chosen, pooled": [short(c24[x]) for x in g],
                  "Residual-only comparator": [f"{S19.loc[x,'pooled_cf']:.4f}" for x in g]})
out += ["**Table S11.** The protocol on MIMIC-IV, by group. AUROC on each group's held-out patients, means of three seeds. Component types chosen by validation in each seed (R residual, H representation head, F fine-tuned copy). The strict base and the confined system use the retained-only pipeline; the pooled comparators use the pooled model's pipeline.", "", md(T), ""]
# ---------------- eICU per consortium
S23 = pd.read_csv(f"{R}/v23_eicu_strict/summary.csv").pivot_table(index="consortium", columns="arm", values="auroc")
S25 = pd.read_csv(f"{R}/v25_symmetric_eicu/summary.csv").pivot_table(index="consortium", columns="arm", values="auroc")
c23 = choices(f"{R}/v23_eicu_strict/runs/*.json", "consortium"); c25 = choices(f"{R}/v25_symmetric_eicu/runs/*.json", "consortium")
r23 = {json.load(open(f))["consortium"]: json.load(open(f)) for f in glob.glob(f"{R}/v23_eicu_strict/runs/*__s0.json")}
k = list(S23.index)
T = pd.DataFrame({"Consortium": [x.replace("_-_", " to ").replace(">=_", "≥").replace("_", " ") for x in k], "Training patients": [f"{r23[x]['n_train_patients']:,}" for x in k], "Held-out stays": [f"{r23[x]['n_test_stays']:,}" for x in k],
                  "Pooled": [f"{S23.loc[x,'pooled']:.4f}" for x in k], "Original-pipeline base": [f"{S23.loc[x,'base_v72']:.4f}" for x in k], "Strict base (state 3)": [f"{S23.loc[x,'base']:.4f}" for x in k],
                  "Confined (state 2)": [f"{S23.loc[x,'chosen']:.4f}" for x in k], "Type chosen, confined": [short(c23[x]) for x in k],
                  "Adapted pooled comparator": [f"{S25.loc[x,'pooled_cf_chosen']:.4f}" for x in k], "Type chosen, pooled": [short(c25[x]) for x in k]})
out += ["**Table S12.** The protocol on the centralised eICU model, by consortium. AUROC on each consortium's held-out stays, means of three seeds; types as in Table S11.", "", md(T), ""]
# ---------------- comparators
I17 = json.load(open(f"{R}/v17_crossfit/intervals.json"))["results"]; I24 = json.load(open(f"{R}/v24_symmetric_mimic/intervals.json")); I25 = json.load(open(f"{R}/v25_symmetric_eicu/intervals.json")); I26 = json.load(open(f"{R}/v26_eicu_table2/all.json"))
G17 = pd.DataFrame([json.load(open(f))["logit_gap"] for f in glob.glob(f"{R}/v17_crossfit/runs/*.json")])
rows = [["MIMIC-IV", "Pooled model + residual component fitted in-sample, minus pooled", iv(I17, "pooled + in-sample component minus pooled")],
        ["MIMIC-IV", "Pooled model + residual component, cross-fitted, minus pooled", iv(I17, "pooled + cross-fitted component minus pooled")],
        ["MIMIC-IV", "Pooled model + head, cross-fitted, minus pooled", iv(I24["all"]["results"], "pooled + cross-fitted head minus pooled")],
        ["MIMIC-IV", "Pooled model + fine-tuned copy, cross-fitted, minus pooled", iv(I24["all"]["results"], "pooled + cross-fitted fine-tuned copy minus pooled")],
        ["MIMIC-IV", "Adapted pooled comparator (type chosen by validation) minus pooled", iv(I24["all"]["results"], "pooled with chosen cross-fitted component minus pooled")],
        ["MIMIC-IV", "Confined minus adapted pooled comparator", iv(I24["all"]["results"], "confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)")],
        ["MIMIC-IV", "Confined minus residual-only comparator", iv(I24["all"]["results"], "confined (chosen) minus pooled with cross-fitted residual (v17/v19 comparator)")],
        ["MIMIC-IV, five most isolated", "Confined minus adapted pooled comparator", iv(I24["most_isolated"]["results"], "confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)")],
        ["MIMIC-IV, five most isolated", "Confined minus residual-only comparator", iv(I24["most_isolated"]["results"], "confined (chosen) minus pooled with cross-fitted residual (v17/v19 comparator)")],
        ["eICU", "Pooled model + residual, cross-fitted, minus pooled", iv(I25["all"]["results"], "pooled + cross-fitted residual minus pooled")],
        ["eICU", "Pooled model + head, cross-fitted, minus pooled", iv(I25["all"]["results"], "pooled + cross-fitted head minus pooled")],
        ["eICU", "Pooled model + fine-tuned copy, cross-fitted, minus pooled", iv(I25["all"]["results"], "pooled + cross-fitted fine-tuned copy minus pooled")],
        ["eICU", "Adapted pooled comparator minus pooled", iv(I26, "AUROC | symmetric comparator minus pooled")],
        ["eICU", "Confined (strict pipeline) minus adapted pooled comparator", iv(I26, "AUROC | strict confined minus symmetric comparator")],
        ["eICU", "Confined (original pipeline) minus adapted pooled comparator", iv(I25["all"]["results"], "confined (chosen) minus pooled with chosen cross-fitted component (symmetric comparator)")],
        ["eICU", "Confined (original pipeline) minus residual-only comparator", iv(I25["all"]["results"], "confined (chosen) minus pooled with cross-fitted residual (v20 comparator)")],
        ["eICU, 24-hour landmark", "Confined (strict pipeline) minus adapted pooled comparator", iv(I25["landmark"]["results"], "confined strict (chosen) minus symmetric comparator")]]
out += ["**Table S13.** Adapted pooled comparators. Change in AUROC on the group's held-out patients, 95% intervals (MIMIC-IV two-stage bootstrap; eICU hospital-cluster bootstrap; 1,000 draws).", "",
        md(pd.DataFrame(rows, columns=["Setting", "Contrast", "Estimate (95% interval)"])), "",
        f"On the group's MIMIC-IV training admissions the pooled model's AUROC was {G17.auroc_insample_on_train.mean():.3f} in-sample and {G17.auroc_oof_on_train.mean():.3f} out-of-fold (mean of 75 group-seeds), so a component fitted on in-sample predictions learns from scores that are better than the pooled model achieves on new patients.", ""]
# ---------------- census groups under confinement (v16)
S16 = pd.read_csv(f"{R}/v16_strict/summary.csv"); E16 = S16.pivot_table(index="group", columns="arm", values="ece"); S16 = S16.pivot_table(index="group", columns="arm", values="auroc")
rows = []
for gname, lab in [("race/native", "Administrative Native category (576 training patients)"), ("language/non_english", "Non-English speakers")]:
    rows.append([lab, f"{S16.loc[gname,'pooled']:.4f} ({E16.loc[gname,'pooled']:.4f})", f"{S16.loc[gname,'strict_base']:.4f} ({E16.loc[gname,'strict_base']:.4f})",
                 f"{S16.loc[gname,'strict_resid']:.4f} ({E16.loc[gname,'strict_resid']:.4f})", f"{S16.loc[gname,'strict_head']:.4f} ({E16.loc[gname,'strict_head']:.4f})", f"{S16.loc[gname,'strict_control']:.4f} ({E16.loc[gname,'strict_control']:.4f})"])
out += ["**Table S15.** Two census groups under confinement on the retained-only pipeline. AUROC (ECE) on the group's held-out patients, means of three seeds.", "",
        md(pd.DataFrame(rows, columns=["Group", "Pooled", "Strict base", "Base + residual component", "Base + representation head", "Base + control component"])), ""]
# ---------------- SISA
I21 = json.load(open(f"{R}/v21_sisa/intervals.json")); res = I21["results"]
rows = [[k, iv(res, k)] for k in res if not k.startswith("events")] + [[k, ivn(res, k)] for k in res if k.startswith("events")]
ft = I21["full_test"]; o = I21["others"]
out += ["**Table S17.** Group-aligned sharded training (SISA with the 25 groups as shards), MIMIC-IV. Contrasts on the group's held-out patients, two-stage bootstrap (1,000 draws).", "",
        md(pd.DataFrame(rows, columns=["Contrast", "Estimate (95% interval)"])), "",
        f"AUROC on all held-out admissions, means of three seeds: pooled model {np.mean([v['pooled'] for v in ft.values()]):.3f}, uniform ensemble {np.mean([v['sisa'] for v in ft.values()]):.3f}, size-weighted ensemble {np.mean([v['sisa_w'] for v in ft.values()]):.3f}. "
        f"Deleting a group's shard changed other patients' predicted risk by {o['mean_abs_change_mean']:.4f} on average (maximum {o['max_abs_change_max']:.3f}); their AUROC changed by {o['others_auroc_full_minus_minus_mean']:+.5f}.", ""]
# ---------------- landmark
L18 = json.load(open(f"{R}/v18_landmark/landmark_contrasts.json")); pop = L18["population"]; l = L18["results"]; L23 = I25["landmark"]["results"]; S23L = json.load(open(f"{R}/v23_eicu_strict/intervals.json"))["landmark"]["results"]
rows = [["Centralised penalty: retrained without the consortium minus random removal", iv(l, "AUROC | exact_group minus exact_random (centralised penalty)")],
        ["Retrained without the consortium minus pooled model", iv(l, "AUROC | exact_group minus original")],
        ["Federated penalty: federation without the consortium minus federation after random removal", iv(l, "AUROC | fl_without minus fl_random (federated penalty)")],
        ["Federation without the consortium minus full federation", iv(l, "AUROC | fl_without minus fl_global")],
        ["Consortium component in the federation minus full federation", iv(l, "AUROC | fl_resid minus fl_global")],
        ["Consortium-specific gain in the federation (component minus control)", iv(l, "AUROC | fl_resid minus fl_resid_random (consortium-specific)")],
        ["Strict base minus pooled model (centralised)", iv(S23L, "strict base minus pooled")],
        ["Confined (strict) minus pooled model", iv(S23L, "chosen minus pooled")],
        ["Confined (strict) minus adapted pooled comparator", iv(L23, "confined strict (chosen) minus symmetric comparator")],
        ["Deaths unflagged per 1,000, retrained without the consortium minus random removal", ivn(l, "events unflagged per 1,000 at matched flag rate | exact_group minus exact_random")],
        ["Deaths unflagged per 1,000, federation without the consortium minus federation after random removal", ivn(l, "events unflagged per 1,000 at matched flag rate | fl_without minus fl_random")]]
out += [f"**Table S6.** The 24-hour landmark, eICU. Evaluation restricted to held-out stays still in the unit at 24 hours ({pop['held_out_stays']:,} of {pop['of_total']:,}; mortality {pop['mortality']:.3f}; {pop['hospitals']} hospitals); models unchanged. Hospital-cluster bootstrap (1,000 draws).", "",
        md(pd.DataFrame(rows, columns=["Contrast", "Estimate (95% interval)"])), ""]
# ---------------- unlearning flags
I22 = json.load(open(f"{R}/v22_unlearning_flags/intervals.json")); r22 = I22["results"]; w = I22["worse_in"]
lab = {"exact": "Exact retraining", "finetune": "Fine-tuning", "neggrad": "NegGrad+", "fu_race": "Fair Unlearning, race groups protected", "fu_community": "Fair Unlearning, withdrawing group protected"}
rows = [[lab[m], iv(r22, f"AUROC | {m} group minus {m} random"), f"{w[m]['auroc']} of 25", ivn(r22, f"unflagged per 1,000 | {m} group minus {m} random"), ivn(r22, f"unflagged per 1,000 | {m} group minus original")] for m in lab]
out += ["**Table S7.** Unlearning methods on the 25 clustered groups. Each method applied to the group is compared with the same method applied to a patient-matched random removal; readmissions unflagged per 1,000 admissions at the matched flag rate. Two-stage bootstrap (1,000 draws).", "",
        md(pd.DataFrame(rows, columns=["Method", "AUROC, group minus random removal", "Lower in", "Readmissions unflagged per 1,000, group minus random removal", "Readmissions unflagged per 1,000, minus pooled model"])), ""]
open(f"{ROOT}/nejmai_build/si_tables_v6.md", "w").write("\n".join(out)); print("written", len(out), "lines")
