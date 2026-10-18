"""Supplementary Appendix tables for draft v4, built from aggregate result files only. Writes si_tables_v4.md and si_checks_v4.txt."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); RES = f"{ROOT}/results"; OUT = os.path.dirname(os.path.abspath(__file__))
L, C = [], []
def gb(v, n=5000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; r = np.random.RandomState(0); b = [r.choice(v, len(v)).mean() for _ in range(n)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def f4(x): return "" if pd.isna(x) else f"{x:+.4f}"
def ci(t): return f"{t[0]:+.4f} ({t[1]:+.4f} to {t[2]:+.4f})"
def table(title, df, note=""):
    L.append(f"\n{title}\n"); L.append("| " + " | ".join(df.columns) + " |"); L.append("|" + "---|" * len(df.columns))
    for _, r in df.iterrows(): L.append("| " + " | ".join(str(x) for x in r.values) + " |")
    if note: L.append(f"\n{note}\n")
NICE = {"community_trainfit": "Clustered", "race": "Race", "language": "Language", "insurance": "Insurance", "anchor_year": "Anchor year"}
# ---------- S1 MIMIC-IV partitions (v5)
v5 = pd.read_csv(f"{RES}/v5_partitions/results.csv")
W = v5.pivot_table(index=["family", "group"], columns="model", values="auroc"); E = v5.pivot_table(index=["family", "group"], columns="model", values="d_ece")
O = v5[v5.model == "original"].groupby(["family", "group"]).agg(n=("n_train_patients", "first"), iso=("isolation_ratio", "mean"))
S1 = pd.DataFrame({"Partition": [f"{NICE[f]} {g}" for f, g in W.index], "Training patients": [f"{int(n):,}" for n in O.n],
                   "Pooled AUROC": W.original.map(lambda x: f"{x:.4f}").values, "Retrained minus pooled": (W.exact_group - W.original).map(f4).values,
                   "Random removal minus pooled": (W.exact_random_pat - W.original).map(f4).values, "Paired, group minus random": (W.exact_group - W.exact_random_pat).map(f4).values,
                   "ECE change, retrained": E.exact_group.map(f4).values, "ECE change, random": E.exact_random_pat.map(f4).values, "Isolation ratio": O.iso.map(lambda x: f"{x:.2f}").values})
S1["_o"] = [0 if f == "community_trainfit" else 1 for f, g in W.index]; S1 = S1.sort_values(["_o", "Partition"]).drop(columns="_o")
table("Table S1. MIMIC-IV partitions. AUROC on each group's held-out patients after exact retraining without the group and after removal of a patient-matched random set, means of three seeds.", S1)
# ---------- S2 eICU consortia
E7 = pd.read_csv(f"{RES}/v7_eicu_consortium_es/summary.csv"); W7 = E7.pivot_table(index=["family", "group"], columns="model", values="auroc")
M7 = E7[E7.model == "original"].groupby(["family", "group"]).agg(h=("n_hospitals", "first"), fr=("frac_training", "first"))
E12 = pd.read_csv(f"{RES}/v12_eicu_federated_es/summary.csv"); W12 = E12.pivot_table(index=["family", "group"], columns="model", values="auroc")
fig = pd.read_csv(f"{OUT}/figure_data_v4.csv"); fb = fig[(fig.figure == 2) & (fig.panel == "B") & fig.label.str.endswith(" point")]
S2 = pd.DataFrame({"Consortium": [f"{f} {g}".replace("_-_", " to ").replace("_", " ") for f, g in W7.index], "Hospitals": M7.h.astype(int).values,
                   "Share of training stays": M7.fr.map(lambda x: f"{100*x:.1f}%").values, "Pooled AUROC": W7.original.map(lambda x: f"{x:.4f}").values,
                   "Retrained minus pooled": (W7.exact_group - W7.original).map(f4).values, "Paired, centralised": (W7.exact_group - W7.exact_random).map(f4).values,
                   "Fine-tuning minus pooled": (W7.finetune_group - W7.original).map(f4).values, "NegGrad+ minus pooled": (W7.neggrad_group - W7.original).map(f4).values,
                   "Federation AUROC": W12.reindex(W7.index).fl_global.map(lambda x: f"{x:.4f}").values,
                   "Paired, federated": (W12.fl_without - W12.fl_random).reindex(W7.index).map(f4).values})
table("Table S2. eICU consortia. AUROC on each consortium's held-out stays, means of three seeds; paired columns are consortium withdrawal minus withdrawal of the same number of random stays.", S2)
# ---------- S3 confinement per clustered group
V = pd.read_csv(f"{RES}/v11_confinement_1/results.csv")
A = V.groupby(["family", "group", "name"]).d_auroc_vs_original.mean().unstack()
B14 = pd.read_csv(f"{RES}/v14b_pooled_adapt/summary.csv").pivot_table(index="community", columns="arm", values="auroc")
iso = O.loc["community_trainfit"].iso
rows = []
for (f, g), r in A.iterrows():
    b = B14.loc[g] if g in B14.index else None
    rows.append({"Group": f"{NICE[f]} {g}", "Training patients": f"{int(O.loc[(f, g)].n):,}", "Isolation ratio": f"{O.loc[(f, g)].iso:.2f}", "Base": f4(r["base"]),
                 "Base + control": f4(r["resid_random"]), "Base + residual": f4(r["resid_group@1"]), "Base + head": f4(r["rep_group@1"]), "Base + fine-tuned": f4(r["full_group@1"]),
                 "Pooled + residual": f4(b["pooled_resid"] - b["pooled"]) if b is not None else "", "Base + residual minus pooled + residual": f4(b["single_base_resid"] - b["pooled_resid"]) if b is not None else ""})
S3 = pd.DataFrame(rows); S3["_o"] = [0 if x.startswith("Clustered") else 1 for x in S3.Group]; S3 = S3.sort_values(["_o", "Group"]).drop(columns="_o")
table("Table S3. Contribution confinement on MIMIC-IV. Change in AUROC relative to the pooled model on each group's held-out patients, means of three seeds; components fitted on all of the group's training patients.", S3,
      "Pooled + residual columns are from the pooled-plus-component experiment (v14b), which ran on the 25 clustered groups only.")
# ---------- S4 data-size curve
rows = []
for pre, lab in [("platt_group", "Recalibration on the group's data"), ("resid_group", "Residual component"), ("rep_group", "Representation head"), ("full_group", "Fine-tuned copy")]:
    sub = A.loc["community_trainfit"]; rows.append({"Component": lab, **{f"{int(100*fr)}%": ci(gb(sub[f"{pre}@{fr:g}"].values)) for fr in [0.1, 0.25, 0.5, 1.0]}})
table("Table S4. Data-size curve, 25 clustered groups. Change in AUROC relative to the pooled model as each component is fitted on a share of the group's training patients (nested subsets), mean (95% interval, bootstrap over groups).", pd.DataFrame(rows))
# ---------- S5 decision metrics
MF = json.load(open(f"{RES}/v13_intervals_1/matched_flags.json"))["results"]; IV = json.load(open(f"{RES}/v13_intervals_1/intervals.json"))["results"]
rows = [{"Contrast": k, "Estimate (95% interval)": f"{v['d']:+.3f} ({v['ci'][0]:+.3f} to {v['ci'][1]:+.3f})"} for k, v in MF.items() if "false alarms" not in k]
table("Table S5. Decision metrics at a matched flag rate. Each arm flags the same number of the group's patients, the number the pooled model or full federation flagged at the system-wide top-decile threshold, by its own ranking; at matched flag counts the change in false alarms equals the change in events unflagged.", pd.DataFrame(rows))
rows = [{"Contrast": k, "Estimate (95% interval)": f"{v['d']:+.3f} ({v['ci'][0]:+.3f} to {v['ci'][1]:+.3f})"} for k, v in IV.items() if ("missed" in k or "flagged" in k)]
table("Table S6. Fixed-threshold and fixed-capacity counts. Each arm uses the system-wide top-decile threshold of its own scores, or a fixed probability (0.3 readmission, 0.1 death). A better-calibrated component assigns lower probabilities to the group's patients and flags fewer of them, so these counts confound ranking with the flag-rate shift.", pd.DataFrame(rows))
# ---------- S7 intervals (all)
rows = [{"Contrast": k, "Estimate (95% interval)": f"{v['d']:+.4f} ({v['ci'][0]:+.4f} to {v['ci'][1]:+.4f})"} for k, v in IV.items() if not ("missed" in k or "flagged" in k)]
X = json.load(open(f"{RES}/v13_intervals_1/extra_contrasts.json"))
rows += [{"Contrast": k, "Estimate (95% interval)": f"{v['point']:+.4f} ({v['ci'][0]:+.4f} to {v['ci'][1]:+.4f})"} for k, v in X.items()]
table("Table S7. AUROC contrasts with dependence-aware intervals (MIMIC-IV two-stage bootstrap over groups then patients; eICU hospital-cluster bootstrap; 1,000 draws).", pd.DataFrame(rows))
# ---------- S8 multi-group and pooled-plus-component
P14 = json.load(open(f"{RES}/v14_commonbase_1/paired.json")); P14b = json.load(open(f"{RES}/v14b_pooled_adapt/paired.json"))
def pj(v): v = v.get("auroc", v); return f"{v['d']:+.4f} ({v['ci'][0]:+.4f} to {v['ci'][1]:+.4f}); lower in {round(v['frac_worse']*v['n'])} of {v['n']}"
rows = [{"Analysis": "Pooled-plus-component | " + k, "Estimate": pj(v)} for k, v in P14b.items()] + [{"Analysis": ("Cross-fitted adaptation | " if k.startswith("A|") else "Multi-group | ") + k[2:], "Estimate": pj(v)} for k, v in P14.items()]
table("Table S8. Pooled-plus-component, multi-group and cross-fitted analyses (bootstrap over groups of seed-mean differences, 5,000 draws).", pd.DataFrame(rows),
      "Cross-fitted adaptation fits each component on one patient-level half of the group's held-out patients and applies it to the other half, for the pooled model and for the base; it tests adaptation without the group's training data.")
# ---------- S9 preprocessing
P15 = pd.read_csv(f"{RES}/v15_preproc/summary.csv"); W15 = P15.pivot_table(index="community", columns="arm", values="auroc")
ov = P15[P15.arm == "standard_base"].set_index("community").vocab_overlap
S9 = pd.DataFrame({"Group": W15.index, "Standard base": W15.standard_base.map(lambda x: f"{x:.4f}").values, "Strict base": W15.strict_base.map(lambda x: f"{x:.4f}").values,
                   "Standard base + residual": W15.standard_base_resid.map(lambda x: f"{x:.4f}").values, "Strict base + residual": W15.strict_base_resid.map(lambda x: f"{x:.4f}").values,
                   "Vocabulary codes shared (of 100)": ov.reindex(W15.index).astype(int).values})
table("Table S9. Preprocessing sensitivity, five most isolated groups. Standard preprocessing fits the diagnosis vocabulary, imputation and scaling on all training rows; strict preprocessing refits them on the retained rows and retrains the base. AUROC on the group's held-out patients, means of three seeds.", S9)
# ---------- S10 unlearning methods and audits
P10 = json.load(open(f"{RES}/v10_fu_trainfit/paired.json")); S10s = pd.read_csv(f"{RES}/v10_fu_trainfit/summary.csv")
rows = [{"Method": k, "Group minus random removal, AUROC": f"{v['d_auroc']:+.4f} ({v['ci'][0]:+.4f} to {v['ci'][1]:+.4f})", "Lower in": f"{round(v['frac_worse']*v['n'])} of {v['n']}", "ECE difference": f"{v['d_ece']:+.4f}"} for k, v in P10.items()]
g = S10s[S10s.kind == "community"].groupby("model")[["test_auroc", "race_tpr_gap", "race_fpr_gap"]].mean()
table("Table S10. Unlearning methods on the 25 clustered groups, each compared with the same method applied to a patient-matched random forget set.", pd.DataFrame(rows),
      "Full-test-set AUROC and race gaps (true-positive rate, false-positive rate), mean over groups and seeds: " + "; ".join(f"{m} {r.test_auroc:.3f}, {r.race_tpr_gap:.3f}, {r.race_fpr_gap:.3f}" for m, r in g.iterrows()) + ".")
S9v = pd.read_csv(f"{RES}/v9_verify_trainfit/summary.csv"); A6 = pd.read_csv(f"{RES}/v6_groupaudit/summary.csv")
rows = [{"Model": r.model, "Loss attack AUROC (range)": f"{r.loss_mean:.3f} ({r.loss_min:.3f} to {r.loss_max:.3f})", "Reference-model attack AUROC (range)": f"{r.lira_mean:.3f} ({r.lira_min:.3f} to {r.lira_max:.3f})"} for _, r in S9v.iterrows()]
table("Table S11. Record-level membership inference on the clustered groups' contributed records.", pd.DataFrame(rows))
rows = [{"Model": r.model, "Audit A flag rate": f"{r.A_loss_contrib_flag:.3f}", "Audit B flag rate": f"{r.B_flag:.3f}"} for _, r in A6.iterrows()]
table("Table S12. Group-level audits. Flag rate across groups and seeds; the exact-retrain row is the false-positive rate.", pd.DataFrame(rows))
# ---------- checks
sel = V[V.name == "resid_group@1"].adapter_C.astype(str).value_counts().to_dict(); C.append(f"residual component selection (all groups, seeds): {sel}")
shas = V.groupby(["family", "group", "seed"]).base_sha256.nunique(); C.append(f"base checksum unique per group-seed across arms: max {shas.max()}, groups-seeds {len(shas)}")
open(f"{OUT}/si_tables_v4.md", "w").write("\n".join(L)); open(f"{OUT}/si_checks_v4.txt", "w").write("\n".join(C)); print("\n".join(C)); print(len(L), "lines")
