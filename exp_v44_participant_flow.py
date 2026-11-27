"""
exp_v44_participant_flow.py. Participant flow and the participants and outcome events of each group (TRIPOD+AI items 20a and
21), aggregate only.

MIMIC-IV: the exclusions of extract_cohort.py applied in sequence to the hospital module (death in hospital; death within 30
days of discharge without readmission; discharge time at or before admission time), then the v5 patient split
(RandomState(0), 20 percent of patients held out). eICU-CRD: the cohort rule of exp_eicu.py (hospital discharge status recorded
as Alive or Expired), the v7 patient split (RandomState(0), 20 percent held out; RandomState(7), 10 percent of training
patients for validation) and the 24-hour landmark (unitdischargeoffset >= 1440 minutes).
Groups: the 25 clustered and 14 administrative MIMIC-IV groups as in v5 (checked against results/v5_partitions/runs) and the
10 eICU consortia as in v7.2 (training stays are the stays used to fit models, validation stays excluded).

Reads hosp/{admissions,patients}.csv.gz, derived/cohort_latest.parquet, derived/cluster_assign_v5.csv, eicu/patient.csv.gz,
derived/eicu_cohort.parquet. Writes results/v44_participant_flow/{flow.json, groups_mimic.csv, groups_eicu.csv,
table_flow.md, table_groups_mimic.md, table_groups_eicu.md}. No count from 1 to 9 is reported (shown as <10). Env: ROOT.
"""
import os, json
import numpy as np
import pandas as pd

ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); OUT = f"{ROOT}/results/v44_participant_flow"; os.makedirs(OUT, exist_ok=True)
def n(x): x = int(x); return "<10" if 0 < x < 10 else f"{x:,}"
def npct(k, tot): k = int(k); return "<10" if 0 < k < 10 else f"{k:,} ({100 * k / tot:.1f})"
flow = {}; checks = []
def check(name, ok): checks.append((name, bool(ok))); print(("ok   " if ok else "FAIL ") + name, flush=True)

# ------------------------------------------------------------------ MIMIC-IV: flow from the hospital module to the cohort
adm = pd.read_csv(f"{ROOT}/hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"], parse_dates=["admittime", "dischtime"])
pat = pd.read_csv(f"{ROOT}/hosp/patients.csv.gz", usecols=["subject_id", "dod"], parse_dates=["dod"])
d = adm.merge(pat, on="subject_id", how="inner").sort_values(["subject_id", "admittime"]).reset_index(drop=True)
d["next_admit"] = d.groupby("subject_id")["admittime"].shift(-1)
d["days_to_next"] = (d["next_admit"] - d["dischtime"]).dt.total_seconds() / 86400
d["readmit30"] = ((d["days_to_next"] >= 0) & (d["days_to_next"] <= 30)).astype(int)
died_inhosp = (d["hospital_expire_flag"] == 1).values
died_30 = (d["dod"].notna() & ((d["dod"] - d["dischtime"]).dt.total_seconds() / 86400 <= 30) & (d["readmit30"] == 0)).values
bad_los = (d["dischtime"] <= d["admittime"]).values
k1 = ~died_inhosp; k2 = k1 & ~died_30; k3 = k2 & ~bad_los
m = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet", columns=["subject_id", "hadm_id", "readmit30", "race_group", "english", "insurance", "anchor_year_group"])
check("MIMIC-IV: sequential exclusions reproduce the cohort's admissions", set(d.loc[k3, "hadm_id"]) == set(m.hadm_id) and int(k3.sum()) == len(m))
check("MIMIC-IV: readmission labels reproduce the cohort's", int(d.loc[k3, "readmit30"].sum()) == int(m.readmit30.sum()))
sid = m.subject_id.values; y = m.readmit30.values.astype(int)
rng = np.random.RandomState(0); pids = m.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = m.subject_id.isin(test_p).values; is_train = ~is_test
def mrow(mask): return dict(units=int(mask.sum()), patients=int(len(set(sid[mask]))), hospitals=1, events=int(y[mask].sum()))
flow["mimic"] = dict(source=dict(units=int(len(d)), patients=int(d.subject_id.nunique()), hospitals=1),
                     excluded_died_in_hospital=int(died_inhosp.sum()), excluded_died_30d_no_readmission=int((k1 & died_30).sum()),
                     excluded_discharge_not_after_admission=int((k2 & bad_los).sum()),
                     analysed=mrow(np.ones(len(m), bool)), training=mrow(is_train), held_out=mrow(is_test))

# ------------------------------------------------------------------ MIMIC-IV: groups exactly as v5 / v38
def modal_label(col):
    cnt = m.groupby(["subject_id", col]).size().reset_index(name="n").sort_values(["subject_id", "n"], ascending=[True, False]).drop_duplicates("subject_id")
    return cnt.set_index("subject_id")[col]
labels = {}
race = modal_label("race_group")
for g in sorted(race.unique()): labels[("race", str(g))] = set(race.index[race == g])
eng = modal_label("english"); labels[("language", "non_english")] = set(eng.index[eng == 0])
ins = modal_label("insurance"); labels[("insurance", "medicaid")] = set(ins.index[ins == "Medicaid"])
ayg = modal_label("anchor_year_group")
for g in sorted(ayg.unique()): labels[("anchor_year", str(g).replace(" ", "").replace("-", "_"))] = set(ayg.index[ayg == g])
assign = pd.read_csv(f"{ROOT}/derived/cluster_assign_v5.csv").set_index("subject_id")["cluster"].astype(int)
for c in range(25): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
RACE = dict(asian="Race, Asian", black="Race, Black", hispanic="Race, Hispanic", native="Race, combined Native category", other="Race, other", unknown="Race, unknown", white="Race, white")
def pretty(fam, name):
    if fam == "community_trainfit": return f"Clustered group {name}"
    if fam == "anchor_year": return "Anchor-year group " + name.replace("_", " to ")
    return {"insurance": "Medicaid", "language": "Non-English speakers"}.get(fam) or RACE[name]
ORDER = {"community_trainfit": 0, "anchor_year": 1, "insurance": 2, "language": 3, "race": 4}
G = []; v5_ok = True; v5_n = 0
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
    if n_tr_p < 300 or n_te_p < 50: continue
    row = dict(family=fam, name=name, group=pretty(fam, name), train_patients=n_tr_p, train_admissions=int(f_tr.sum()), train_events=int(y[f_tr].sum()),
               heldout_patients=n_te_p, heldout_admissions=int(f_te.sum()), heldout_events=int(y[f_te].sum()))
    ref_path = f"{ROOT}/results/v5_partitions/runs/{fam}__{name}__s0.json"
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path)); v5_n += 1
        v5_ok &= (ref["n_train_patients"] == n_tr_p and ref["n_test_patients"] == n_te_p and ref["n_train_rows"] == row["train_admissions"] and ref["n_test_rows"] == row["heldout_admissions"]
                  and abs(ref["prevalence_group_train"] - row["train_events"] / row["train_admissions"]) < 1e-6)
    G.append(row)
G.sort(key=lambda r: (ORDER[r["family"]], r["name"]))
check(f"MIMIC-IV: {len(G)} groups (25 clustered, 14 administrative)", len(G) == 39 and sum(r["family"] == "community_trainfit" for r in G) == 25)
check(f"MIMIC-IV: patients, admissions and training prevalence match the v5 records ({v5_n} of {len(G)} groups have a record)", v5_ok and v5_n == len(G))
pd.DataFrame(G).to_csv(f"{OUT}/groups_mimic.csv", index=False)

# ------------------------------------------------------------------ eICU-CRD: flow, split, landmark and consortia as v7.2
p = pd.read_csv(f"{ROOT}/eicu/patient.csv.gz", usecols=["patientunitstayid", "uniquepid", "hospitalid", "hospitaldischargestatus", "unitdischargeoffset"])
status_ok = p.hospitaldischargestatus.isin(["Alive", "Expired"]).values
e = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["patientunitstayid", "uniquepid", "hospitalid", "region", "numbedscategory", "teachingstatus", "died"])
check("eICU-CRD: the discharge-status rule reproduces the cohort's stays", set(p.loc[status_ok, "patientunitstayid"]) == set(e.patientunitstayid) and int(status_ok.sum()) == len(e))
epid = e.uniquepid.values; eh = e.hospitalid.values; ey = e.died.values.astype(int)
rng = np.random.RandomState(0); ups = e.uniquepid.unique(); rng.shuffle(ups)
etest_p = set(ups[: int(0.2 * len(ups))]); e_test = e.uniquepid.isin(etest_p).values; e_train = ~e_test
rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - etest_p)); rv.shuffle(tr_p)
e_val = e.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & e_train; e_fit = e_train & ~e_val
lm = (p.set_index("patientunitstayid").unitdischargeoffset.reindex(e.patientunitstayid).values >= 1440)
def erow(mask): return dict(units=int(mask.sum()), patients=int(len(set(epid[mask]))), hospitals=int(len(set(eh[mask]))), events=int(ey[mask].sum()))
flow["eicu"] = dict(source=dict(units=int(len(p)), patients=int(p.uniquepid.nunique()), hospitals=int(p.hospitalid.nunique())),
                    excluded_discharge_status_not_recorded=int((~status_ok).sum()),
                    analysed=erow(np.ones(len(e), bool)), training=erow(e_train), validation=erow(e_val), fit=erow(e_fit), held_out=erow(e_test), held_out_landmark=erow(e_test & lm))
hattr = e.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first()
elabels = {}
for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
    for v in sorted(hattr[col].dropna().unique(), key=str):
        elabels[(fam, str(v).replace(" ", "_").replace("/", "_"))] = set(hattr.index[hattr[col] == v])
EP = {"<100": "Beds <100", "100_-_249": "Beds 100 to 249", "250_-_499": "Beds 250 to 499", ">=_500": "Beds ≥500", "f": "Non-teaching", "t": "Teaching", "False": "Non-teaching", "True": "Teaching"}
EG = []
for (fam, name), hs in elabels.items():
    mem = np.isin(eh, np.fromiter(hs, dtype=eh.dtype)); f_tr, f_te = e_fit & mem, e_test & mem
    if f_tr.sum() < 800 or f_te.sum() < 100: continue
    EG.append(dict(family=fam, name=name, group=(f"Region {name}" if fam == "region" else EP[name]), hospitals=len(hs), train_stays=int(f_tr.sum()), train_events=int(ey[f_tr].sum()),
                   share_of_training_stays=float(f_tr.sum() / e_fit.sum()), heldout_stays=int(f_te.sum()), heldout_events=int(ey[f_te].sum()),
                   landmark_stays=int((f_te & lm).sum()), landmark_events=int(ey[f_te & lm].sum())))
EG.sort(key=lambda r: ({"bedsize": 0, "region": 1, "teaching": 2}[r["family"]], r["name"]))   # the order of Table S2
check(f"eICU-CRD: {len(EG)} consortia", len(EG) == 10)
pd.DataFrame(EG).to_csv(f"{OUT}/groups_eicu.csv", index=False)
json.dump(dict(flow=flow, checks=[dict(check=a, ok=b) for a, b in checks]), open(f"{OUT}/flow.json", "w"), indent=1)

# ------------------------------------------------------------------ tables for the Supplementary Appendix
F, Em = flow["mimic"], flow["eicu"]
def line(label, r, events=True):
    return f"| {label} | {n(r['units'])} | {n(r['patients'])} | {r['hospitals']:,} | " + (f"{npct(r['events'], r['units'])} | {n(r['units'] - r['events'])} |" if events else "| |")
def excl(label, k): return f"| {label} | {n(k)} | | | | |"
T = ["| Step | Admissions or stays | Patients | Hospitals | With the outcome, n (%) | Without the outcome |", "|---|---|---|---|---|---|",
     "| **MIMIC-IV, readmission within 30 days of discharge** | | | | | |",
     line("Admissions in the hospital module", F["source"], events=False),
     excl("Excluded, died in hospital", F["excluded_died_in_hospital"]),
     excl("Excluded, died within 30 days of discharge without readmission", F["excluded_died_30d_no_readmission"]),
     excl("Excluded, discharge time at or before admission time", F["excluded_discharge_not_after_admission"]),
     line("Analysed", F["analysed"]), line("Training patients (80 percent of patients)", F["training"]), line("Held-out patients (20 percent of patients)", F["held_out"]),
     "| **eICU-CRD, death in hospital** | | | | | |",
     line("Unit stays in the patient table", Em["source"], events=False),
     excl("Excluded, hospital discharge status not recorded as alive or expired", Em["excluded_discharge_status_not_recorded"]),
     line("Analysed", Em["analysed"]), line("Training patients (80 percent of patients)", Em["training"]),
     line("Of these, validation patients (10 percent of training patients), used for early stopping", Em["validation"]),
     line("Of these, stays used to fit models", Em["fit"]), line("Held-out patients (20 percent of patients)", Em["held_out"]),
     line("Of these, stays still in the unit at 24 hours, the primary evaluation cohort", Em["held_out_landmark"])]
open(f"{OUT}/table_flow.md", "w").write("\n".join(T) + "\n")
T = ["| Group | Training patients | Training admissions | Readmissions in training admissions, n (%) | Held-out patients | Held-out admissions | Readmissions in held-out admissions, n (%) |", "|---|---|---|---|---|---|---|"]
T += [f"| {r['group']} | {n(r['train_patients'])} | {n(r['train_admissions'])} | {npct(r['train_events'], r['train_admissions'])} | {n(r['heldout_patients'])} | {n(r['heldout_admissions'])} | {npct(r['heldout_events'], r['heldout_admissions'])} |" for r in G]
open(f"{OUT}/table_groups_mimic.md", "w").write("\n".join(T) + "\n")
T = ["| Consortium | Hospitals | Training stays | Deaths in training stays, n (%) | Held-out stays | Deaths in held-out stays, n (%) | Held-out stays in the unit at 24 hours | Deaths among them, n (%) |", "|---|---|---|---|---|---|---|---|"]
T += [f"| {r['group']} | {r['hospitals']:,} | {n(r['train_stays'])} | {npct(r['train_events'], r['train_stays'])} | {n(r['heldout_stays'])} | {npct(r['heldout_events'], r['heldout_stays'])} | {n(r['landmark_stays'])} | {npct(r['landmark_events'], r['landmark_stays'])} |" for r in EG]
open(f"{OUT}/table_groups_eicu.md", "w").write("\n".join(T) + "\n")
small = sum(0 < v < 10 for r in G for k, v in r.items() if isinstance(v, int)) + sum(0 < v < 10 for r in EG for k, v in r.items() if isinstance(v, int) and k != "hospitals")
print(f"cells below 10 in the group tables: {small}; all checks passed: {all(b for _, b in checks)}; results in {OUT}")
