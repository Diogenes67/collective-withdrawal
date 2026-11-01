"""
exp_v41_baseline.py. Baseline characteristics by dataset and split (TRIPOD+AI item 20b), aggregate only.
MIMIC-IV: all admissions, training and held-out patients (the v5 patient split, RandomState(0), 20 percent held out).
eICU-CRD: all stays, held-out stays (the v7 patient split, RandomState(0), 20 percent) and held-out stays still in the unit at
24 hours (unitdischargeoffset >= 1440 minutes, the primary evaluation cohort).
Reads derived/cohort_latest.parquet, derived/eicu_cohort.parquet, eicu/patient.csv.gz. Writes results/v41_baseline/{table_baseline.md,
table_baseline.csv}. No cell below 10 is reported (shown as <10).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); OUT = f"{ROOT}/results/v41_baseline"; os.makedirs(OUT, exist_ok=True)
rows = []
def add(section, label, cols): rows.append(dict(section=section, characteristic=label, **cols))
def pct(mask): n = int(mask.sum()); return "<10" if 0 < n < 10 else f"{n:,} ({100 * mask.mean():.1f})"
def mean_sd(x): x = pd.Series(x).dropna(); return f"{x.mean():.1f} ({x.std():.1f})"
def med_iqr(x): x = pd.Series(x).dropna(); return f"{x.median():.1f} ({x.quantile(.25):.1f} to {x.quantile(.75):.1f})"

# ------------------------------------------------------------------ MIMIC-IV
m = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet")
rng = np.random.RandomState(0); pids = m.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
te = m.subject_id.isin(test_p).values
M = {"MIMIC-IV, all": np.ones(len(m), bool), "MIMIC-IV, training": ~te, "MIMIC-IV, held out": te}
def mcol(f): return {k: f(m[v]) for k, v in M.items()}
add("MIMIC-IV", "Admissions", mcol(lambda d: f"{len(d):,}"))
add("MIMIC-IV", "Patients", mcol(lambda d: f"{d.subject_id.nunique():,}"))
add("MIMIC-IV", "Age, years, mean (SD)", mcol(lambda d: mean_sd(d.age)))
add("MIMIC-IV", "Female, n (%)", mcol(lambda d: pct(d.female == 1)))
add("MIMIC-IV", "English speaking, n (%)", mcol(lambda d: pct(d.english == 1)))
add("MIMIC-IV", "Arrived through the emergency department, n (%)", mcol(lambda d: pct(d.via_ed == 1)))
add("MIMIC-IV", "Length of stay, days, median (IQR)", mcol(lambda d: med_iqr(d.los_days)))
add("MIMIC-IV", "Admissions in the prior 365 days, mean (SD)", mcol(lambda d: mean_sd(d.n_prior_365)))
for c, lab in [("cm_chf", "Congestive heart failure"), ("cm_diabetes", "Diabetes"), ("cm_renal", "Renal disease"), ("cm_cancer", "Cancer"), ("cm_liver", "Liver disease")]:
    add("MIMIC-IV", f"{lab}, n (%)", mcol(lambda d, c=c: pct(d[c] == 1)))
add("MIMIC-IV", "Readmitted within 30 days, n (%)", mcol(lambda d: pct(d.readmit30 == 1)))
for g in m.race_group.value_counts().index:
    add("MIMIC-IV", f"Race (record field): {g}, n (%)", mcol(lambda d, g=g: pct(d.race_group == g)))
for g in m.insurance.fillna("Unknown").value_counts().index:
    add("MIMIC-IV", f"Insurance: {g}, n (%)", mcol(lambda d, g=g: pct(d.insurance.fillna("Unknown") == g)))
for g in sorted(m.anchor_year_group.unique()):
    add("MIMIC-IV", f"Anchor year group: {g}, n (%)", mcol(lambda d, g=g: pct(d.anchor_year_group == g)))

# ------------------------------------------------------------------ eICU-CRD
e = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
rng = np.random.RandomState(0); ups = e.uniquepid.unique(); rng.shuffle(ups); tp = set(ups[: int(0.2 * len(ups))])
ete = e.uniquepid.isin(tp).values
off = pd.read_csv(f"{ROOT}/eicu/patient.csv.gz", usecols=["patientunitstayid", "unitdischargeoffset"]).set_index("patientunitstayid").unitdischargeoffset
lm = (off.reindex(e.patientunitstayid).values >= 1440)
E = {"eICU-CRD, all stays": np.ones(len(e), bool), "eICU-CRD, held out": ete, "eICU-CRD, held out, in the unit at 24 hours": ete & lm}
def ecol(f): return {k: f(e[v]) for k, v in E.items()}
unit_cols = [c for c in e.columns if c.startswith("unit_")]
add("eICU-CRD", "Stays", ecol(lambda d: f"{len(d):,}"))
add("eICU-CRD", "Patients", ecol(lambda d: f"{d.uniquepid.nunique():,}"))
add("eICU-CRD", "Hospitals", ecol(lambda d: f"{d.hospitalid.nunique():,}"))
add("eICU-CRD", "Age, years, mean (SD)", ecol(lambda d: mean_sd(d.age_num)))
add("eICU-CRD", "Female, n (%)", ecol(lambda d: pct(d.female == 1)))
if "src_Emergency Department" in e: add("eICU-CRD", "Admitted from the emergency department, n (%)", ecol(lambda d: pct(d["src_Emergency Department"] == 1)))
if "aps_vent" in e: add("eICU-CRD", "Ventilated on day one (APACHE), n (%)", ecol(lambda d: pct(d.aps_vent == 1)))
for c in unit_cols:
    add("eICU-CRD", f"Unit type: {c[5:]}, n (%)", ecol(lambda d, c=c: pct(d[c] == 1)))
add("eICU-CRD", "Died in hospital, n (%)", ecol(lambda d: pct(d.died == 1)))
for col, lab in [("region", "Region"), ("teachingstatus", "Teaching hospital"), ("numbedscategory", "Bed-size band"), ("ethnicity", "Ethnicity (record field)")]:
    vals = e[col].astype(str).replace({"nan": "Unknown", "": "Unknown"})
    for g in vals.value_counts().index:
        add("eICU-CRD", f"{lab}: {g}, n (%)", {k: pct(vals[v] == g) for k, v in E.items()})

T = pd.DataFrame(rows).fillna("")
T.to_csv(f"{OUT}/table_baseline.csv", index=False)
L = []
for sec, cols in [("MIMIC-IV", list(M)), ("eICU-CRD", list(E))]:
    sub = T[T.section == sec]
    L += [f"**{sec}**", "", "| Characteristic | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += ["| " + r.characteristic + " | " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in sub.iterrows()]
    L += [""]
open(f"{OUT}/table_baseline.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
