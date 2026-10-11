"""
Build a 30-day readmission cohort from MIMIC-IV v3.1 hosp module.

Usage:
    python extract_cohort.py            # admissions + patients + diagnoses only
    python extract_cohort.py --labs     # also pulls discharge labs from labevents.csv.gz

Input:  ~/mimic/hosp/*.csv.gz
Output: ~/mimic/derived/cohort_<timestamp>.parquet  and  cohort_latest.parquet
Nothing patient-level leaves this machine. Only aggregate metrics get shared.
"""

import os, sys, time
import numpy as np
import pandas as pd

HOSP = os.path.expanduser("~/mimic/hosp")
OUT = os.path.expanduser("~/mimic/derived")
os.makedirs(OUT, exist_ok=True)
WITH_LABS = "--labs" in sys.argv
t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ------------------------------------------------------------------ load
log("loading admissions, patients")
adm = pd.read_csv(f"{HOSP}/admissions.csv.gz",
                  parse_dates=["admittime", "dischtime", "edregtime"])
pat = pd.read_csv(f"{HOSP}/patients.csv.gz", parse_dates=["dod"])
df = adm.merge(pat, on="subject_id", how="inner")
log(f"{len(df)} admissions, {df.subject_id.nunique()} patients")

# ------------------------------------------------------------------ label
# next admission for same patient; readmit30 = next admittime within 30 days of discharge
df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
df["next_admit"] = df.groupby("subject_id")["admittime"].shift(-1)
df["days_to_next"] = (df["next_admit"] - df["dischtime"]).dt.total_seconds() / 86400
df["readmit30"] = ((df["days_to_next"] >= 0) & (df["days_to_next"] <= 30)).astype(int)

# exclusions: died in hospital; died within 30 d of discharge without readmission (competing risk)
died_inhosp = df["hospital_expire_flag"] == 1
died_30 = (df["dod"].notna()
           & ((df["dod"] - df["dischtime"]).dt.total_seconds() / 86400 <= 30)
           & (df["readmit30"] == 0))
bad_los = df["dischtime"] <= df["admittime"]
df = df[~(died_inhosp | died_30 | bad_los)].copy()
log(f"after exclusions: {len(df)} admissions, readmit30 rate {df.readmit30.mean():.3f}")

# ------------------------------------------------------------------ features
df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
df["via_ed"] = df["edregtime"].notna().astype(int)
df["female"] = (df["gender"] == "F").astype(int)
df["english"] = (df["language"] == "English").astype(int)

# prior admissions in previous 365 days and lifetime (uses all admissions, pre-exclusion)
adm_s = adm.sort_values(["subject_id", "admittime"])
adm_s["n_prior_total"] = adm_s.groupby("subject_id").cumcount()
prior365 = []
for sid, g in adm_s.groupby("subject_id"):
    t = g["admittime"].values
    # count admissions with admittime in (t_i - 365d, t_i)
    left = np.searchsorted(t, t - np.timedelta64(365, "D"), side="right")
    prior365.append(pd.Series(np.arange(len(t)) - left, index=g.index))
adm_s["n_prior_365"] = pd.concat(prior365)
df = df.merge(adm_s[["hadm_id", "n_prior_total", "n_prior_365"]], on="hadm_id", how="left")

# coarsen race to a handful of groups (MIMIC has ~30 labels)
def race_group(r):
    r = str(r).upper()
    if r.startswith("WHITE"): return "white"
    if r.startswith("BLACK"): return "black"
    if r.startswith("HISPANIC"): return "hispanic"
    if r.startswith("ASIAN"): return "asian"
    if "AMERICAN INDIAN" in r or "NATIVE" in r: return "native"
    if r in ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"): return "unknown"
    return "other"
df["race_group"] = df["race"].map(race_group)

# ------------------------------------------------------------------ diagnoses
log("loading diagnoses")
dx = pd.read_csv(f"{HOSP}/diagnoses_icd.csv.gz", dtype={"icd_code": str})
dx = dx[dx["hadm_id"].isin(df["hadm_id"])]
dx["code3"] = dx["icd_version"].astype(str) + "_" + dx["icd_code"].str[:3]

n_dx = dx.groupby("hadm_id").size().rename("n_dx")
df = df.merge(n_dx, on="hadm_id", how="left")
df["n_dx"] = df["n_dx"].fillna(0)

# Charlson-style comorbidity flags by ICD prefix (ICD-9 and ICD-10). Approximate, documented as such.
CHARLSON = {
    "mi":        (["410", "412"],                     ["I21", "I22", "I252"]),
    "chf":       (["428"],                            ["I50"]),
    "pvd":       (["4439", "441"],                    ["I70", "I71", "I73"]),
    "cva":       (["430", "431", "432", "433", "434", "435", "436", "437", "438"], ["I6", "G45"]),
    "dementia":  (["290"],                            ["F00", "F01", "F02", "F03", "G30"]),
    "copd":      (["490", "491", "492", "493", "494", "495", "496"], ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"]),
    "rheum":     (["7100", "7101", "7104", "714"],    ["M05", "M06", "M32", "M33", "M34"]),
    "pud":       (["531", "532", "533", "534"],       ["K25", "K26", "K27", "K28"]),
    "liver":     (["5712", "5714", "5715", "5716", "5718", "5719"], ["K70", "K71", "K73", "K74", "K76"]),
    "diabetes":  (["250"],                            ["E10", "E11", "E12", "E13", "E14"]),
    "hemiplegia":(["342", "344"],                     ["G81", "G82"]),
    "renal":     (["582", "583", "585", "586", "588"], ["N18", "N19", "N05", "Z49"]),
    "cancer":    ([f"{i}" for i in range(140, 200)],  ["C"]),
    "hiv":       (["042", "043", "044"],              ["B20", "B21", "B22", "B24"]),
}
def has_prefix(codes, prefixes):
    return codes.str.startswith(tuple(prefixes))
flags = {}
for name, (p9, p10) in CHARLSON.items():
    m9 = (dx["icd_version"] == 9) & has_prefix(dx["icd_code"], p9)
    m10 = (dx["icd_version"] == 10) & has_prefix(dx["icd_code"], p10)
    flags[f"cm_{name}"] = dx.loc[m9 | m10, "hadm_id"].unique()
for name, ids in flags.items():
    df[name] = df["hadm_id"].isin(ids).astype(int)

# top-100 3-character diagnosis codes as binary indicators
top = dx["code3"].value_counts().head(100).index
dx_top = dx[dx["code3"].isin(top)]
wide = (dx_top.assign(v=1)
        .pivot_table(index="hadm_id", columns="code3", values="v", aggfunc="max", fill_value=0))
wide.columns = ["dx_" + c for c in wide.columns]
df = df.merge(wide, on="hadm_id", how="left")
df[wide.columns] = df[wide.columns].fillna(0).astype(int)
log(f"diagnosis features: {len(flags)} comorbidity flags + {len(wide.columns)} code indicators")

# ------------------------------------------------------------------ labs (optional)
if WITH_LABS:
    import duckdb
    log("reading labevents via duckdb (this takes a few minutes)")
    ITEMS = {50912: "creatinine", 50983: "sodium", 50971: "potassium", 51222: "hemoglobin",
             51301: "wbc", 51265: "platelets", 50931: "glucose", 50882: "bicarbonate",
             51006: "bun", 50862: "albumin", 50813: "lactate", 51237: "inr"}
    item_list = ",".join(str(i) for i in ITEMS)
    q = f"""
        with x as (
            select hadm_id, itemid, valuenum, charttime,
                   row_number() over (partition by hadm_id, itemid order by charttime desc) as rn
            from read_csv_auto('{HOSP}/labevents.csv.gz')
            where hadm_id is not null and valuenum is not null and itemid in ({item_list})
        )
        select hadm_id, itemid, valuenum from x where rn = 1
    """
    labs = duckdb.sql(q).df()
    labs["name"] = "lab_" + labs["itemid"].map(ITEMS)
    labs_w = labs.pivot_table(index="hadm_id", columns="name", values="valuenum")
    df = df.merge(labs_w, on="hadm_id", how="left")
    log(f"labs merged: {labs_w.shape[1]} analytes, coverage {df[labs_w.columns].notna().mean().round(2).to_dict()}")

# ------------------------------------------------------------------ save
keep_meta = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group",
             "admission_type", "admission_location", "discharge_location", "insurance",
             "marital_status", "race_group", "readmit30"]
feat_num = ["age", "female", "english", "los_days", "via_ed", "n_prior_total", "n_prior_365", "n_dx"]
feat_cm = [c for c in df.columns if c.startswith("cm_")]
feat_dx = [c for c in df.columns if c.startswith("dx_")]
feat_lab = [c for c in df.columns if c.startswith("lab_")]
cols = keep_meta + feat_num + feat_cm + feat_dx + feat_lab
out = df[cols].copy()

stamp = time.strftime("%Y%m%d_%H%M%S")
path = f"{OUT}/cohort_{stamp}.parquet"
out.to_parquet(path, index=False)
out.to_parquet(f"{OUT}/cohort_latest.parquet", index=False)

log(f"saved {path}")
print()
print("rows            :", len(out))
print("patients        :", out.subject_id.nunique())
print("readmit30 rate  :", round(out.readmit30.mean(), 4))
print("features        :", len(feat_num + feat_cm + feat_dx + feat_lab),
      f"({len(feat_num)} num, {len(feat_cm)} comorb, {len(feat_dx)} dx, {len(feat_lab)} lab)")
print("race_group      :", out.race_group.value_counts().to_dict())
print("insurance       :", out.insurance.value_counts().to_dict())
