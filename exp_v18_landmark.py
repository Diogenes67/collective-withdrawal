"""exp_v18_landmark.py. eICU contrasts restricted to the 24-hour landmark population, plus matched-flag contrasts against random
removal (review points 3 and 4). No models are loaded: everything comes from the v13.1 cached predictions on the held-out set.

The eICU cohort includes every stay, with features from the first 24 hours (day-one APACHE physiology, diagnoses recorded
within 24 hours). Stays that ended within 24 hours are therefore scored with information that may postdate death or discharge.
POP=landmark keeps held-out stays whose ICU stay lasted at least 24 hours (unitdischargeoffset >= 1440 minutes), so the
patient was alive and in the unit at the prediction time; POP=all keeps every held-out stay (reproduces v13/v13b as a check).
Models were trained on all stays in both cases; this is an evaluation restriction.

Hospital-cluster bootstrap exactly as v13 (one hospital draw feeds all ten consortia; AUROC and decision metrics are means over
seeds, contrasts are means over consortia). Matched flag rate as v13b: k = number of the consortium's held-out stays the reference
model (the centralised original, or the full federation for federated arms) flags at its system-wide top-decile threshold;
every arm flags its own top k; events unflagged per 1,000 stays.

Run in chunks to stay inside short shells: POP=landmark CHUNK=0..4 (200 draws each), then POP=landmark CHUNK=combine.
Writes results/v18_landmark/{POP}_chunk{k}.npz and {POP}_contrasts.json.
"""
import os, json, numpy as np, pandas as pd
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); IN = f"{ROOT}/results/v13_intervals_1"; OUT = f"{ROOT}/results/v18_landmark"
os.makedirs(OUT, exist_ok=True)
POP = os.environ.get("POP", "landmark"); CHUNK = os.environ.get("CHUNK", "combine"); NPER = 200; SEEDS = [0, 1, 2]
ARMS = ["original", "exact_group", "exact_random", "fl_global", "fl_without", "fl_random", "fl_resid", "fl_resid_random"]
AUC_PAIRS = {"exact_group minus exact_random (centralised penalty)": ("exact_group", "exact_random"),
             "exact_group minus original": ("exact_group", "original"),
             "fl_without minus fl_random (federated penalty)": ("fl_without", "fl_random"),
             "fl_without minus fl_global": ("fl_without", "fl_global"),
             "fl_resid minus fl_without (recovered)": ("fl_resid", "fl_without"),
             "fl_resid minus fl_global": ("fl_resid", "fl_global"),
             "fl_resid_random minus fl_without (control)": ("fl_resid_random", "fl_without"),
             "fl_resid minus fl_resid_random (consortium-specific)": ("fl_resid", "fl_resid_random")}
MF_PAIRS = {"exact_group minus original": ("exact_group", "original"), "exact_random minus original": ("exact_random", "original"),
            "exact_group minus exact_random": ("exact_group", "exact_random"), "fl_without minus fl_global": ("fl_without", "fl_global"),
            "fl_without minus fl_random": ("fl_without", "fl_random"), "fl_resid minus fl_global": ("fl_resid", "fl_global"),
            "fl_resid minus fl_resid_random": ("fl_resid", "fl_resid_random")}
REF = {a: ("fl_global" if a.startswith("fl_") else "original") for a in ARMS}

def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["patientunitstayid", "uniquepid", "hospitalid", "died"])
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values
y_all = df.died.values[is_test].astype(int); h_all = df.hospitalid.values[is_test]; stay = df.patientunitstayid.values[is_test]
P = np.load(f"{IN}/preds_eicu.npz", allow_pickle=True)["P"].item(); assert len(P[next(iter(P))]) == len(y_all)
mem = json.load(open(f"{IN}/membership_eicu.json")); keys = list(mem)
# system-wide top-decile thresholds on the full held-out set, per consortium key, seed and reference model (as v13b)
thr = {(k, s, a): float(np.percentile(P[(k, s, a)], 90)) for k in keys for s in SEEDS for a in ["original", "fl_global"]}
if POP == "landmark":
    pt = pd.read_csv(f"{ROOT}/eicu/patient.csv.gz", usecols=["patientunitstayid", "unitdischargeoffset"]).set_index("patientunitstayid").unitdischargeoffset
    keep = pt.reindex(stay).values >= 1440
else:
    keep = np.ones(len(y_all), bool)
rows_all = np.where(keep)[0]; y_te = y_all; h_te = h_all
hospitals = np.unique(h_te[rows_all]); h_rows = {h: rows_all[h_te[rows_all] == h] for h in hospitals}
def stats(rows, hs):
    out = {}
    for k in keys:
        rr = rows[np.isin(hs, mem[k])]
        if len(rr) < 50 or len(np.unique(y_te[rr])) < 2: continue
        yy = y_te[rr]; o = {}
        for a in ARMS:
            aucs, miss = [], []
            for s in SEEDS:
                p = P[(k, s, a)][rr]; aucs.append(auc(yy, p))
                kf = int((P[(k, s, REF[a])][rr] >= thr[(k, s, REF[a])]).sum()); top = np.argsort(-p, kind="mergesort")[:kf]
                miss.append(1000.0 * (yy.sum() - yy[top].sum()) / len(yy))
            o[a] = (np.nanmean(aucs), np.mean(miss))
        out[k] = o
    return out
def contrasts(st):
    d = {f"AUROC | {n}": np.mean([o[a][0] - o[b][0] for o in st.values()]) for n, (a, b) in AUC_PAIRS.items()}
    d.update({f"events unflagged per 1,000 at matched flag rate | {n}": np.mean([o[a][1] - o[b][1] for o in st.values()]) for n, (a, b) in MF_PAIRS.items()})
    return d
if CHUNK == "combine":
    pt_ = contrasts(stats(rows_all, h_te[rows_all])); B = {k: [] for k in pt_}
    for c in range(5):
        Z = np.load(f"{OUT}/{POP}_chunk{c}.npz", allow_pickle=True)["B"].item()
        for k in pt_: B[k] += Z[k]
    base = dict(population=POP, held_out_stays=int(keep.sum()), of_total=int(len(keep)), mortality=float(y_te[rows_all].mean()), hospitals=int(len(hospitals)))
    res = {k: dict(d=float(pt_[k]), ci=[float(np.percentile(B[k], 2.5)), float(np.percentile(B[k], 97.5))], n_boot=len(B[k])) for k in pt_}
    json.dump(dict(population=base, results=res), open(f"{OUT}/{POP}_contrasts.json", "w"), indent=1)
    print(base)
    for k, v in res.items(): print(f"{k:105s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
else:
    c = int(CHUNK); r = np.random.RandomState(2000 + c + (100 if POP == "landmark" else 0)); B = None
    for b in range(NPER):
        drawn = r.choice(hospitals, len(hospitals), True); rows = np.concatenate([h_rows[h] for h in drawn])
        d = contrasts(stats(rows, h_te[rows]))
        if B is None: B = {k: [] for k in d}
        for k, v in d.items(): B[k].append(v)
    np.savez_compressed(f"{OUT}/{POP}_chunk{c}.npz", B=np.array(B, dtype=object)); print(POP, "chunk", c, "done")
