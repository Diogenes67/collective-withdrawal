"""
Whole-hospital withdrawal on eICU-CRD v2.0: a naturally occurring institutional partition.

Cohort: one row per ICU stay; label = in-hospital death. Features: demographics, unit type and admit
source, admission height/weight, APACHE day-one physiology (apacheApsVar), top-50 admission diagnoses,
top-100 diagnosis strings recorded in the first 24 h. Hospital id, region and ethnicity are metadata.
Forget set: all training stays of one hospital. Control: random training stays matched on stay count.
Evaluation: the hospital's own held-out patients, for original, exact retrain, fine-tune, NegGrad+,
and exact retrain after random removal. Three seeds. Paired contrast across hospitals.

Usage: python exp_eicu.py [--quick]
Input:  ~/mimic/eicu/{patient,hospital,apacheApsVar,diagnosis}.csv.gz
Output: ~/mimic/results/eicu_<stamp>/  cohort_summary.txt, results.csv, summary.csv, paired.json, figure.png
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUICK = "--quick" in sys.argv
HOME = os.path.expanduser("~/mimic")
EICU = f"{HOME}/eicu"
STAMP = time.strftime("%Y%m%d_%H%M%S") + ("_quick" if QUICK else "")
RUN = f"{HOME}/results/eicu_{STAMP}"
CACHE = f"{HOME}/results/model_cache_eicu" + ("_quick" if QUICK else "")
os.makedirs(RUN, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
CFG = dict(seeds=[0] if QUICK else [0, 1, 2], min_hospital_stays=1000, max_hospitals=3 if QUICK else 80,
           epochs=12, unlearn_epochs=2, lr=1e-3, unlearn_lr=1e-4, neggrad_alpha=0.05, neggrad_cap=0.6,
           batch=2048, n_boot=200, eps=0.01)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
json.dump(CFG, open(f"{RUN}/config.json", "w"), indent=2)

# ============================================================== cohort
cohort_path = f"{HOME}/derived/eicu_cohort.parquet"
if os.path.exists(cohort_path):
    df = pd.read_parquet(cohort_path); log(f"cohort loaded {len(df)} stays")
else:
    log("building eICU cohort")
    pat = pd.read_csv(f"{EICU}/patient.csv.gz")
    hosp = pd.read_csv(f"{EICU}/hospital.csv.gz")
    aps = pd.read_csv(f"{EICU}/apacheApsVar.csv.gz")
    dx = pd.read_csv(f"{EICU}/diagnosis.csv.gz", usecols=["patientunitstayid", "diagnosisstring", "diagnosisoffset"])
    pat = pat[pat.hospitaldischargestatus.isin(["Alive", "Expired"])].copy()
    pat["died"] = (pat.hospitaldischargestatus == "Expired").astype(np.float32)
    pat["age_num"] = pd.to_numeric(pat.age.replace("> 89", "90"), errors="coerce")
    pat["female"] = (pat.gender == "Female").astype(float)
    for c in ["admissionheight", "admissionweight"]:
        pat[c] = pd.to_numeric(pat[c], errors="coerce")
        pat.loc[(pat[c] <= 0) | (pat[c] > 400), c] = np.nan
    pat = pat.merge(hosp[["hospitalid", "region", "numbedscategory", "teachingstatus"]], on="hospitalid", how="left")
    # categorical one-hots
    cat = pd.get_dummies(pat[["unittype", "unitadmitsource", "unitstaytype", "hospitaladmitsource"]].fillna("NA"),
                         prefix=["unit", "src", "stay", "hsrc"]).astype(float)
    topdx = pat.apacheadmissiondx.fillna("NA").value_counts().head(50).index
    adx = pd.get_dummies(pat.apacheadmissiondx.where(pat.apacheadmissiondx.isin(topdx), "other").fillna("NA"), prefix="adx").astype(float)
    # APACHE physiology: -1 means missing
    aps = aps.set_index("patientunitstayid").drop(columns=[c for c in ["apacheapsvarid"] if c in aps.columns])
    aps = aps.replace(-1, np.nan)
    aps.columns = ["aps_" + c for c in aps.columns]
    # diagnosis strings in first 24 h, top 100
    dx = dx[(dx.diagnosisoffset >= -1440) & (dx.diagnosisoffset <= 1440)]
    dx["d"] = dx.diagnosisstring.str.split("|").str[-1].str.strip().str.lower()
    top = dx.d.value_counts().head(100).index
    dxw = (dx[dx.d.isin(top)].assign(v=1.0).pivot_table(index="patientunitstayid", columns="d", values="v", aggfunc="max", fill_value=0.0))
    dxw.columns = ["dx_" + c[:40] for c in dxw.columns]
    df = pd.concat([pat.set_index("patientunitstayid")[["uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus",
                                                         "died", "age_num", "female", "admissionheight", "admissionweight", "unitvisitnumber"]],
                    cat.set_index(pat.patientunitstayid), adx.set_index(pat.patientunitstayid)], axis=1)
    df = df.join(aps, how="left").join(dxw, how="left")
    df[dxw.columns] = df[dxw.columns].fillna(0.0)
    df = df.reset_index().rename(columns={"index": "patientunitstayid"})
    os.makedirs(f"{HOME}/derived", exist_ok=True); df.to_parquet(cohort_path, index=False)
    log(f"cohort built: {len(df)} stays, {df.uniquepid.nunique()} patients, {df.hospitalid.nunique()} hospitals, mortality {df.died.mean():.3f}")

META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
feat = [c for c in df.columns if c not in META]
Xdf = df[feat].astype(float)
miss = Xdf.columns[Xdf.isna().any()]
for c in miss:
    Xdf[c + "_missing"] = Xdf[c].isna().astype(float); Xdf[c] = Xdf[c].fillna(Xdf[c].median())
X = Xdf.values.astype(np.float32); y = df.died.values.astype(np.float32)
pid = df.uniquepid.values; hosp_id = df.hospitalid.values
rng = np.random.RandomState(0)
ups = df.uniquepid.unique(); rng.shuffle(ups)
test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6
X = (X - mu) / sd
with open(f"{RUN}/cohort_summary.txt", "w") as f:
    f.write(f"stays {len(df)}\npatients {df.uniquepid.nunique()}\nhospitals {df.hospitalid.nunique()}\nmortality {df.died.mean():.4f}\n"
            f"features {X.shape[1]}\ntrain {is_train.sum()} test {is_test.sum()}\n")
log(f"{len(df)} stays, {X.shape[1]} features, mortality {y.mean():.3f}, train {is_train.sum()}, test {is_test.sum()}")

# target hospitals
hs = pd.Series(hosp_id[is_train]).value_counts()
targets_h = hs[hs >= CFG["min_hospital_stays"] * 0.8].sort_values().index.tolist()
targets_h = [h for h in targets_h if (is_test & (hosp_id == h)).sum() >= 100][: CFG["max_hospitals"]]
log(f"{len(targets_h)} target hospitals with >= {int(CFG['min_hospital_stays']*0.8)} training stays and >= 100 test stays")

# ============================================================== model
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def train(Xtr, ytr, seed, epochs, lr, init=None, forget=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]).to(DEVICE)
    if init is not None: m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=lr); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(Xtr), to_t(ytr)
    if forget is not None: Xf, yf = to_t(forget[0]), to_t(forget[1])
    n, B = len(Xtr), CFG["batch"]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; loss = bce(m(Xt[idx]), yt[idx])
            if forget is not None:
                j = torch.randint(0, len(Xf), (min(B, len(Xf)),), device=DEVICE)
                loss = loss - CFG["neggrad_alpha"] * torch.clamp(bce(m(Xf[j]), yf[j]), max=CFG["neggrad_cap"])
            opt.zero_grad(); loss.backward(); opt.step()
    return m
def cached(key, fn):
    path = f"{CACHE}/{key}.pt"; m = MLP(X.shape[1]).to(DEVICE)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True)); return m
    m = fn(); torch.save(m.state_dict(), path); return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def ece(yy, pp, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pp > lo) & (pp <= hi)
        if mk.any(): e += mk.mean() * abs(yy[mk].mean() - pp[mk].mean())
    return e
def auc(yy, pp): return roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan
def boot_patient(fn, yy, pp, pp_id):
    r = np.random.RandomState(1); u = np.unique(pp_id); groups = {p: np.where(pp_id == p)[0] for p in u}; vals = []
    for _ in range(CFG["n_boot"]):
        i = np.concatenate([groups[p] for p in r.choice(u, len(u), True)]); vals.append(fn(yy[i], pp[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

# ============================================================== run
Xte, yte = X[is_test], y[is_test]
originals = {s: cached(f"orig_s{s}", lambda s=s: train(X[is_train], y[is_train], s, CFG["epochs"], CFG["lr"])) for s in CFG["seeds"]}
for s, m in originals.items(): log(f"original seed {s}: test AUROC {auc(yte, predict(m, Xte)):.4f}")

rows = []; r_ctrl = np.random.RandomState(7)
for h in targets_h:
    f_tr = is_train & (hosp_id == h); te = np.where(is_test & (hosp_id == h))[0]
    # stay-matched random control from other hospitals' training stays
    pool = np.where(is_train & (hosp_id != h))[0]
    ctrl = np.zeros(len(X), bool); ctrl[r_ctrl.choice(pool, int(f_tr.sum()), False)] = True
    yy, Xh, pid_h = y[te], X[te], pid[te]
    region = df.region.iloc[te[0]] if len(te) else ""
    for s in CFG["seeds"]:
        st = originals[s].state_dict(); r_tr = is_train & ~f_tr; r_ctrl_tr = is_train & ~ctrl
        models = {"original": originals[s],
                  "exact_hospital": cached(f"exact_h{h}_s{s}", lambda: train(X[r_tr], y[r_tr], s, CFG["epochs"], CFG["lr"])),
                  "finetune_hospital": cached(f"ft_h{h}_s{s}", lambda: train(X[r_tr], y[r_tr], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st)),
                  "neggrad_hospital": cached(f"ng_h{h}_s{s}", lambda: train(X[r_tr], y[r_tr], s, CFG["unlearn_epochs"], CFG["unlearn_lr"], init=st, forget=(X[f_tr], y[f_tr]))),
                  "exact_random": cached(f"exact_r{h}_s{s}", lambda: train(X[r_ctrl_tr], y[r_ctrl_tr], s, CFG["epochs"], CFG["lr"]))}
        p0 = predict(models["original"], Xh); a0, e0 = auc(yy, p0), ece(yy, p0)
        for name, m in models.items():
            p = predict(m, Xh); a, e = auc(yy, p), ece(yy, p); alo, ahi = boot_patient(auc, yy, p, pid_h)
            rows.append(dict(hospital=int(h), region=region, seed=s, model=name, forget_stays=int(f_tr.sum()), test_stays=len(te),
                             test_patients=len(np.unique(pid_h)), mortality_hospital=float(y[f_tr].mean()), mortality_control=float(y[ctrl].mean()),
                             test_auroc=auc(yte, predict(m, Xte)), auroc=a, auroc_lo=alo, auroc_hi=ahi, ece=e, d_auroc=a - a0, d_ece=e - e0))
    pd.DataFrame(rows).to_csv(f"{RUN}/results.csv", index=False)
    log(f"hospital {h}: done ({int(f_tr.sum())} stays)")

R = pd.DataFrame(rows)
S = R.groupby(["hospital", "model"]).mean(numeric_only=True).reset_index()
S.to_csv(f"{RUN}/summary.csv", index=False)
W = S.pivot(index="hospital", columns="model", values="d_auroc"); WE = S.pivot(index="hospital", columns="model", values="d_ece")
def paired(a, b):
    d = a.values - b.values; r = np.random.RandomState(0)
    bs = [r.choice(d, len(d), True).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d < 0).mean()
out = {}
for k, (a, b) in {"exact_hospital_vs_exact_random": ("exact_hospital", "exact_random"),
                  "finetune_vs_exact_random": ("finetune_hospital", "exact_random"),
                  "neggrad_vs_exact_random": ("neggrad_hospital", "exact_random")}.items():
    m_, lo, hi, fneg = paired(W[a], W[b]); me, elo, ehi, _ = paired(WE[a], WE[b])
    out[k] = dict(d_auroc=m_, ci=(lo, hi), frac_hospital_worse=fneg, d_ece=me, ece_ci=(elo, ehi))
json.dump(out, open(f"{RUN}/paired.json", "w"), indent=2, default=float)
print("\nchange vs original on the hospital's own held-out patients, mean over hospitals:")
print(W.mean().round(4).to_string()); print(); print(WE.mean().round(4).to_string())
print("\npaired contrasts (hospital removal minus stay-matched random removal, same test patients):")
for k, v in out.items():
    print(f"  {k}: dAUROC {v['d_auroc']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] hospital worse {100*v['frac_hospital_worse']:.0f}%   dECE {v['d_ece']:+.4f} [{v['ece_ci'][0]:+.4f}, {v['ece_ci'][1]:+.4f}]")
print(f"\nhospitals {len(W)}, forget stays {S[S.model=='original'].forget_stays.min():.0f} to {S[S.model=='original'].forget_stays.max():.0f}, eps-bound breached (exact) in {100*(WE['exact_hospital'] > CFG['eps']).mean():.0f}% vs random {100*(WE['exact_random'] > CFG['eps']).mean():.0f}%")

sz = S[S.model == "original"].set_index("hospital").forget_stays; o = sz.sort_values().index; xs = np.arange(len(o))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for col, lab, mk in [("exact_hospital", "remove the hospital", "o"), ("exact_random", "remove stay-matched random stays", "s"), ("finetune_hospital", "fine-tune without the hospital", "^")]:
    ax[0].plot(xs, W.loc[o, col], mk, label=lab, alpha=0.75); ax[1].plot(xs, WE.loc[o, col], mk, label=lab, alpha=0.75)
for a, ttl in zip(ax, ["Change in AUROC on the hospital's held-out patients", "Change in ECE on the hospital's held-out patients"]):
    a.axhline(0, ls=":", color="grey"); a.set_xlabel("hospitals, ordered by training stays removed"); a.set_title(ttl); a.set_xticks([])
ax[1].axhline(CFG["eps"], ls=":", color="red"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RUN}/figure.png", dpi=150)
log(f"done, run folder {RUN}")
