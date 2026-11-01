"""
exp_v42_strict_penalty.py. The withdrawal penalty under complete-pipeline retraining (external review, point 2): group withdrawal
and matched random removal both retrained with the whole shared pipeline (vocabulary, imputation, scaling and model) refitted on
the retained training rows. The group arm is the strict base already trained (v16 on MIMIC-IV, v23 on eICU) and is read from
their prediction caches; this script trains the matched random-removal arm on the same strict recipe and scores both on the
group's held-out patients.

MIMIC-IV: 25 clustered groups x seeds 0-2. Groups and patient-matched random sets exactly as v5 (cluster membership from
derived/cluster_assign_v5.csv; random set RandomState(100 + v5 group index); checked against the v5 run records). Strict random
arm: vocabulary, medians and scaling fitted on the training admissions outside the random set (build_X of v16), MLP trained 12
epochs (v5 recipe), no validation data.
eICU: 10 consortia x seeds 0-2. Stay-matched random sets exactly as v7.2 (RandomState(300 + consortium index), drawn from all
fit stays). Strict random arm: indicators, medians and scaling fitted on the training stays outside the random set (build_X of
v23), trained on the fit stays outside it with early stopping on all validation stays (as the v7.2 random arm).

Outputs per run (runs/*.json, preds/*.npz with the strict group and strict random predictions on the group's held-out rows), then
the summary: complete-pipeline penalty (strict group minus strict random), strict random minus pooled, and the published
reference-retraining penalty for comparison; MIMIC-IV two-stage bootstrap (groups, then patients), eICU hospital-cluster bootstrap
(one draw for all consortia) on all held-out stays and stays in the unit at 24 hours; 1,000 draws, RandomState(0), draws saved.
Env: ROOT, DATA=mimic|eicu|both (default both), SEEDS (default 0,1,2), DEVICE, NBOOT, SUMMARY=0, QUICK=1 (two groups, seed 0,
two epochs, 30 draws). Writes results/v42_strict_penalty/{runs, preds, summary.txt, intervals.json, draws_*.npz}. Resumable.
"""
import os, json, time, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"
OUT = f"{R}/v42_strict_penalty" + ("_quick" if QUICK else "")
for d in ["runs", "preds"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
DATA = os.environ.get("DATA", "both"); SEEDS = [0] if QUICK else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
NBOOT = int(os.environ.get("NBOOT", 30 if QUICK else 1000))
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def probs(m, Xa):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def fast_auc(yv, pv):
    n1 = yv.sum(); n0 = len(yv) - n1
    if n1 == 0 or n0 == 0: return np.nan
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1)
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
log(f"device {DEVICE}; data {DATA}; seeds {SEEDS}; out {OUT}")

# ======================================================================= MIMIC-IV
def run_mimic():
    V5 = f"{R}/v5_partitions"; V16 = f"{R}/v16_strict"; EPOCHS = 2 if QUICK else 12
    META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
            "insurance", "marital_status", "race_group", "readmit30"]
    df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
    feat_dx = [c for c in feat if c.startswith("dx_")]; feat_other = [c for c in feat if not c.startswith("dx_")]; feat_lab = [c for c in feat if c.startswith("lab_")]
    y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values; hadm = df.hadm_id.values
    rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
    is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
    # groups in the v5 order (needed for the random-set seeds)
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
    assign = pd.read_csv(f"{ROOT}/derived/cluster_assign_v5.csv").set_index("subject_id")["cluster"]
    for c in range(25): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
    groups = []
    for (fam, name), members in labels.items():
        mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
        n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
        if n_tr_p < 300 or n_te_p < 50: continue
        groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p, n_te_p=n_te_p))
    comms = []
    for i, g in enumerate(groups):
        if g["family"] != "community_trainfit": continue
        r = np.random.RandomState(100 + i); tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]])))
        g["r_pat"] = np.isin(sid, r.choice(tr_p, min(g["n_tr_p"], len(tr_p)), False)) & is_train
        ref = json.load(open(f"{V5}/runs/community_trainfit__{g['name']}__s0.json"))
        if ref["n_train_patients"] != g["n_tr_p"] or ref["n_test_patients"] != g["n_te_p"] or ref["rows_random_pat"] != int(g["r_pat"].sum()):
            raise SystemExit(f"{g['name']}: membership or random set differs from v5")
        comms.append(g)
    if QUICK: comms = comms[:2]
    log(f"MIMIC-IV: {len(comms)} clustered groups match the v5 records")
    dx = pd.read_csv(f"{ROOT}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}, usecols=["hadm_id", "icd_code", "icd_version"])
    dx = dx[dx.hadm_id.isin(set(hadm))]; dx["code3"] = dx.icd_version.astype(str) + "_" + dx.icd_code.str[:3]; dx = dx[["hadm_id", "code3"]].drop_duplicates()
    def build_X(fit_rows):   # v16 build_X
        fit_hadm = set(hadm[fit_rows]); top = dx[dx.hadm_id.isin(fit_hadm)].code3.value_counts().head(100).index
        wide = dx[dx.code3.isin(top)].assign(v=1).pivot_table(index="hadm_id", columns="code3", values="v", aggfunc="max", fill_value=0)
        wide = wide.reindex(columns=sorted(top), fill_value=0).reindex(hadm, fill_value=0)
        Xo = df[feat_other].astype(float).copy()
        for c in feat_lab:
            med = Xo.loc[fit_rows, c].median(); Xo[c + "_missing"] = Xo[c].isna().astype(float); Xo[c] = Xo[c].fillna(med)
        Xa = np.concatenate([Xo.values, wide.values.astype(float)], axis=1).astype(np.float32)
        mu, sd = Xa[fit_rows].mean(0), Xa[fit_rows].std(0); sd = np.where(sd < 1e-8, 1.0, sd)
        return ((Xa - mu) / sd).astype(np.float32)
    def train(Xtr, ytr, seed):   # v5 / v16 recipe
        torch.manual_seed(seed); np.random.seed(seed); m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
        Xt, yt = to_t(Xtr), to_t(ytr); n = len(Xtr)
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, 2048): idx = perm[i:i + 2048]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); return m
    for g in comms:
        key = f"community_trainfit__{g['name']}"; te = np.where(g["f_te"])[0]
        for s in SEEDS:
            path = f"{OUT}/runs/mimic__{key}__s{s}.json"
            if os.path.exists(path): continue
            tg = time.time(); Z16 = np.load(f"{V16}/preds/{key}__s{s}.npz")
            assert np.array_equal(Z16["pid"], sid[te]), f"{key}: v16 held-out order differs"
            fit = is_train & ~g["r_pat"]; X = build_X(fit); m = train(X[fit], y[fit], s); p_rand = probs(m, X[te]); del m, X
            if DEVICE == "mps": torch.mps.empty_cache()
            yy = y[te]; rec = dict(data="mimic", group=key, seed=s, auroc_strict_group=fast_auc(yy, Z16["strict_base"]), auroc_strict_random=fast_auc(yy, p_rand),
                                   auroc_pooled=fast_auc(yy, Z16["pooled"]), n_fit_rows=int(fit.sum()), device=DEVICE, seconds=time.time() - tg, date=time.strftime("%F %T"))
            np.savez_compressed(f"{OUT}/preds/mimic__{key}__s{s}.npz", y=yy, pid=sid[te], k_flag=int(Z16["k_flag"]), pooled=Z16["pooled"], strict_group=Z16["strict_base"], strict_random=p_rand)
            save_json(path, rec)
            log(f"{key} s{s}: strict group {rec['auroc_strict_group']:.4f} strict random {rec['auroc_strict_random']:.4f} pooled {rec['auroc_pooled']:.4f} ({rec['seconds']:.0f}s)")

# ======================================================================= eICU
def run_eicu():
    V23 = f"{R}/v23_eicu_strict"; EPOCHS = 2 if QUICK else 30
    df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet")
    META = ["patientunitstayid", "uniquepid", "hospitalid", "region", "ethnicity", "numbedscategory", "teachingstatus", "died"]
    feat = [c for c in df.columns if c not in META]; RAW = df[feat].astype(float)
    y = df.died.values.astype(np.float32); pid = df.uniquepid.values; hosp_id = df.hospitalid.values
    rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
    is_test = df.uniquepid.isin(test_p).values; is_train = ~is_test
    rv = np.random.RandomState(7); tr_p = np.array(sorted(set(ups) - test_p)); rv.shuffle(tr_p)
    is_val = df.uniquepid.isin(set(tr_p[: int(0.1 * len(tr_p))])).values & is_train; is_fit = is_train & ~is_val
    def build_X(fit_rows):   # v23 build_X
        Xd = RAW.copy(); miss = [c for c in Xd.columns if Xd.loc[fit_rows, c].isna().any()]
        for c in Xd.columns:
            if Xd[c].isna().any():
                med = Xd.loc[fit_rows, c].median(); med = 0.0 if np.isnan(med) else med
                if c in miss: Xd[c + "_missing"] = Xd[c].isna().astype(float)
                Xd[c] = Xd[c].fillna(med)
        Xa = Xd.values.astype(np.float32); mu_, sd_ = Xa[fit_rows].mean(0), Xa[fit_rows].std(0); sd_ = np.where(sd_ < 1e-8, 1.0, sd_)
        return ((Xa - mu_) / sd_).astype(np.float32)
    def train_es(Xtr, ytr, Xv, yv, seed):   # v23 recipe
        torch.manual_seed(seed); np.random.seed(seed); m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
        Xt, yt, Xvt, yvt = to_t(Xtr), to_t(ytr), to_t(Xv), to_t(yv); n = len(Xtr); best, best_state, bad, best_ep = float("inf"), None, 0, 0
        for ep in range(1, EPOCHS + 1):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, 2048): idx = perm[i:i + 2048]; loss = bce(m(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): v = float(bce(m(Xvt), yvt))
            m.train()
            if v < best - 1e-5: best, best_state, bad, best_ep = v, {k: t.detach().clone() for k, t in m.state_dict().items()}, 0, ep
            else:
                bad += 1
                if bad >= 3: break
        m.load_state_dict(best_state); m.eval(); return m, best_ep
    hattr = df.groupby("hospitalid")[["region", "numbedscategory", "teachingstatus"]].first(); groups = []
    for col, fam in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "bedsize")]:
        for v in sorted(hattr[col].dropna().unique(), key=str):
            hs = set(hattr.index[hattr[col] == v]); mem = np.isin(hosp_id, np.fromiter(hs, dtype=hosp_id.dtype)); f_tr, f_te = is_fit & mem, is_test & mem
            if f_tr.sum() < 800 or f_te.sum() < 100: continue
            groups.append(dict(family=fam, name=str(v).replace(" ", "_").replace("/", "_"), mem=mem, f_tr=f_tr, f_te=f_te))
    for i, g in enumerate(groups):   # v7.2 control, CTRL=whole
        r = np.random.RandomState(300 + i); pick = r.choice(np.where(is_fit)[0], int(g["f_tr"].sum()), False); ctrl = np.zeros(len(df), bool); ctrl[pick] = True; g["r_ctrl"] = ctrl
    if QUICK: groups = groups[:2]
    log(f"eICU: {len(groups)} consortia")
    for g in groups:
        key = f"{g['family']}_{g['name']}"; te = np.where(g["f_te"])[0]
        for s in SEEDS:
            path = f"{OUT}/runs/eicu__{key}__s{s}.json"
            if os.path.exists(path): continue
            tg = time.time(); Z23 = np.load(f"{V23}/preds/{key}__s{s}.npz")
            assert len(Z23["y"]) == len(te) and np.array_equal(Z23["hosp"], hosp_id[te]), f"{key}: v23 held-out rows differ"
            pre = is_train & ~g["r_ctrl"]; fit = is_fit & ~g["r_ctrl"]; X = build_X(pre)
            m, ep = train_es(X[fit], y[fit], X[is_val], y[is_val], s); p_rand = probs(m, X[te]); del m, X
            if DEVICE == "mps": torch.mps.empty_cache()
            yy = y[te]; lm = Z23["landmark"].astype(bool)
            rec = dict(data="eicu", group=key, seed=s, best_epoch=ep, auroc_strict_group=fast_auc(yy, Z23["base"]), auroc_strict_random=fast_auc(yy, p_rand), auroc_pooled=fast_auc(yy, Z23["pooled"]),
                       auroc_strict_group_landmark=fast_auc(yy[lm], Z23["base"][lm]), auroc_strict_random_landmark=fast_auc(yy[lm], p_rand[lm]),
                       n_fit_rows=int(fit.sum()), device=DEVICE, seconds=time.time() - tg, date=time.strftime("%F %T"))
            np.savez_compressed(f"{OUT}/preds/eicu__{key}__s{s}.npz", y=yy, hosp=hosp_id[te], landmark=lm, k_flag=int(Z23["k_flag"]), pooled=Z23["pooled"], strict_group=Z23["base"], strict_random=p_rand)
            save_json(path, rec)
            log(f"{key} s{s}: strict group {rec['auroc_strict_group']:.4f} strict random {rec['auroc_strict_random']:.4f} (landmark {rec['auroc_strict_group_landmark']:.4f} vs {rec['auroc_strict_random_landmark']:.4f}); epoch {ep} ({rec['seconds']:.0f}s)")

if DATA in ("mimic", "both"): run_mimic()
if DATA in ("eicu", "both"): run_eicu()
if os.environ.get("SUMMARY", "1") == "0": log("SUMMARY=0; stopping"); raise SystemExit(0)

# ======================================================================= summary from disk
import boot_draws
res = {}; L = [f"v42 complete-pipeline penalty ({time.strftime('%F %T')}); {NBOOT} draws"]
def load_preds(prefix):
    P = {}
    for f in glob.glob(f"{OUT}/preds/{prefix}__*.npz"):
        k, s = os.path.basename(f)[len(prefix) + 2:-4].rsplit("__s", 1); P[(k, int(s))] = dict(np.load(f))
    seeds = sorted(set(s for _, s in P)); keys = sorted(set(k for k, _ in P if all((k, s) in P for s in seeds)))
    return P, seeds, keys
ARMS = {"complete-pipeline penalty: strict group minus strict random": ("strict_group", "strict_random"),
        "strict random minus pooled": ("strict_random", "pooled"), "strict group minus pooled": ("strict_group", "pooled")}
P, seeds, keys = load_preds("mimic")
if keys:
    def stat(k, idx): return {a: np.nanmean([fast_auc(P[(k, s)]["y"][idx], P[(k, s)][a][idx]) for s in seeds]) for a in ("strict_group", "strict_random", "pooled")}
    def contr(st): return {lab: float(np.mean([o[a] - o[b] for o in st])) for lab, (a, b) in ARMS.items()}
    full = {k: stat(k, np.arange(len(P[(k, seeds[0])]["y"]))) for k in keys}; pt = contr(list(full.values())); pat = {}
    for k in keys:
        u, inv = np.unique(P[(k, seeds[0])]["pid"], return_inverse=True); pat[k] = [np.where(inv == i)[0] for i in range(len(u))]
    r = np.random.RandomState(0); B = {lab: [] for lab in pt}
    for b in range(NBOOT):
        st = [stat(k, np.concatenate([pat[k][j] for j in r.choice(len(pat[k]), len(pat[k]), True)])) for k in r.choice(keys, len(keys), True)]
        for lab, v in contr(st).items(): B[lab].append(v)
    rr = {lab: dict(d=pt[lab], ci=[float(np.percentile(B[lab], 2.5)), float(np.percentile(B[lab], 97.5))], n_groups=len(keys), seeds=seeds, n_boot=NBOOT,
                    lower_in=int(sum(full[k][ARMS[lab][0]] < full[k][ARMS[lab][1]] for k in keys))) for lab in pt}
    res["mimic"] = boot_draws.attach(rr, B, f"{OUT}/draws_mimic.npz")
    L.append(f"MIMIC-IV, {len(keys)} clustered groups, seeds {seeds} (reference-retraining penalty, v5: -0.0106 [-0.0157, -0.0061])")
    for lab, v in res["mimic"].items(): L.append(f"  {lab:62s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p_exact']:.4f}  lower in {v['lower_in']} of {len(keys)}")
P, seeds, keys = load_preds("eicu")
if keys:
    hosps = np.unique(np.concatenate([P[(k, seeds[0])]["hosp"] for k in keys]))
    def stat(k, w, lm):
        o = {}
        for a in ("strict_group", "strict_random", "pooled"):
            vals = []
            for s in seeds:
                Z = P[(k, s)]; m = Z["landmark"].astype(bool) if lm else np.ones(len(Z["y"]), bool)
                rows = np.where(m)[0]; rep = np.repeat(rows, w[np.searchsorted(hosps, Z["hosp"][rows])].astype(int)) if w is not None else rows
                vals.append(fast_auc(Z["y"][rep], Z[a][rep]))
            o[a] = np.nanmean(vals)
        return o
    for lm, tag in [(True, "landmark"), (False, "all")]:
        full = {k: stat(k, None, lm) for k in keys}; pt = {lab: float(np.mean([full[k][a] - full[k][b] for k in keys])) for lab, (a, b) in ARMS.items()}
        r = np.random.RandomState(0); B = {lab: [] for lab in pt}
        for b in range(NBOOT):
            w = np.bincount(r.choice(len(hosps), len(hosps), True), minlength=len(hosps)); st = {k: stat(k, w, lm) for k in keys}
            for lab, (a, bb) in ARMS.items(): B[lab].append(float(np.nanmean([st[k][a] - st[k][bb] for k in keys])))
        rr = {lab: dict(d=pt[lab], ci=[float(np.nanpercentile(B[lab], 2.5)), float(np.nanpercentile(B[lab], 97.5))], n_groups=len(keys), seeds=seeds, n_boot=NBOOT,
                        lower_in=int(sum(full[k][ARMS[lab][0]] < full[k][ARMS[lab][1]] for k in keys))) for lab in pt}
        res[f"eicu_{tag}"] = boot_draws.attach(rr, B, f"{OUT}/draws_eicu_{tag}.npz")
        ref = "-0.0049 [-0.0064, -0.0032]" if lm else "-0.0066 [-0.0083, -0.0045]"
        L.append(f"eICU, {len(keys)} consortia, {'stays in the unit at 24 hours' if lm else 'all held-out stays'}, seeds {seeds} (reference-retraining penalty: {ref})")
        for lab, v in rr.items(): L.append(f"  {lab:62s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p_exact']:.4f}  lower in {v['lower_in']} of {len(keys)}")
save_json(f"{OUT}/intervals.json", dict(results=res, date=time.strftime("%F %T")))
open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); log(f"done, results in {OUT}")
