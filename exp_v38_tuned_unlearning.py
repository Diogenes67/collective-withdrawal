"""
exp_v38_tuned_unlearning.py. NegGrad+ and Fair Unlearning tuned on retained validation data (review item M3), MIMIC-IV, 25
clustered groups x 3 seeds, each method applied to the group and to its patient-matched random set as in v10 and v22.

Tuning. For each group, seed and arm (group withdrawal; random removal), a fifth of the retained training patients is held
back (RandomState(4000 + group index + 31 x seed), the component-selection split), the method is run from the original
weights on the remaining retained rows for every configuration of the grid, and the configuration with the lowest log loss
on the held-back patients is refitted on all retained rows. The held-out patients play no part. The budget matches the
component selection of the protocol: selection on a held-back fifth, refit on all. Grid (GRID=small, default): learning rate
{1e-4, 1e-3}, epochs {2, 4}, forget weight {0.05, 0.2, 0.5} (NegGrad+, the weight on the clamped forget-set loss) and
{0.1, 0.3, 1.0} (Fair Unlearning, the weight on the projected forget gradient; the equality weight stays 1). GRID=full adds
learning rate 3e-4 and 1 epoch. GRID=ext (review: the small grid chose its lowest learning rate and forget weight) extends
every bounded direction and closes the forget weight at zero: learning rate {1e-5, 3e-5, 1e-4, 1e-3}, epochs {1, 2, 4}, forget
weight {0, 0.02, 0.05, 0.2, 0.5} (NegGrad+) and {0, 0.03, 0.1, 0.3, 1.0} (Fair Unlearning); the small grid is a subset. At
forget weight 0 NegGrad+ is fine-tuning on the retained rows and Fair Unlearning is fine-tuning with the equality term.
GRID=ext writes to results/v38_tuned_unlearning_ext. Seeds in SEEDS_TUNE (default 0) are tuned; the other seeds reuse the configuration chosen
for the group and arm at the first tuned seed, so that training randomness enters the refit but not the selection.
Removal diagnostics per configuration, recorded but not used for selection: the forget gap (mean cross-entropy of the exact
retrain on the group's training records minus that of the candidate; positive means the candidate still fits the group's
records better than a model that never saw them) and the loss-based membership attack of v9 (logistic classifier on
per-record loss, confidence, entropy and label, five-fold cross-validated AUROC separating the group's training records
from its held-out records; 0.5 is no separation).
Evaluation. On the group's held-out patients: AUROC and readmissions unflagged per 1,000 at the matched flag rate (k = the
original model's flags at its system-wide top-decile threshold), each tuned method on the group minus the same tuned method on
the random set, and minus the original; two-stage bootstrap (groups, then patients; 1,000 draws; RandomState(0); draws saved;
exact percentile p-values). The frontier table gives, per configuration and method, the mean validation log loss, forget gap,
attack AUROC and change in the group's held-out AUROC, so that the trade-off between utility and removal is visible.

Env: ROOT, QUICK=1 (two groups, seed 0, two epochs, four configurations, 30 draws), DEVICE, GRID=small|full|ext, SEEDS (default 0,1,2),
SEEDS_TUNE (default 0), ONLY_GROUPS (comma list, e.g. c05,c20), NBOOT, SUMMARY=0. Reads derived/cohort_latest.parquet,
results/v5_partitions/{ckpt,runs}. Writes results/v38_tuned_unlearning/{runs, preds, ckpt, configs.csv, frontier.csv, summary.csv,
intervals.json, draws.npz, summary.txt, log.txt}. Resumable: finished runs, configurations and checkpoints are skipped.
"""
import os, json, time, glob, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

QUICK = os.environ.get("QUICK", "0") == "1"
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); V5 = f"{ROOT}/results/v5_partitions"
GRID = os.environ.get("GRID", "small")
OUT = f"{ROOT}/results/v38_tuned_unlearning" + ("_quick" if QUICK else "") + ("_ext" if GRID == "ext" else "")
for d in ["runs", "preds", "ckpt", "configs"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
SEEDS = [0] if QUICK else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
SEEDS_TUNE = [0] if QUICK else [int(x) for x in os.environ.get("SEEDS_TUNE", "0").split(",")]
LRS = [1e-4, 1e-3] if GRID == "small" else ([1e-5, 3e-5, 1e-4, 1e-3] if GRID == "ext" else [1e-4, 3e-4, 1e-3]); EPOCHS = [2, 4] if GRID == "small" else [1, 2, 4]
FW = {"neggrad": [0.0, 0.02, 0.05, 0.2, 0.5], "fu_community": [0.0, 0.03, 0.1, 0.3, 1.0]} if GRID == "ext" else {"neggrad": [0.05, 0.2, 0.5], "fu_community": [0.1, 0.3, 1.0]}
if QUICK: LRS, EPOCHS, FW = [1e-4, 1e-3], [1], {"neggrad": [0.05, 0.5], "fu_community": [0.1, 1.0]}
CONFIGS = {m: [dict(lr=lr, epochs=ep, fw=fw) for lr, ep, fw in itertools.product(LRS, EPOCHS, FW[m])] for m in FW}
CFG = dict(seeds=SEEDS, seeds_tune=SEEDS_TUNE, grid=GRID, lrs=LRS, epochs=EPOCHS, forget_weights=FW, batch=2048, forget_cap=0.6, fu_beta=1.0, fu_gamma=1.0, fu_min_group_in_batch=5,
           select_val_frac=0.2, min_patients=300, attack_sample=5000, n_boot=int(os.environ.get("NBOOT", 30 if QUICK else 1000)), version="v38.1" if GRID == "ext" else "v38.0")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
t0 = time.time(); LOG = open(f"{OUT}/log.txt", "a")
def log(m):
    line = f"[{time.time()-t0:7.1f}s] {m}"; print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
log(f"device {DEVICE}; quick={QUICK}; grid {GRID}: {len(CONFIGS['neggrad'])} NegGrad+ and {len(CONFIGS['fu_community'])} Fair Unlearning configurations; seeds {SEEDS} (tuned on {SEEDS_TUNE}); out {OUT}")

# ---------------------------------------------------------------- data, split, groups and random sets exactly as v5 / v10 / v30
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); y = df["readmit30"].values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test; test_idx = np.where(is_test)[0]
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
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
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
ASSIGN = f"{ROOT}/derived/cluster_assign_v5.csv"   # the clustered groups as fitted on the analysis machine; KMeans on another platform can assign patients differently
if os.path.exists(ASSIGN):
    assign = pd.read_csv(ASSIGN).set_index("subject_id")["cluster"].reindex(pat_mean.index)
    if assign.isna().any(): raise SystemExit(f"{ASSIGN} lacks {int(assign.isna().sum())} patients")
    assign = assign.astype(int); log(f"clustered groups read from {ASSIGN}")
else:
    km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
comm = assign.loc[sid].values
for c in range(25): labels[("community_trainfit", f"c{c:02d}")] = set(assign.index[assign == c])
groups = []
for (fam, name), members in labels.items():
    mem = np.isin(sid, np.fromiter(members, dtype=sid.dtype)); f_tr, f_te = is_train & mem, is_test & mem
    n_tr_p, n_te_p = len(set(sid[f_tr])), len(set(sid[f_te]))
    if n_tr_p < CFG["min_patients"] or n_te_p < 50: continue
    groups.append(dict(family=fam, name=name, f_tr=f_tr, f_te=f_te, n_tr_p=n_tr_p, n_te_p=n_te_p))
for i, g in enumerate(groups):
    if g["family"] == "community_trainfit":
        r = np.random.RandomState(100 + i); tr_p = np.array(sorted(set(sid[is_train]) - set(sid[g["f_tr"]])))
        g["r_pat"] = np.isin(sid, r.choice(tr_p, min(g["n_tr_p"], len(tr_p)), False)) & is_train; g["gi"] = i
comms = [g for g in groups if g["family"] == "community_trainfit"]
for g in comms:
    ref = json.load(open(f"{V5}/runs/community_trainfit__{g['name']}__s0.json"))
    if ref["n_train_patients"] != g["n_tr_p"] or ref["n_test_patients"] != g["n_te_p"] or ref["rows_random_pat"] != int(g["r_pat"].sum()): raise SystemExit(f"{g['name']}: membership or random set differs from v5")
only = os.environ.get("ONLY_GROUPS")
if only: comms = [g for g in comms if g["name"] in only.split(",")]
if QUICK: comms = comms[:2]
log(f"{len(comms)} clustered groups match the v5 records")

# ---------------------------------------------------------------- model, methods (v5 NegGrad+, v10 Fair Unlearning) and diagnostics
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
def to_t(a): return torch.tensor(a, device=DEVICE)
def load(path):
    m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval(); return m
def predict(m, Xa):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 65536): out.append(torch.sigmoid(m(to_t(Xa[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def neggrad(init, r_rows, f_rows, seed, lr, epochs, fw):
    torch.manual_seed(seed); np.random.seed(seed); m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=lr); bce = nn.BCEWithLogitsLoss()
    Xt, yt, Xf, yf = to_t(X[r_rows]), to_t(y[r_rows]), to_t(X[f_rows]), to_t(y[f_rows]); n, B = len(Xt), CFG["batch"]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; j = torch.randint(0, len(Xf), (min(B, len(Xf)),), device=DEVICE)
            loss = bce(m(Xt[idx]), yt[idx]) - fw * torch.clamp(bce(m(Xf[j]), yf[j]), max=CFG["forget_cap"]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m
def equality_loss(p, yy, g):
    loss = p.new_zeros(()); gs = torch.unique(g)
    for lab in (0.0, 1.0):
        means = []
        for gi in gs:
            mk = (g == gi) & (yy == lab)
            if mk.sum() >= CFG["fu_min_group_in_batch"]: means.append(p[mk].mean())
        for i in range(len(means)):
            for j in range(i + 1, len(means)): loss = loss + (means[i] - means[j]) ** 2
    return loss
def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
def fair_unlearn(init, r_rows, f_rows, group, seed, lr, epochs, fw):
    torch.manual_seed(seed); np.random.seed(seed); m = MLP(X.shape[1]).to(DEVICE); m.load_state_dict(init); params = [p for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr); bce = nn.BCEWithLogitsLoss()
    Xr, yr, gr = to_t(X[r_rows]), to_t(y[r_rows]), to_t(group[r_rows]); Xf, yf = to_t(X[f_rows]), to_t(y[f_rows]); n, B = len(Xr), CFG["batch"]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, B):
            idx = perm[i:i + B]; logits = m(Xr[idx]); p = torch.sigmoid(logits)
            L_r = bce(logits, yr[idx]); L_e = equality_loss(p, yr[idx], gr[idx])
            j = torch.randint(0, len(Xf), (min(B, len(Xf)),), device=DEVICE); L_f = torch.clamp(bce(m(Xf[j]), yf[j]), max=CFG["forget_cap"])
            g_r = flat_grad(L_r, params); g_e = flat_grad(L_e, params) if L_e.requires_grad else torch.zeros_like(g_r); g_f = flat_grad(-L_f, params)
            v1 = g_r / (g_r.norm() + 1e-12); e2 = g_e - (g_e @ v1) * v1; v2 = e2 / (e2.norm() + 1e-12) if e2.norm() > 1e-8 else torch.zeros_like(v1)
            g_f_perp = g_f - (g_f @ v1) * v1 - (g_f @ v2) * v2; g = fw * g_f_perp + CFG["fu_beta"] * g_r + CFG["fu_gamma"] * g_e
            opt.zero_grad(); off = 0
            for p_ in params:
                k = p_.numel(); p_.grad = g[off:off + k].reshape(p_.shape).clone(); off += k
            opt.step()
    m.eval(); return m
def logloss(yy, pp): pp = np.clip(pp, 1e-7, 1 - 1e-7); return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))
def record_losses(yy, pp): pp = np.clip(pp, 1e-7, 1 - 1e-7); return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
def attack_features(yy, pp):
    pp = np.clip(pp, 1e-7, 1 - 1e-7); loss = record_losses(yy, pp); conf = np.where(yy == 1, pp, 1 - pp); ent = -(pp * np.log(pp) + (1 - pp) * np.log(1 - pp))
    return np.column_stack([loss, conf, ent, yy])
def loss_attack(m, tr_rows, te_rows, seed):
    """v9 loss-based membership attack: the group's training records against its held-out records, five-fold CV AUROC."""
    r = np.random.RandomState(seed); a = r.choice(tr_rows, min(CFG["attack_sample"], len(tr_rows)), False); b = r.choice(te_rows, min(CFG["attack_sample"], len(te_rows)), False)
    F = np.vstack([attack_features(y[a], predict(m, X[a])), attack_features(y[b], predict(m, X[b]))]); lab = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=1000), F, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")))
def auc(yy, pp):
    n1 = int(yy.sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0: return np.nan
    o = np.argsort(pp, kind="mergesort"); ps = pp[o]; _, idx, cnt = np.unique(ps, return_index=True, return_counts=True)
    r = np.empty(len(pp)); r[o] = np.repeat(idx + (cnt + 1) / 2.0, cnt); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
def unflagged_at_k(yy, pp, k): top = np.argsort(-pp, kind="mergesort")[:k]; return 1000.0 * (yy.sum() - yy[top].sum()) / len(yy)
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
def run_method(meth, init, r_rows, f_rows, group_vec, seed, cfg):
    return neggrad(init, r_rows, f_rows, seed, cfg["lr"], cfg["epochs"], cfg["fw"]) if meth == "neggrad" else fair_unlearn(init, r_rows, f_rows, group_vec, seed, cfg["lr"], cfg["epochs"], cfg["fw"])

# ---------------------------------------------------------------- runs
originals = {s: load(f"{V5}/ckpt/orig_s{s}.pt") for s in SEEDS}
thr = {s: float(np.percentile(predict(originals[s], X[test_idx]), 90)) for s in SEEDS}
ARMS = {"group": "", "random": "rp"}
for g in comms:
    key = f"community_trainfit_{g['name']}"; c = int(g["name"][1:]); te = np.where(g["f_te"])[0]; yy = y[te]; comm_vec = (comm == c).astype(np.int64)
    for s in SEEDS:
        path = f"{OUT}/runs/{key}__s{s}.json"
        if os.path.exists(path): continue
        tg = time.time(); init = originals[s].state_dict(); p = {"original": predict(originals[s], X[te])}; rec_arms = {}
        for arm, suf in ARMS.items():
            f_mask = g["f_tr"] if arm == "group" else g["r_pat"]; f_rows = np.where(f_mask)[0]; r_all = np.where(is_train & ~f_mask)[0]
            exact = load(f"{V5}/ckpt/exact{suf}_{key}_s{s}.pt"); p[f"exact_{arm}"] = predict(exact, X[te]); loss_exact_f = logloss(y[f_rows], predict(exact, X[f_rows]))
            # held-back fifth of the retained patients for selection
            rs = np.random.RandomState(4000 + g["gi"] + 31 * s); ups = np.unique(sid[r_all]); rs.shuffle(ups); val_p = set(ups[: max(2, int(CFG["select_val_frac"] * len(ups)))])
            in_val = np.isin(sid[r_all], list(val_p)); r_fit, r_val = r_all[~in_val], r_all[in_val]
            for meth in CONFIGS:
                cpath = f"{OUT}/configs/{key}__{arm}__{meth}__s{s}.json"
                if s in SEEDS_TUNE and not os.path.exists(cpath):
                    rows = []
                    for ci, cfg in enumerate(CONFIGS[meth]):
                        tc = time.time(); m = run_method(meth, init, r_fit, f_rows, comm_vec, s, cfg)
                        rows.append(dict(config=ci, **cfg, val_logloss=logloss(y[r_val], predict(m, X[r_val])), forget_gap=loss_exact_f - logloss(y[f_rows], predict(m, X[f_rows])),
                                         attack_auroc=loss_attack(m, f_rows, te, s) if arm == "group" else float("nan"), group_auroc=auc(yy, predict(m, X[te])), seconds=time.time() - tc))
                        del m
                        if DEVICE == "mps": torch.mps.empty_cache()
                    chosen = int(min(rows, key=lambda r_: r_["val_logloss"])["config"])
                    save_json(cpath, dict(group=key, arm=arm, method=meth, seed=s, chosen=chosen, configs=rows, loss_exact_forget=loss_exact_f))
                    log(f"  {key} s{s} {arm} {meth}: chose {CONFIGS[meth][chosen]} (val log loss {rows[chosen]['val_logloss']:.4f}; forget gap {rows[chosen]['forget_gap']:+.4f}; attack {rows[chosen]['attack_auroc']:.3f}) ({time.time()-tg:.0f}s)")
                elif not os.path.exists(cpath):
                    src = f"{OUT}/configs/{key}__{arm}__{meth}__s{SEEDS_TUNE[0]}.json"
                    if not os.path.exists(src): raise SystemExit(f"{key} {arm} {meth}: tune seed {SEEDS_TUNE[0]} first (missing {src})")
                    save_json(cpath, dict(group=key, arm=arm, method=meth, seed=s, chosen=json.load(open(src))["chosen"], configs=None, reused_from_seed=SEEDS_TUNE[0]))
                cj = json.load(open(cpath)); cfg = CONFIGS[meth][cj["chosen"]]
                ck = f"{OUT}/ckpt/{meth}_{arm}_{key}_s{s}.pt"
                if os.path.exists(ck): m = load(ck)
                else:
                    m = run_method(meth, init, r_all, f_rows, comm_vec, s, cfg); torch.save(m.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
                p[f"{meth}_{arm}"] = predict(m, X[te])
                rec_arms[f"{meth}_{arm}"] = dict(config=cfg, chosen=cj["chosen"], tuned=cj.get("configs") is not None, auroc=auc(yy, p[f"{meth}_{arm}"]),
                                                 forget_gap=loss_exact_f - logloss(y[f_rows], predict(m, X[f_rows])), attack_auroc=loss_attack(m, f_rows, te, s) if arm == "group" else float("nan"))
                del m
        k_flag = int((p["original"] >= thr[s]).sum())
        np.savez_compressed(f"{OUT}/preds/{key}__s{s}.npz", y=yy, pid=sid[te], k_flag=k_flag, **p)
        save_json(path, dict(group=key, seed=s, n_train_patients=g["n_tr_p"], n_test_patients=g["n_te_p"], k_flag=k_flag, arms=rec_arms,
                             auroc={a: auc(yy, v) for a, v in p.items()}, unflagged={a: unflagged_at_k(yy, v, k_flag) for a, v in p.items()}, cfg=CFG, device=DEVICE, date=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - tg))
        log(f"{key} s{s}: original {auc(yy, p['original']):.4f} exact {auc(yy, p['exact_group']):.4f} | NegGrad+ group {auc(yy, p['neggrad_group']):.4f} random {auc(yy, p['neggrad_random']):.4f} | "
            f"FU group {auc(yy, p['fu_community_group']):.4f} random {auc(yy, p['fu_community_random']):.4f} ({time.time()-tg:.0f}s)")
if os.environ.get("SUMMARY", "1") == "0": log("SUMMARY=0; stopping before the summary stage"); raise SystemExit(0)

# ---------------------------------------------------------------- summary from disk
import boot_draws
recs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/runs/*.json"))]
frontier = []
for f in sorted(glob.glob(f"{OUT}/configs/*.json")):
    cj = json.load(open(f))
    if cj.get("configs"): frontier += [dict(group=cj["group"], arm=cj["arm"], method=cj["method"], seed=cj["seed"], chosen=(r_["config"] == cj["chosen"]), **r_) for r_ in cj["configs"]]
FR = pd.DataFrame(frontier)
if len(FR):
    FR.to_csv(f"{OUT}/configs.csv", index=False)
    FT = FR[FR.arm == "group"].groupby(["method", "lr", "epochs", "fw"]).agg(val_logloss=("val_logloss", "mean"), forget_gap=("forget_gap", "mean"), attack_auroc=("attack_auroc", "mean"), group_auroc=("group_auroc", "mean"), chosen_in=("chosen", "sum"), n=("group", "count")).reset_index()
    FT.to_csv(f"{OUT}/frontier.csv", index=False)
rows = [dict(group=r["group"], seed=r["seed"], arm=a, auroc=r["auroc"][a], unflagged=r["unflagged"][a]) for r in recs for a in r["auroc"]]
S = pd.DataFrame(rows).groupby(["group", "arm"]).mean(numeric_only=True).reset_index(); S.to_csv(f"{OUT}/summary.csv", index=False)
W = S.pivot_table(index="group", columns="arm", values="auroc"); U = S.pivot_table(index="group", columns="arm", values="unflagged")
P = {}
for f in glob.glob(f"{OUT}/preds/*.npz"):
    Z = np.load(f); k_, s_ = os.path.basename(f)[:-4].rsplit("__s", 1); P[(k_, int(s_))] = {k: Z[k] for k in Z.files}
seeds = sorted(set(s_ for _, s_ in P)); keys = sorted(set(k for k, _ in P if all((k, s_) in P for s_ in seeds)))
arms = [a for a in P[(keys[0], seeds[0])] if a not in ("y", "pid", "k_flag")]; meths = ["exact", "neggrad", "fu_community"]
def stats(k, idx):
    o = {}
    for a in arms:
        o[a] = (np.nanmean([auc(P[(k, s)]["y"][idx], P[(k, s)][a][idx]) for s in seeds]),
                np.mean([unflagged_at_k(P[(k, s)]["y"][idx], P[(k, s)][a][idx], max(1, int(round(int(P[(k, s)]["k_flag"]) * len(idx) / len(P[(k, s)]["y"]))))) for s in seeds]))
    return o
def contrasts(st):
    d = {}
    for m in meths:
        d[f"AUROC | {m} group minus {m} random"] = np.mean([o[f"{m}_group"][0] - o[f"{m}_random"][0] for o in st])
        d[f"AUROC | {m} group minus original"] = np.mean([o[f"{m}_group"][0] - o["original"][0] for o in st])
        d[f"unflagged per 1,000 | {m} group minus {m} random"] = np.mean([o[f"{m}_group"][1] - o[f"{m}_random"][1] for o in st])
        d[f"unflagged per 1,000 | {m} group minus original"] = np.mean([o[f"{m}_group"][1] - o["original"][1] for o in st])
    for m in ["neggrad", "fu_community"]: d[f"AUROC | {m} group minus exact group (tuned method against exact retraining)"] = np.mean([o[f"{m}_group"][0] - o["exact_group"][0] for o in st])
    return d
pt = contrasts([stats(k, np.arange(len(P[(k, seeds[0])]["y"]))) for k in keys]); pat = {}
for k in keys:
    u, inv = np.unique(P[(k, seeds[0])]["pid"], return_inverse=True); pat[k] = [np.where(inv == i)[0] for i in range(len(u))]
r = np.random.RandomState(0); boots = {k: [] for k in pt}
for b in range(CFG["n_boot"]):
    st = [stats(k, np.concatenate([pat[k][j] for j in r.choice(len(pat[k]), len(pat[k]), True)])) for k in r.choice(keys, len(keys), True)]
    for kk, v in contrasts(st).items(): boots[kk].append(v)
    if (b + 1) % 200 == 0: log(f"  bootstrap draw {b + 1}")
res = {k: dict(d=float(pt[k]), ci=[float(np.nanpercentile(boots[k], 2.5)), float(np.nanpercentile(boots[k], 97.5))], n_groups=len(keys), n_boot=CFG["n_boot"]) for k in pt}
res = boot_draws.attach(res, boots, f"{OUT}/draws.npz")
lw = {m: int(sum(W.loc[k, f"{m}_group"] < W.loc[k, f"{m}_random"] for k in keys)) for m in meths}
diag = pd.DataFrame([dict(group=r_["group"], seed=r_["seed"], arm=a, **{kk: v for kk, v in d.items() if kk in ("forget_gap", "attack_auroc", "chosen")}, **d["config"]) for r_ in recs for a, d in r_["arms"].items()])
diag.to_csv(f"{OUT}/chosen_configs.csv", index=False)
L = [f"v38 tuned unlearning ({time.strftime('%Y-%m-%d %H:%M')}): {len(keys)} groups, seeds {seeds}, grid {GRID}, two-stage bootstrap ({CFG['n_boot']} draws)"]
for k, v in res.items(): L.append(f"  {k:95s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v.get('p_exact', float('nan')):.4f}")
L.append("groups with lower AUROC than the same method on random removal: " + ", ".join(f"{m} {lw[m]} of {len(keys)}" for m in meths))
for m in ["neggrad", "fu_community"]:
    sub = diag[diag.arm == f"{m}_group"]
    L.append(f"  {m}: chosen configurations (group arm) " + ", ".join(f"lr {lr:g}/ep {int(ep)}/fw {fw:g}: {n}" for (lr, ep, fw), n in sub.groupby(["lr", "epochs", "fw"]).size().items())
             + f"; mean forget gap {sub.forget_gap.mean():+.4f}; mean attack AUROC {sub.attack_auroc.mean():.3f}")
if len(FR):
    L.append("frontier (group arm; per configuration, mean over tuned group-seeds): method lr epochs fw | val log loss | forget gap | attack AUROC | group AUROC | chosen in")
    for r_ in FT.itertuples(): L.append(f"  {r_.method:13s} {r_.lr:g} {int(r_.epochs)} {r_.fw:g} | {r_.val_logloss:.4f} | {r_.forget_gap:+.4f} | {r_.attack_auroc:.3f} | {r_.group_auroc:.4f} | {int(r_.chosen_in)} of {int(r_.n)}")
save_json(f"{OUT}/intervals.json", dict(results=res, lower_in=lw, cfg=CFG, date=time.strftime("%Y-%m-%d %H:%M:%S")))
open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); log(f"done, results in {OUT}")
