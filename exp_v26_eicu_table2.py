"""exp_v26_eicu_table2.py. Table 2 cells for the strict eICU protocol at a matched flag rate, from cached predictions only (numpy):
unflagged deaths per 1,000 for the strict confined system minus the symmetric adapted pooled comparator (v25), the strict confined
system minus the pooled model, and the strict base minus the pooled model (v23), with AUROC for the same contrasts as a check.
Hospital-cluster bootstrap as v20/v23 (one hospital draw feeds all consortia). Run in chunks: CHUNK=0..4 (200 draws), then CHUNK=combine.
"""
import os, json, glob, numpy as np
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); OUT = f"{ROOT}/results/v26_eicu_table2"; os.makedirs(OUT, exist_ok=True)
CHUNK = os.environ.get("CHUNK", "combine"); POP = os.environ.get("POP", "all")
P = {}
for f in glob.glob(f"{ROOT}/results/v25_symmetric_eicu/preds/*.npz"):
    k, s = os.path.basename(f)[:-4].rsplit("__s", 1); Z = np.load(f); Y = np.load(f"{ROOT}/results/v23_eicu_strict/preds/{k}__s{s}.npz")
    assert np.array_equal(Z["y"], Y["y"])
    P[(k, int(s))] = dict(y=Z["y"], hosp=Z["hosp"], landmark=Z["landmark"], k_flag=int(Z["k_flag"]), pooled=Z["pooled"], sym=Z["pooled_cf_chosen"],
                          strict_chosen=Z["confined_strict_chosen"], strict_base=Y["base"])
seeds = sorted(set(s for _, s in P)); keys = sorted(set(k for k, _ in P))
PAIRS = {"strict confined minus symmetric comparator": ("strict_chosen", "sym"), "strict confined minus pooled": ("strict_chosen", "pooled"),
         "strict base minus pooled": ("strict_base", "pooled"), "symmetric comparator minus pooled": ("sym", "pooled")}
ARMS = ["pooled", "sym", "strict_chosen", "strict_base"]
def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
hosp_rows = {k: {h: np.where(P[(k, seeds[0])]["hosp"] == h)[0] for h in np.unique(P[(k, seeds[0])]["hosp"])} for k in keys}
hosp_all = np.unique(np.concatenate([P[(k, seeds[0])]["hosp"] for k in keys]))
def stats(w):
    out = []
    for k in keys:
        Z0 = P[(k, seeds[0])]
        idx = np.concatenate([np.tile(hosp_rows[k][h], c) for h, c in w.items() if h in hosp_rows[k]]) if w is not None else np.arange(len(Z0["y"]))
        if POP == "landmark": idx = idx[Z0["landmark"][idx]]
        if len(idx) < 50 or len(np.unique(Z0["y"][idx])) < 2: continue
        o = {}
        for a in ARMS:
            o[a] = (np.nanmean([fast_auc(P[(k, s)]["y"][idx], P[(k, s)][a][idx]) for s in seeds]),
                    np.mean([missed(P[(k, s)]["y"][idx], P[(k, s)][a][idx], max(1, int(round(P[(k, s)]["k_flag"] * len(idx) / len(Z0["y"]))))) for s in seeds]))
        out.append(o)
    return out
def contrasts(st):
    d = {}
    for n, (a, b) in PAIRS.items():
        d["AUROC | " + n] = np.mean([o[a][0] - o[b][0] for o in st]); d["unflagged per 1,000 | " + n] = np.mean([o[a][1] - o[b][1] for o in st])
    return d
if CHUNK == "combine":
    pt = contrasts(stats(None)); B = {k: [] for k in pt}
    for c in range(5):
        Z = np.load(f"{OUT}/{POP}_chunk{c}.npz", allow_pickle=True)["B"].item()
        for k in pt: B[k] += Z[k]
    res = {k: dict(d=float(pt[k]), ci=[float(np.percentile(B[k], 2.5)), float(np.percentile(B[k], 97.5))], n_boot=len(B[k])) for k in pt}
    json.dump(res, open(f"{OUT}/{POP}.json", "w"), indent=1)
    for k, v in res.items(): print(f"{k:70s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
else:
    c = int(CHUNK); r = np.random.RandomState(9000 + c + (100 if POP == "landmark" else 0)); B = None
    for b in range(200):
        drawn = r.choice(hosp_all, len(hosp_all), True); w = dict(zip(*np.unique(drawn, return_counts=True)))
        d = contrasts(stats(w))
        if B is None: B = {k: [] for k in d}
        for k, v in d.items(): B[k].append(v)
    np.savez_compressed(f"{OUT}/{POP}_chunk{c}.npz", B=np.array(B, dtype=object)); print(POP, "chunk", c, "done")
