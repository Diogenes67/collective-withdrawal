"""
exp_v40_boot10k.py. 10,000-draw two-stage bootstrap (groups, then patients within each drawn group) for the three contrasts whose
Holm decision depended on bootstrap resolution in v37 (review item 7): no draw of 1,000 crossed zero, so p could only be bounded
below 0.002.

  v19  events unflagged per 1,000 at matched flag rate, chosen minus pooled (25 clustered groups)
  v21  confined (chosen component) minus SISA full, AUROC (25 clustered groups)
  v33  common base + chosen component minus single-exclusion confined system, AUROC (fifteen random groups, set random_15)

Each contrast is recomputed from the saved predictions with the code and draw order of its source script (RandomState(0); groups
drawn first, then patients within each drawn group; seeds 0, 1, 2), so NBOOT=1000 reproduces the published interval (to float32 rounding) and
NBOOT=10000 extends the same stream. CHECK=1 runs 1,000 draws and asserts that point estimates and intervals match the source
intervals.json to a relative 1e-5. Outputs results/v40_boot10k/{<job>.json, <job>_draws.npz, summary.txt}; one JSON per job, written atomically, so
a rerun skips finished jobs; draws are checkpointed every 250 with the generator state, so an interrupted job resumes on the same
stream. Env: ROOT, NBOOT (default 10000), CHECK=1, ONLY (comma list of v19,v21,v33), OUT_TAG (suffix for a test folder).
"""
import os, sys, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_draws

ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); R = f"{ROOT}/results"
CHECK = os.environ.get("CHECK", "0") == "1"; NB = 1000 if CHECK else int(os.environ.get("NBOOT", "10000"))
OUT = f"{R}/v40_boot10k" + ("_check" if CHECK else "") + os.environ.get("OUT_TAG", ""); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]; t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

def fast_auc(yv, pv):
    o = np.argsort(pv, kind="mergesort"); r = np.empty(len(pv)); r[o] = np.arange(1, len(pv) + 1); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def missed_at_k(yv, pv, k): top = np.argsort(-pv, kind="mergesort")[:k]; return 1000.0 * (yv.sum() - yv[top].sum()) / len(yv)
def save_json(path, obj):
    with open(path + ".tmp", "w") as f: json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)

def community_preds(d):
    P = {}
    for f in glob.glob(f"{d}/community_trainfit__*.npz"):
        g, s = os.path.basename(f)[:-4].rsplit("__s", 1)
        if int(s) in SEEDS: P[(g.replace("__", "/"), int(s))] = dict(np.load(f))
    return P, sorted(set(g for g, _ in P if all((g, s) in P for s in SEEDS)))
def v33_preds(sname):
    members = json.load(open(f"{R}/v33_strict_multigroup/sets.json"))["sets"][sname]
    P = {(n, s): dict(np.load(f"{R}/v33_strict_multigroup/preds/{sname}__community_trainfit__{n}__s{s}.npz")) for n in members for s in SEEDS}
    return P, members

KEYS = {"v19": "events unflagged per 1,000 at matched flag rate, chosen minus pooled",
        "v21": "confined (chosen component) minus SISA full",
        "v33": "common base + chosen component minus single-exclusion confined system"}
JOBS = {"v19": dict(load=lambda: community_preds(f"{R}/v19_protocol/preds"), a="chosen", b="pooled", metric=1,
                    ref=lambda: json.load(open(f"{R}/v19_protocol/intervals.json"))["all"]["results"][KEYS["v19"]]),
        "v21": dict(load=lambda: community_preds(f"{R}/v21_sisa/preds"), a="chosen", b="sisa_full", metric=0,
                    ref=lambda: json.load(open(f"{R}/v21_sisa/intervals.json"))["results"][KEYS["v21"]]),
        "v33": dict(load=lambda: v33_preds("random_15"), a="common_confined", b="single_confined", metric=0,
                    ref=lambda: json.load(open(f"{R}/v33_strict_multigroup/intervals.json"))["random_15"]["results"][KEYS["v33"]])}
only = os.environ.get("ONLY")
L = [f"v40 bootstrap resolution ({time.strftime('%Y-%m-%d %H:%M')}): {NB} draws" + (" (CHECK: reproduces the 1,000-draw intervals)" if CHECK else "")]
for job, J in JOBS.items():
    if only and job not in only.split(","): continue
    path = f"{OUT}/{job}.json"
    if os.path.exists(path):
        res = json.load(open(path)); L.append(f"  {job} {KEYS[job]:75s} {res['d']:+.4f} [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}] p {res['p_exact']:.5f} (n {res['n_boot']}; cached)"); continue
    P, names = J["load"](); a, b, mt = J["a"], J["b"], J["metric"]
    def stat(g, idx):
        out = []
        for arm in (a, b):
            if mt == 0: out.append(np.nanmean([fast_auc(P[(g, s)]["y"][idx], P[(g, s)][arm][idx]) for s in SEEDS]))
            else: out.append(np.mean([missed_at_k(P[(g, s)]["y"][idx], P[(g, s)][arm][idx], max(1, int(round(int(P[(g, s)]["k_flag"]) * len(idx) / len(P[(g, s)]["y"]))))) for s in SEEDS]))
        return out[0] - out[1]
    pt = float(np.mean([stat(g, np.arange(len(P[(g, SEEDS[0])]["y"]))) for g in names])); pat = {}
    for g in names:
        u, inv = np.unique(P[(g, SEEDS[0])]["pid"], return_inverse=True); pat[g] = [np.where(inv == i)[0] for i in range(len(u))]
    r = np.random.RandomState(0); draws = []; ckp = f"{OUT}/{job}_partial.npz"   # checkpoint: draws so far and the generator state
    if os.path.exists(ckp):
        Z = np.load(ckp, allow_pickle=True); draws = list(Z["draws"]); r.set_state(tuple(Z["state"])); log(f"{job}: resuming at draw {len(draws)}")
    log(f"{job}: {len(names)} groups, {NB} draws")
    for i in range(len(draws), NB):
        st = []
        for g in r.choice(names, len(names), True):
            idx = np.concatenate([pat[g][j] for j in r.choice(len(pat[g]), len(pat[g]), True)]); st.append(stat(g, idx))
        draws.append(float(np.mean(st)))
        if (i + 1) % 250 == 0:
            np.savez(ckp + ".tmp.npz", draws=np.asarray(draws), state=np.array(r.get_state(), dtype=object)); os.replace(ckp + ".tmp.npz", ckp)
        if (i + 1) % 1000 == 0: log(f"  {job} draw {i + 1}")
    v = np.asarray(draws); ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    res = dict(source=job, contrast=KEYS[job], d=pt, ci=ci, n_boot=NB, n_groups=len(names), p_exact=boot_draws.exact_p(v),
               draws_crossing_zero=int(min((v <= 0).sum(), (v >= 0).sum())), seeds=SEEDS, date=time.strftime("%Y-%m-%d %H:%M:%S"))
    ref = J["ref"](); res["published"] = dict(d=ref["d"], ci=ref["ci"], n_boot=ref.get("n_boot"))
    if CHECK:
        tol = 1e-5 * max(1.0, abs(ref["d"]))   # float32 sums differ in the last digits between numpy versions; the draw stream is the same
        ok = abs(pt - ref["d"]) < tol and max(abs(ci[0] - ref["ci"][0]), abs(ci[1] - ref["ci"][1])) < tol
        res["reproduces_published"] = bool(ok); log(f"  {job} reproduces the published estimate and interval: {ok}")
        if not ok: raise SystemExit(f"{job}: d {pt} vs {ref['d']}; ci {ci} vs {ref['ci']}")
    np.savez_compressed(f"{OUT}/{job}_draws.npz", keys=np.array([KEYS[job]], dtype=object), d0=v)
    save_json(path, res)
    L.append(f"  {job} {KEYS[job]:75s} {pt:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] p {res['p_exact']:.5f} (draws crossing zero {res['draws_crossing_zero']} of {NB})")
open(f"{OUT}/summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); log(f"done, results in {OUT}")
