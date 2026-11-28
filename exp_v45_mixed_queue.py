"""
exp_v45_mixed_queue.py. Other patients' flags when a group's component is deleted and a shared top-decile queue is
recomputed over all held-out patients (external review, 28 Nov 2026). No training; reads cached predictions.

Before deletion the queue scores other held-out patients with the base and the group's held-out patients with the base
plus the group's validation-chosen component. After deletion the group's patients are scored by the base alone. Other
patients' scores are identical in both states; only the shared top-decile threshold can move. Reported per 1,000 other
admissions or stays: flags changed, and flags changed at a threshold held fixed (zero by construction).
MIMIC-IV: v16_strict/preds_others (strict_base on others), v16_strict/preds (strict_base on the group),
v24_symmetric_mimic/preds (confined_chosen on the group). eICU: v23_eicu_strict/preds_others and preds (base, chosen),
all held-out stays and the 24-hour landmark. Writes results/v45_mixed_queue/{per_group.csv, summary.txt}. Aggregates only.
"""
import os, glob, numpy as np, pandas as pd
R = os.path.expanduser("~/mimic/results") if os.path.isdir(os.path.expanduser("~/mimic/results")) else "results"
OUT = f"{R}/v45_mixed_queue"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
def moved(others, g_before, g_after, cap=0.10):
    pre = np.concatenate([others, g_before]); post = np.concatenate([others, g_after])
    t0 = np.percentile(pre, 100 * (1 - cap)); t1 = np.percentile(post, 100 * (1 - cap))
    f0, f1 = others >= t0, others >= t1
    return 1000 * (f0 != f1).mean(), 1000 * (f0 & ~f1).mean(), 1000 * (~f0 & f1).mean(), t0, t1
rows = []
for f in sorted(glob.glob(f"{R}/v24_symmetric_mimic/preds/community_trainfit__*__s*.npz")):
    b = os.path.basename(f); n, s = b.split("__")[1], int(b.split("__s")[1][:-4])
    if s not in SEEDS: continue
    C = np.load(f, allow_pickle=True); G = np.load(f"{R}/v16_strict/preds/{b}"); O = np.load(f"{R}/v16_strict/preds_others/{b}")
    assert np.array_equal(C["pid"], G["pid"]), b
    m, dn, up, t0, t1 = moved(O["strict_base"], C["confined_chosen"], G["strict_base"])
    rows.append(dict(dataset="MIMIC-IV", population="all", group=n, seed=s, n_others=len(O["y"]), n_group=len(G["y"]), moved_per_1000=m, dropped_per_1000=dn, added_per_1000=up, thr_before=t0, thr_after=t1, moved_fixed_per_1000=0.0))
for f in sorted(glob.glob(f"{R}/v23_eicu_strict/preds/*__s*.npz")):
    b = os.path.basename(f); k, s = b.rsplit("__s", 1)[0], int(b.rsplit("__s", 1)[1][:-4])
    if s not in SEEDS: continue
    G = np.load(f); O = np.load(f"{R}/v23_eicu_strict/preds_others/{b}")
    for pop in ["all", "landmark"]:
        go = np.ones(len(G["y"]), bool) if pop == "all" else G["landmark"].astype(bool)
        oo = np.ones(len(O["y"]), bool) if pop == "all" else O["landmark"].astype(bool)
        m, dn, up, t0, t1 = moved(O["strict_base"][oo], G["chosen"][go], G["base"][go])
        rows.append(dict(dataset="eICU", population=pop, group=k, seed=s, n_others=int(oo.sum()), n_group=int(go.sum()), moved_per_1000=m, dropped_per_1000=dn, added_per_1000=up, thr_before=t0, thr_after=t1, moved_fixed_per_1000=0.0))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/per_group.csv", index=False)
g = df.groupby(["dataset", "population", "group"]).mean(numeric_only=True).reset_index()
lines = ["v45 mixed queue: other patients' flags after component deletion, shared top decile recomputed over all held-out patients (seed means; per 1,000 other admissions or stays)"]
for (d, p), h in g.groupby(["dataset", "population"]):
    lines.append(f"{d} {p}: groups {len(h)}, moved mean {h.moved_per_1000.mean():.2f} (range {h.moved_per_1000.min():.2f} to {h.moved_per_1000.max():.2f}); dropped {h.dropped_per_1000.mean():.2f}, added {h.added_per_1000.mean():.2f}; threshold change mean {(h.thr_after-h.thr_before).mean():+.5f}; fixed threshold 0")
open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
