"""
boot_draws.py. Keeps every bootstrap draw beside a script's interval summary and adds exact percentile p-values (v9 revision,
review item M6). The scripts already draw with a stored seed (RandomState(0)); this module saves what they draw so that
exp_v37_exact_p.py can replace the normal-approximation p-values of Table S19 with exact ones.

attach(res, boots, path): res is a script's {contrast: dict(d=..., ci=[lo, hi], ...)} and boots the {contrast: [draws]} it was
computed from. Writes path (an .npz with the contrast names in `keys` and one array per contrast) and adds to each entry
p_exact (two-sided percentile p-value at zero, 2 x the smaller tail, floored at 2/n), n_boot_valid and draws_file. Returns res.
load(path) returns {contrast: draws}. exact_p(draws) is the p-value alone.
"""
import os
import numpy as np

def exact_p(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return float("nan")
    return float(min(1.0, max(2.0 / len(v), 2 * min((v <= 0).mean(), (v >= 0).mean()))))

def attach(res, boots, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = [k for k in boots]
    np.savez_compressed(path, keys=np.array(keys, dtype=object), **{f"d{i}": np.asarray(boots[k], float) for i, k in enumerate(keys)})
    for k in res:
        if k in boots:
            v = np.asarray(boots[k], float); res[k]["p_exact"] = exact_p(v); res[k]["n_boot_valid"] = int((~np.isnan(v)).sum()); res[k]["draws_file"] = os.path.basename(path)
    return res

def load(path):
    Z = np.load(path, allow_pickle=True); keys = list(Z["keys"]); return {k: Z[f"d{i}"] for i, k in enumerate(keys)}
