"""
exp_v38m_matched.py. Removal-matched comparison of the tuned unlearning methods with exact retraining (v38 extended grid,
post hoc): for each clustered group (tuned seed 0), the configuration whose forget gap is closest to zero, i.e. whose fit of the
group's training records matches the exact retrain's, minus exact retraining on the group's held-out patients; and, as a
second rule, the configuration with the lowest held-back log loss among those with forget gap <= 0. Bootstrap over the 25
groups (5,000 draws, RandomState(0)); draws saved in the boot_draws format. Reads results/v38_tuned_unlearning_ext/{configs.csv,
runs/*__s0.json}. Writes results/v38_tuned_unlearning_ext/{matched.json, matched_draws.npz}.
"""
import os, json, glob, numpy as np, pandas as pd, zipfile
import boot_draws
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); D = f"{ROOT}/results/v38_tuned_unlearning_ext"
if not glob.glob(f"{D}/runs/*.json"):
    z = zipfile.ZipFile(f"{ROOT}/results/colab_v9/results_v38_ext.zip")
    for n in z.namelist():
        if "/runs/" in n: z.extract(n, ROOT)
C = pd.read_csv(f"{D}/configs.csv"); C = C[C.arm == "group"]
ex = {}
for f in glob.glob(f"{D}/runs/*__s0.json"):
    r = json.load(open(f)); ex[r["group"]] = r["auroc"]["exact_group"]
res, boots = {}, {}
for m, lab in [("neggrad", "NegGrad+"), ("fu_community", "Fair Unlearning")]:
    d = C[C.method == m]; groups = sorted(d.group.unique())
    v1 = np.array([d[d.group == g].loc[d[d.group == g].forget_gap.abs().idxmin()].group_auroc - ex[g] for g in groups])
    def cons(g):
        dd = d[(d.group == g) & (d.forget_gap <= 0)]; return dd.loc[dd.val_logloss.idxmin()].group_auroc - ex[g]
    v2 = np.array([cons(g) for g in groups])
    for key, v in [(f"AUROC | {m} removal-matched (forget gap closest to zero) minus exact group", v1), (f"AUROC | {m} best with forget gap <= 0 minus exact group", v2)]:
        r = np.random.RandomState(0); bs = [float(v[r.choice(len(v), len(v), True)].mean()) for _ in range(5000)]
        res[key] = dict(d=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], lower_in=int((v < 0).sum()), n_groups=len(v), n_boot=5000); boots[key] = bs
res = boot_draws.attach(res, boots, f"{D}/matched_draws.npz")
json.dump(dict(results=res), open(f"{D}/matched.json", "w"), indent=1)
for k, v in res.items(): print(f"{k:85s} {v['d']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p_exact']:.4f} lower in {v['lower_in']} of {v['n_groups']}")
