"""Figures 1 and 2 for NEJM AI draft v4. Reads aggregate result files under ~/mimic/results and the cached v13.1 eICU predictions
(for per-consortium AUROC with the fixed adapter). Writes Figure1_v4.{pdf,png} and Figure2_v4.{pdf,png} plus figure_data_v4.csv
(every plotted value). Run: ROOT=~/mimic python3 figures_v4.py
"""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); RES = f"{ROOT}/results"; OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b", "xtick.major.width": 0.6, "ytick.major.width": 0.6})
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
MM = 1 / 25.4; rows = []
def gboot(v, n=5000, seed=0):
    v = np.asarray(v, float); r = np.random.RandomState(seed); b = [r.choice(v, len(v), True).mean() for _ in range(n)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def keep(fig, panel, label, value, lo=np.nan, hi=np.nan, source=""): rows.append(dict(figure=fig, panel=panel, label=label, value=value, lo=lo, hi=hi, source=source))
def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
I = json.load(open(f"{RES}/v13_intervals_1/intervals.json"))["results"]; MF = json.load(open(f"{RES}/v13_intervals_1/matched_flags.json"))["results"]
X = json.load(open(f"{RES}/v13_intervals_1/extra_contrasts.json"))
def iv(key, src=I):
    k = [k for k in src if k.endswith(key) or key in k]; assert len(k) == 1, (key, k); v = src[k[0]]
    d = v.get("d", v.get("point")); return d, v["ci"][0], v["ci"][1]
# ---------------------------------------------------------------- data
v5 = pd.read_csv(f"{RES}/v5_partitions/results.csv"); W5 = v5.pivot_table(index=["family", "group"], columns="model", values="auroc")
pair5 = (W5["exact_group"] - W5["exact_random_pat"]).rename("d").reset_index()
clu = pair5[pair5.family == "community_trainfit"]; cen = pair5[pair5.family != "community_trainfit"]
iso = v5[(v5.model == "original") & (v5.family == "community_trainfit")].groupby("group").isolation_ratio.mean()
E7 = pd.read_csv(f"{RES}/v7_eicu_consortium_es/summary.csv").pivot_table(index=["family", "group"], columns="model", values="auroc")
E12 = pd.read_csv(f"{RES}/v12_eicu_federated_es/summary.csv").pivot_table(index=["family", "group"], columns="model", values="auroc")
cons_c = (E7["exact_group"] - E7["exact_random"]).rename("central"); cons_f = (E12["fl_without"] - E12["fl_random"]).rename("federated")
cons = pd.concat([cons_c, cons_f], axis=1).reset_index()
V11 = pd.read_csv(f"{RES}/v11_confinement_1/results.csv"); V11 = V11[V11.family == "community_trainfit"]
A11 = V11.groupby(["group", "name"]).d_auroc_vs_original.mean().unstack()
B14 = pd.read_csv(f"{RES}/v14b_pooled_adapt/summary.csv").pivot_table(index="community", columns="arm", values="auroc")
iso_set = json.load(open(f"{RES}/v14_commonbase_1/sets_B.json"))["most_isolated"]
# eICU per-consortium AUROC with the fixed adapter, from cached predictions
df = pd.read_parquet(f"{ROOT}/derived/eicu_cohort.parquet", columns=["uniquepid", "hospitalid", "died"])
rng = np.random.RandomState(0); ups = df.uniquepid.unique(); rng.shuffle(ups); test_p = set(ups[: int(0.2 * len(ups))])
is_test = df.uniquepid.isin(test_p).values; y_te = df.died.values[is_test].astype(int); h_te = df.hospitalid.values[is_test]
P = np.load(f"{RES}/v13_intervals_1/preds_eicu.npz", allow_pickle=True)["P"].item(); mem = json.load(open(f"{RES}/v13_intervals_1/membership_eicu.json"))
fe = {}
for k, hs in mem.items():
    m = np.isin(h_te, hs); a = {arm: np.mean([auc(y_te[m], P[(k, s, arm)][m]) for s in range(3)]) for arm in ["fl_global", "fl_without", "fl_resid", "fl_resid_random"]}
    fe[k] = {arm: a[arm] - a["fl_global"] for arm in ["fl_without", "fl_resid_random", "fl_resid"]}
fe = pd.DataFrame(fe).T
# ---------------------------------------------------------------- Figure 1
fig = plt.figure(figsize=(180 * MM, 125 * MM)); gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15], hspace=0.55, wspace=0.55)
ax = fig.add_subplot(gs[0, :]); r = np.random.RandomState(3)
blocks = [("Clustered groups, MIMIC-IV (n=25)", clu.d.values, iv("MIMIC communities | base_vs_random (structured minus random withdrawal)"), "two-stage bootstrap"),
          ("Census, insurance and era groups, MIMIC-IV (n=%d)" % len(cen), cen.d.values, gboot(cen.d.values), "bootstrap over groups"),
          ("Hospital consortia, eICU (n=10)", cons.central.values, iv("eICU | all | exact_group_vs_exact_random (centralised penalty)"), "hospital-cluster bootstrap")]
for i, (lab, vals, (d, lo, hi), how) in enumerate(blocks):
    yv = 2 - i; ax.scatter(vals, yv + r.uniform(-0.18, 0.18, len(vals)), s=9, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
    ax.plot([lo, hi], [yv - 0.32, yv - 0.32], color=INK, lw=1.2, solid_capstyle="round"); ax.scatter([d], [yv - 0.32], marker="D", s=16, color=INK, zorder=3)
    ax.text(hi + 0.0015, yv - 0.32, f"{d:+.4f} ({lo:+.4f} to {hi:+.4f})", va="center", fontsize=6, color="#52514e")
    keep(1, "A", lab + " mean", d, lo, hi, how); [keep(1, "A", f"{lab} point", v) for v in vals]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks([2, 1, 0]); ax.set_yticklabels([b[0] for b in blocks]); ax.set_ylim(-0.7, 2.5)
ax.set_xlabel("Change in the group's held-out AUROC, group withdrawal minus matched random removal"); ax.set_title("A", loc="left", fontweight="bold")
ax2 = fig.add_subplot(gs[1, 0])
mf = [("MIMIC-IV, retrained without the group", iv("MIMIC communities | missed events per 1,000 at matched flag rate | base minus original", MF), BLUE, "o"),
      ("eICU centralised, without the consortium", iv("eICU | all | missed events per 1,000 at matched flag rate | exact_group minus original", MF), BLUE, "o"),
      ("eICU centralised, random removal", iv("eICU | all | missed events per 1,000 at matched flag rate | exact_random minus original", MF), ORANGE, "s"),
      ("eICU federation, without the consortium", iv("eICU | all | missed events per 1,000 at matched flag rate | fl_without minus fl_global", MF), BLUE, "o"),
      ("eICU federation, random removal", iv("eICU | all | missed events per 1,000 at matched flag rate | fl_random minus fl_global", MF), ORANGE, "s")]
for i, (lab, (d, lo, hi), c, mk) in enumerate(mf):
    yv = len(mf) - 1 - i; ax2.plot([lo, hi], [yv, yv], color=c, lw=1.4, solid_capstyle="round"); ax2.scatter([d], [yv], marker=mk, s=18, color=c, zorder=3, edgecolors="white", linewidths=0.6)
    keep(1, "B", lab, d, lo, hi, "matched_flags.json")
ax2.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax2.set_yticks(range(len(mf))[::-1]); ax2.set_yticklabels([m[0] for m in mf])
ax2.set_xlabel("Events unflagged per 1,000, change from\nthe pooled model or full federation"); ax2.set_title("B", loc="left", fontweight="bold")
ax3 = fig.add_subplot(gs[1, 1]); cs = cons.sort_values("central"); yy = np.arange(len(cs))
FAM = {"region": "Region", "teaching": "Teaching", "bedsize": "Beds"}
lab = [f"{FAM[f]} " + g.replace("_-_", " to ").replace(">=_", "\u2265").replace("<", "<").replace("_", " ") for f, g in zip(cs.family, cs.group)]
lab = [l.replace("Teaching f", "Non-teaching").replace("Teaching t", "Teaching") for l in lab]
for j, (c, f) in enumerate(zip(cs.central, cs.federated)): ax3.plot([c, f], [j, j], color="#d9d8d3", lw=1.2, zorder=1)
ax3.scatter(cs.central, yy, s=16, color=BLUE, label="Centralised model", zorder=2, edgecolors="white", linewidths=0.5)
ax3.scatter(cs.federated, yy, s=16, color=AQUA, marker="^", label="Federation", zorder=2, edgecolors="white", linewidths=0.5)
for l, c, f in zip(lab, cs.central, cs.federated): keep(1, "C", l + " centralised", c); keep(1, "C", l + " federated", f)
ax3.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax3.set_yticks(yy); ax3.set_yticklabels(lab); ax3.set_xlabel("Change in AUROC, consortium withdrawal\nminus matched random removal")
ax3.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2); ax3.set_title("C", loc="left", fontweight="bold", pad=14)
fig.savefig(f"{OUT}/Figure1_v4.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/Figure1_v4.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------------------------------------------------------- Figure 2
fig = plt.figure(figsize=(180 * MM, 120 * MM)); gs = GridSpec(2, 2, figure=fig, hspace=0.6, wspace=0.62)
ax = fig.add_subplot(gs[0, 0]); r = np.random.RandomState(4)
arms = [("Base (state 3)", "base", GREY, "o"), ("Base + control component", "resid_random", ORANGE, "s"), ("Base + residual component", "resid_group@1", BLUE, "o"),
        ("Base + representation head", "rep_group@1", BLUE, "^"), ("Base + fine-tuned copy", "full_group@1", BLUE, "D")]
for i, (lab, col, c, mk) in enumerate(arms):
    vals = A11[col].values; yv = len(arms) - 1 - i; d, lo, hi = gboot(vals)
    ax.scatter(vals, yv + r.uniform(-0.2, 0.2, len(vals)), s=7, color=c, alpha=0.45, linewidths=0, marker=mk)
    ax.plot([lo, hi], [yv - 0.36, yv - 0.36], color=INK, lw=1.1); ax.scatter([d], [yv - 0.36], s=13, color=INK, marker="D", zorder=3)
    keep(2, "A", lab + " mean", d, lo, hi, "v11_confinement_1, bootstrap over groups"); [keep(2, "A", lab + " point", v) for v in vals]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks(range(len(arms))[::-1]); ax.set_yticklabels([a[0] for a in arms])
ax.set_xlabel("Change in AUROC relative to the pooled model"); ax.set_title("A  MIMIC-IV, 25 clustered groups", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[0, 1]); arms_e = [("Federation without the\nconsortium (state 3)", "fl_without", GREY, "o"), ("+ control component", "fl_resid_random", ORANGE, "s"), ("+ consortium component", "fl_resid", BLUE, "o")]
for i, (lab, col, c, mk) in enumerate(arms_e):
    vals = fe[col].values; yv = len(arms_e) - 1 - i; d, lo, hi = gboot(vals)
    ax.scatter(vals, yv + r.uniform(-0.18, 0.18, len(vals)), s=10, color=c, alpha=0.6, linewidths=0, marker=mk)
    ax.plot([lo, hi], [yv - 0.33, yv - 0.33], color=INK, lw=1.1); ax.scatter([d], [yv - 0.33], s=13, color=INK, marker="D", zorder=3)
    keep(2, "B", lab.replace("\n", " ") + " mean", d, lo, hi, "preds_eicu (v13.1), bootstrap over consortia"); [keep(2, "B", lab.replace("\n", " ") + " point", v) for v in vals]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks(range(len(arms_e))[::-1]); ax.set_yticklabels([a[0] for a in arms_e])
ax.set_xlabel("Change in AUROC relative to the full federation"); ax.set_title("B  eICU federation, 10 consortia", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[1, 0]); fr = [0.1, 0.25, 0.5, 1.0]
for lab, pre, c, mk in [("Residual component", "resid_group", BLUE, "o"), ("Representation head", "rep_group", AQUA, "^"), ("Fine-tuned copy", "full_group", ORANGE, "D")]:
    st = [gboot(A11[f"{pre}@{f:g}"].values) for f in fr]; m = [s[0] for s in st]
    if pre == "resid_group": ax.fill_between(fr, [s[1] for s in st], [s[2] for s in st], color=c, alpha=0.12, linewidth=0)
    ax.plot(fr, m, color=c, lw=1.4, marker=mk, ms=4, label=lab)
    for f, s in zip(fr, st): keep(2, "C", f"{lab} at {f:g}", *s, source="v11_confinement_1, bootstrap over groups")
b = A11["base"].mean(); ax.axhline(b, color=GREY, lw=0.8); ax.text(1.0, b - 0.0012, "Base alone", ha="right", va="top", fontsize=6, color="#52514e")
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.text(0.1, 0.0008, "Pooled model", fontsize=6, color="#52514e")
ax.set_xscale("log"); ax.set_xticks(fr); ax.set_xticklabels(["10%", "25%", "50%", "100%"]); ax.minorticks_off()
ax.set_xlabel("Share of the group's training patients used by its component"); ax.set_ylabel("Change in AUROC relative\nto the pooled model")
ax.legend(frameon=False, loc="upper left", handletextpad=0.3); ax.set_title("C  Data-size curve", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[1, 1]); dd = (B14["single_base_resid"] - B14["pooled_resid"]); xi = iso.reindex(dd.index)
mk = dd.index.isin(iso_set)
ax.scatter(xi[~mk], dd[~mk], s=14, facecolors="none", edgecolors=BLUE, linewidths=0.8, label="Other groups")
ax.scatter(xi[mk], dd[mk], s=16, color=BLUE, label="Five most isolated groups")
for g in dd.index: keep(2, "D", g, dd[g], source=f"v14b summary; isolation ratio {xi[g]:.3f}")
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_xlabel("Isolation ratio of the group in feature space")
ax.set_ylabel("AUROC, base + component minus\npooled model + same component"); ax.legend(frameon=False, loc="upper right", handletextpad=0.3)
ax.set_title("D  Same component on both models", loc="left", fontweight="bold")
fig.savefig(f"{OUT}/Figure2_v4.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/Figure2_v4.png", dpi=300, bbox_inches="tight"); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{OUT}/figure_data_v4.csv", index=False); print("written", OUT, len(rows), "values")
