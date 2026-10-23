"""Figures 1 and 2 for NEJM AI draft v5. Reads aggregate result files under ~/mimic/results (v5, v7.2, v12.2, v13.1, v13b, v16, v17,
v18, v19, v20) and the cached v16/v19 MIMIC predictions (for a two-stage bootstrap of the control component). Writes
Figure1_v5.{pdf,png}, Figure2_v5.{pdf,png} and figure_data_v5.csv (every plotted value). Run: ROOT=~/mimic python3 figures_v5.py
"""
import os, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); RES = f"{ROOT}/results"; OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b", "xtick.major.width": 0.6, "ytick.major.width": 0.6})
BLUE, ORANGE, AQUA, GREY, INK, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#d9d8d3"
MM = 1 / 25.4; rows = []
def gboot(v, n=5000, seed=0):
    v = np.asarray(v, float); r = np.random.RandomState(seed); b = [r.choice(v, len(v), True).mean() for _ in range(n)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def keep(fig, panel, label, value, lo=np.nan, hi=np.nan, source=""): rows.append(dict(figure=fig, panel=panel, label=label, value=value, lo=lo, hi=hi, source=source))
def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def iv(src, key):
    k = [key] if key in src else [k for k in src if k.endswith(key)]; assert len(k) == 1, (key, k); v = src[k[0]]
    return v.get("d", v.get("point")), v["ci"][0], v["ci"][1]
I13 = json.load(open(f"{RES}/v13_intervals_1/intervals.json"))["results"]
X18 = json.load(open(f"{RES}/v18_landmark/all_contrasts.json"))["results"]; MF13 = json.load(open(f"{RES}/v13_intervals_1/matched_flags.json"))["results"]
I19 = json.load(open(f"{RES}/v19_protocol/intervals.json")); I17 = json.load(open(f"{RES}/v17_crossfit/intervals.json"))["results"]
I16 = json.load(open(f"{RES}/v16_strict/intervals.json"))["results"]
V20 = f"{RES}/v20_eicu_protocol/intervals.json"; I20 = json.load(open(V20))["all"]["results"] if os.path.exists(V20) else None
I24 = json.load(open(f"{RES}/v24_symmetric_mimic/intervals.json"))["all"]["results"]; I26 = json.load(open(f"{RES}/v26_eicu_table2/all.json")); I23 = json.load(open(f"{RES}/v23_eicu_strict/intervals.json"))["all"]["results"]
S24 = pd.read_csv(f"{RES}/v24_symmetric_mimic/summary.csv").pivot_table(index="group", columns="arm", values="auroc")
# ---------------------------------------------------------------- data for Figure 1
v5 = pd.read_csv(f"{RES}/v5_partitions/results.csv"); W5 = v5.pivot_table(index=["family", "group"], columns="model", values="auroc")
pair5 = (W5["exact_group"] - W5["exact_random_pat"]).rename("d").reset_index()
clu = pair5[pair5.family == "community_trainfit"]; cen = pair5[pair5.family != "community_trainfit"]
iso = v5[(v5.model == "original") & (v5.family == "community_trainfit")].groupby("group").isolation_ratio.mean()
E7 = pd.read_csv(f"{RES}/v7_eicu_consortium_es/summary.csv").pivot_table(index=["family", "group"], columns="model", values="auroc")
E12 = pd.read_csv(f"{RES}/v12_eicu_federated_es/summary.csv").pivot_table(index=["family", "group"], columns="model", values="auroc")
cons = pd.concat([(E7["exact_group"] - E7["exact_random"]).rename("central"), (E12["fl_without"] - E12["fl_random"]).rename("federated")], axis=1).reset_index()
# ---------------------------------------------------------------- Figure 1
fig = plt.figure(figsize=(180 * MM, 125 * MM)); gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15], hspace=0.55, wspace=0.55)
ax = fig.add_subplot(gs[0, :]); r = np.random.RandomState(3)
blocks = [("Clustered groups, MIMIC-IV (n=25)", clu.d.values, iv(I13, "MIMIC communities | base_vs_random (structured minus random withdrawal)"), "two-stage bootstrap"),
          ("Census, insurance and era groups, MIMIC-IV (n=%d)" % len(cen), cen.d.values, gboot(cen.d.values), "bootstrap over groups"),
          ("Hospital consortia, eICU (n=10)", cons.central.values, iv(I13, "eICU | all | exact_group_vs_exact_random (centralised penalty)"), "hospital-cluster bootstrap")]
for i, (lab, vals, (d, lo, hi), how) in enumerate(blocks):
    yv = 2 - i; ax.scatter(vals, yv + r.uniform(-0.18, 0.18, len(vals)), s=9, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
    ax.plot([lo, hi], [yv - 0.32, yv - 0.32], color=INK, lw=1.2, solid_capstyle="round"); ax.scatter([d], [yv - 0.32], marker="D", s=16, color=INK, zorder=3)
    ax.text(hi + 0.0015, yv - 0.32, f"{d:+.4f} ({lo:+.4f} to {hi:+.4f})", va="center", fontsize=6, color="#52514e")
    keep(1, "A", lab + " mean", d, lo, hi, how); [keep(1, "A", f"{lab} point", v) for v in vals]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks([2, 1, 0]); ax.set_yticklabels([b[0] for b in blocks]); ax.set_ylim(-0.7, 2.5)
ax.set_xlabel("Change in the group's held-out AUROC, group withdrawal minus matched random removal"); ax.set_title("A", loc="left", fontweight="bold")
ax2 = fig.add_subplot(gs[1, 0])
mf = [("MIMIC-IV, retrained\nwithout the group", iv(MF13, "MIMIC communities | missed events per 1,000 at matched flag rate | base minus random"), iv(MF13, "MIMIC communities | missed events per 1,000 at matched flag rate | base minus original"), "v13b matched_flags.json"),
      ("eICU centralised,\nwithout the consortium", iv(X18, "events unflagged per 1,000 at matched flag rate | exact_group minus exact_random"), iv(X18, "events unflagged per 1,000 at matched flag rate | exact_group minus original"), "v18 all_contrasts.json"),
      ("eICU federation,\nwithout the consortium", iv(X18, "events unflagged per 1,000 at matched flag rate | fl_without minus fl_random"), iv(X18, "events unflagged per 1,000 at matched flag rate | fl_without minus fl_global"), "v18 all_contrasts.json")]
for i, (lab, (d, lo, hi), (d2, lo2, hi2), src) in enumerate(mf):
    yv = len(mf) - 1 - i
    ax2.plot([lo2, hi2], [yv + 0.18, yv + 0.18], color=LIGHT, lw=1.4, solid_capstyle="round"); ax2.scatter([d2], [yv + 0.18], marker="s", s=16, color=GREY, zorder=3, edgecolors="white", linewidths=0.6, label="Change from the pooled model or full federation" if i == 0 else None)
    ax2.plot([lo, hi], [yv - 0.18, yv - 0.18], color=BLUE, lw=1.4, solid_capstyle="round"); ax2.scatter([d], [yv - 0.18], marker="o", s=18, color=BLUE, zorder=3, edgecolors="white", linewidths=0.6, label="Group withdrawal minus matched random removal" if i == 0 else None)
    keep(1, "B", lab.replace("\n", " ") + ", group minus random", d, lo, hi, src); keep(1, "B", lab.replace("\n", " ") + ", change from pooled", d2, lo2, hi2, src)
ax2.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax2.set_yticks(range(len(mf))[::-1]); ax2.set_yticklabels([m[0] for m in mf]); ax2.set_ylim(-0.6, len(mf) - 0.4)
ax2.set_xlabel("Events unflagged per 1,000 admissions or stays\nat a matched flag rate"); ax2.set_title("B", loc="left", fontweight="bold", pad=18)
ax2.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
ax3 = fig.add_subplot(gs[1, 1]); cs = cons.sort_values("central"); yy = np.arange(len(cs))
FAM = {"region": "Region", "teaching": "Teaching", "bedsize": "Beds"}
lab = [f"{FAM[f]} " + g.replace("_-_", " to ").replace(">=_", "≥").replace("_", " ") for f, g in zip(cs.family, cs.group)]
lab = [l.replace("Teaching f", "Non-teaching").replace("Teaching t", "Teaching") for l in lab]
for j, (c, f) in enumerate(zip(cs.central, cs.federated)): ax3.plot([c, f], [j, j], color=LIGHT, lw=1.2, zorder=1)
ax3.scatter(cs.central, yy, s=16, color=BLUE, label="Centralised model", zorder=2, edgecolors="white", linewidths=0.5)
ax3.scatter(cs.federated, yy, s=16, color=AQUA, marker="^", label="Federation", zorder=2, edgecolors="white", linewidths=0.5)
for l, c, f in zip(lab, cs.central, cs.federated): keep(1, "C", l + " centralised", c); keep(1, "C", l + " federated", f)
ax3.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax3.set_yticks(yy); ax3.set_yticklabels(lab); ax3.set_xlabel("Change in AUROC, consortium withdrawal\nminus matched random removal")
ax3.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2); ax3.set_title("C", loc="left", fontweight="bold", pad=14)
fig.savefig(f"{OUT}/Figure1_v5.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/Figure1_v5.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------------------------------------------------------- data for Figure 2
S19 = pd.read_csv(f"{RES}/v19_protocol/summary.csv"); S19 = S19[S19.group.str.startswith("community_trainfit")].pivot_table(index="group", columns="arm", values="auroc")
S16 = pd.read_csv(f"{RES}/v16_strict/summary.csv"); S16 = S16[S16.group.str.startswith("community_trainfit")].pivot_table(index="group", columns="arm", values="auroc")
groups = list(S19.index); assert len(groups) == 25
# control component relative to the pooled model: two-stage bootstrap (groups, then patients) from the aligned v16/v19 prediction caches
cache = f"{OUT}/figure2_control_boot.json"
if os.path.exists(cache): ctrl = json.load(open(cache))
else:
    P = {}
    for g in groups:
        for s in range(3):
            z19 = np.load(f"{RES}/v19_protocol/preds/{g.replace('/', '__')}__s{s}.npz"); z16 = np.load(f"{RES}/v16_strict/preds/{g.replace('/', '__')}__s{s}.npz")
            assert np.array_equal(z19["pid"], z16["pid"]); P[(g, s)] = dict(y=z19["y"].astype(int), pid=z19["pid"], pooled=z19["pooled"], control=z16["strict_control"])
    def stat(g, idx): return np.mean([auc(P[(g, s)]["y"][idx], P[(g, s)]["control"][idx]) - auc(P[(g, s)]["y"][idx], P[(g, s)]["pooled"][idx]) for s in range(3)])
    pat = {}
    for g in groups:
        u, inv = np.unique(P[(g, 0)]["pid"], return_inverse=True); pat[g] = [np.where(inv == i)[0] for i in range(len(u))]
    pt = np.mean([stat(g, np.arange(len(P[(g, 0)]["y"]))) for g in groups]); r = np.random.RandomState(0); b = []
    for _ in range(1000):
        b.append(np.mean([stat(g, np.concatenate([pat[g][j] for j in r.choice(len(pat[g]), len(pat[g]), True)])) for g in r.choice(groups, len(groups), True)]))
    ctrl = dict(d=float(pt), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], n_boot=1000); json.dump(ctrl, open(cache, "w"))
ISO = ["community_trainfit/" + c for c in json.load(open(f"{RES}/v14_commonbase_1/sets_B.json"))["most_isolated"]]
chosen = {}
for g in groups:
    ch = [json.load(open(f"{RES}/v19_protocol/runs/{g.replace('/', '__')}__s{s}.json"))["selection"]["chosen"] for s in range(3)]
    top = sorted(set(ch), key=lambda k: (-ch.count(k), k)); chosen[g] = top[0] if ch.count(top[0]) >= 2 else "mixed"
# ---------------------------------------------------------------- Figure 2
fig = plt.figure(figsize=(180 * MM, 125 * MM)); gs = GridSpec(2, 2, figure=fig, hspace=0.85, wspace=0.5, width_ratios=[1.35, 1])
ax = fig.add_subplot(gs[0, 0]); r = np.random.RandomState(4)
arms = [("Base alone (state 3)", S19["strict_base"] - S19["pooled"], iv(I19["all"]["results"], "strict base minus pooled"), GREY, "o", "v19 intervals.json"),
        ("Base + control component", S16["strict_control"] - S16["pooled"], (ctrl["d"], ctrl["ci"][0], ctrl["ci"][1]), ORANGE, "s", "two-stage bootstrap of v16 predictions"),
        ("Base + chosen component (state 2)", S19["chosen"] - S19["pooled"], iv(I19["all"]["results"], "chosen component minus pooled"), BLUE, "o", "v19 intervals.json"),
        ("Adapted pooled comparator", S24["pooled_cf_chosen"] - S24["pooled"], iv(I24, "pooled with chosen cross-fitted component minus pooled"), AQUA, "^", "v24 intervals.json")]
for i, (lab, vals, (d, lo, hi), c, mk, src) in enumerate(arms):
    yv = len(arms) - 1 - i; ax.scatter(vals.values, yv + r.uniform(-0.2, 0.2, len(vals)), s=7, color=c, alpha=0.45, linewidths=0, marker=mk)
    ax.plot([lo, hi], [yv - 0.36, yv - 0.36], color=INK, lw=1.1); ax.scatter([d], [yv - 0.36], s=13, color=INK, marker="D", zorder=3)
    keep(2, "A", lab + " mean", d, lo, hi, src); [keep(2, "A", f"{lab} point {g}", v) for g, v in vals.items()]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks(range(len(arms))[::-1]); ax.set_yticklabels([a[0] for a in arms])
ax.set_xlabel("Change in AUROC relative to the pooled model"); ax.set_title("A  MIMIC-IV, 25 clustered groups", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[0, 1])
if I20 is not None:
    S20 = pd.read_csv(f"{RES}/v23_eicu_strict/summary.csv").pivot_table(index="consortium", columns="arm", values="auroc")
    S25 = pd.read_csv(f"{RES}/v25_symmetric_eicu/summary.csv").pivot_table(index="consortium", columns="arm", values="auroc"); S20["pooled_cf"] = S25["pooled_cf_chosen"]
    base_iv = iv(I26, "AUROC | strict base minus pooled"); chosen_iv = iv(I26, "AUROC | strict confined minus pooled"); cf_iv = iv(I26, "AUROC | symmetric comparator minus pooled")
    cb = iv(I23, "control minus strict base"); ctrl_iv = (base_iv[0] + cb[0], np.nan, np.nan)
    arms_e = [("Base alone (state 3)", "base", base_iv, GREY, "o"), ("Base + control component", "control", ctrl_iv, ORANGE, "s"),
              ("Base + chosen component (state 2)", "chosen", chosen_iv, BLUE, "o"), ("Adapted pooled comparator", "pooled_cf", cf_iv, AQUA, "^")]
    for i, (lab, col, (d, lo, hi), c, mk) in enumerate(arms_e):
        yv = len(arms_e) - 1 - i
        if S20 is not None and col in S20 and "pooled" in S20:
            vals = (S20[col] - S20["pooled"]).values; ax.scatter(vals, yv + r.uniform(-0.18, 0.18, len(vals)), s=10, color=c, alpha=0.6, linewidths=0, marker=mk); [keep(2, "B", f"{lab} point {g}", v) for g, v in zip(S20.index, vals)]
        if np.isfinite(lo): ax.plot([lo, hi], [yv - 0.33, yv - 0.33], color=INK, lw=1.1)
        ax.scatter([d], [yv - 0.33], s=13, color=INK, marker="D", zorder=3); keep(2, "B", lab + " mean", d, lo, hi, "v23/v25/v26")
    ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks(range(len(arms_e))[::-1]); ax.set_yticklabels([])
    ax.set_xlabel("Change in AUROC relative to the pooled model")
else:
    ax.text(0.5, 0.5, "v20 pending", ha="center", va="center", transform=ax.transAxes, color=GREY); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("B  eICU centralised model, 10 consortia", loc="left", fontweight="bold")
# C: schematic of the three permission states (one row per state)
ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.set_title("C  The three permission states", loc="left", fontweight="bold")
def box(x, y, w, h, text, fc="white", ec=INK, ls="-", fs=5.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)
ax.text(26, 99, "Shared model", ha="center", va="top", fontsize=6, fontweight="bold"); ax.text(78, 99, "Group's component", ha="center", va="top", fontsize=6, fontweight="bold")
rowsC = [(64, "State 1, pooled", "Trained on the pool\nand the group's data", "#e3edf9", BLUE, "None", "white", GREY, (0, (2, 2))),
         (34, "State 2, confined", "Base trained on\nthe pool only\n(provenance, checksum)", "#f1f1ef", GREY, "Fitted on the group's\ndata, type chosen by\nvalidation, applied to\nthe group's patients only", "#e3edf9", BLUE, "-"),
         (4, "State 3, withdrawn", "Base unchanged\n(same checksum)", "#f1f1ef", GREY, "Deleted", "white", GREY, (0, (2, 2)))]
for y, lab, t1, f1, e1, t2, f2, e2, ls2 in rowsC:
    ax.text(8, y + 21.5, lab, ha="left", va="bottom", fontsize=5.8, fontweight="bold")
    box(8, y, 36, 20, t1, fc=f1, ec=e1, fs=5); box(60, y, 36, 20, t2, fc=f2, ec=e2, ls=ls2, fs=5)
ax.annotate("", xy=(78, 25), xytext=(78, 33), arrowprops=dict(arrowstyle="->", lw=0.8, color=INK)); ax.text(80, 29, "withdraw: delete", ha="left", va="center", fontsize=5)
ax.annotate("", xy=(52, 14), xytext=(52, 74), arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
ax.text(52, 44, "withdraw:\nretrain", ha="center", va="center", fontsize=5, bbox=dict(fc="white", ec="none", pad=1.5))
# D: confined minus best pooled against isolation, with the chosen type
ax = fig.add_subplot(gs[1, 1]); dd = S24["confined_chosen"] - S24["pooled_cf_chosen"]; xi = iso.reindex([g.split("/")[1] for g in dd.index]).values
MK = {"residual": ("o", "Residual component"), "head": ("^", "Representation head"), "finetuned": ("D", "Fine-tuned copy"), "mixed": ("X", "No type chosen in 2 of 3 seeds")}
for kind, (mk, lab) in MK.items():
    m = np.array([chosen[g] == kind for g in dd.index]); iso_m = np.array([g in ISO for g in dd.index])
    ax.scatter(xi[m & ~iso_m], dd.values[m & ~iso_m], s=16, facecolors="none", edgecolors=BLUE, linewidths=0.8, marker=mk, label=lab)
    ax.scatter(xi[m & iso_m], dd.values[m & iso_m], s=18, color=BLUE, marker=mk)
ax.scatter([], [], s=18, color=BLUE, marker="s", label="Filled: five most isolated groups")
for g, x, v in zip(dd.index, xi, dd.values): keep(2, "D", g, v, source=f"v24 summary; isolation ratio {x:.3f}; chosen {chosen[g]}")
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_xlabel("Isolation ratio of the group in feature space")
ax.set_ylabel("AUROC, base + chosen component\nminus adapted pooled comparator"); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.3, labelspacing=0.2, columnspacing=0.8, borderaxespad=0.1)
ax.set_title("D  Confined minus adapted pooled comparator", loc="left", fontweight="bold", pad=36)
fig.savefig(f"{OUT}/Figure2_v5.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/Figure2_v5.png", dpi=300, bbox_inches="tight"); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{OUT}/figure_data_v5.csv", index=False); print("written", OUT, len(rows), "values; v20", "present" if I20 else "pending")
