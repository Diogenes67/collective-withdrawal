"""Figures 1, 2 and S1 for NEJM AI v9. Aggregate result files only (no patient-level data).
Figure 1: A, excess loss per group (clustered MIMIC-IV groups, census and era groups with the white group marked and excluded from
the summary, eICU consortia among stays still in the unit at 24 hours); B, events unflagged at matched alert counts (eICU landmark);
C, per-consortium centralised and federated excess loss (landmark); D, excess loss against the equal-pool isolation ratio, fit over
the 48 groups with a matched control. Figure 2: A, MIMIC-IV permission states; B, eICU permission states (landmark); C, schematic;
D, confined minus adapted pooled comparator against the equal-pool ratio. Figure S1: Figure 1D with the isolation ratio of
Supplementary Note 2. Writes Figure1_v9, Figure2_v9, FigureS1_v9 (.pdf, .png) and figure_data_v9.csv to OUT.
Run: ROOT=~/mimic OUT=~/mimic/nejmai_build/v9_figures python3 figures_v9.py
"""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
ROOT = os.path.expanduser(os.environ.get("ROOT", "~/mimic")); RES = f"{ROOT}/results"
OUT = os.path.expanduser(os.environ.get("OUT", f"{ROOT}/nejmai_build/v9_figures")); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.unicode_minus": True, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b", "xtick.major.width": 0.6, "ytick.major.width": 0.6})
BLUE, ORANGE, AQUA, GREY, INK, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#d9d8d3"
MM = 1 / 25.4; rows = []
def fmt(v): return f"{v:+.4f}".replace("-", "\u2212")
def gboot(v, n=5000, seed=0):
    v = np.asarray(v, float); r = np.random.RandomState(seed); b = [r.choice(v, len(v), True).mean() for _ in range(n)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def keep(fig, panel, label, value, lo=np.nan, hi=np.nan, source=""): rows.append(dict(figure=fig, panel=panel, label=label, value=value, lo=lo, hi=hi, source=source))
def iv(src, key):
    k = [key] if key in src else [k for k in src if k.endswith(key)]; assert len(k) == 1, (key, k); v = src[k[0]]
    return v.get("d", v.get("point")), v["ci"][0], v["ci"][1]
FAM = {"region": "Region", "teaching": "Teaching", "bedsize": "Beds"}
def cons_label(f, g):
    l = f"{FAM[f]} " + g.replace("_-_", " to ").replace(">=_", "≥").replace("_", " ")
    return l.replace("Teaching f", "Non-teaching").replace("Teaching t", "Teaching")
I13 = json.load(open(f"{RES}/v13_intervals_1/intervals.json"))["results"]; MF13 = json.load(open(f"{RES}/v13_intervals_1/matched_flags.json"))["results"]
L35 = json.load(open(f"{RES}/v35_landmark/contrasts_landmark.json"))["results"]
PC = pd.read_csv(f"{RES}/v35_landmark/per_consortium.csv"); PL = pd.read_csv(f"{RES}/v35_landmark/per_consortium_long.csv"); PL = PL[PL.population == "landmark"].set_index("consortium")
ISOD = pd.read_csv(f"{RES}/v34_isolation/figure_data.csv")
v5 = pd.read_csv(f"{RES}/v5_partitions/results.csv"); W5 = v5.pivot_table(index=["family", "group"], columns="model", values="auroc")
pair5 = (W5["exact_group"] - W5["exact_random_pat"]).rename("d").reset_index()
clu = pair5[pair5.family == "community_trainfit"]; cen = pair5[pair5.family != "community_trainfit"]
white = (cen.family == "race") & (cen.group == "white"); cen_m = cen[~white]

# ================================================================ Figure 1
fig = plt.figure(figsize=(180 * MM, 150 * MM)); gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.25], hspace=0.6, wspace=0.75)
ax = fig.add_subplot(gs[0, :]); r = np.random.RandomState(3)
blocks = [("Clustered groups,\nMIMIC-IV (n=25)", clu.d.values, iv(I13, "MIMIC communities | base_vs_random (structured minus random withdrawal)"), "two-stage bootstrap (v13)"),
          (f"Census, insurance and era\ngroups, MIMIC-IV (n={len(cen_m)})", cen_m.d.values, gboot(cen_m.d.values), "bootstrap over groups; white group excluded"),
          ("eICU hospital consortia, in\nthe unit at 24 h (n=10)", PC.penalty_landmark.values, iv(L35, "AUROC | centralised penalty: retrained without the consortium minus random removal"), "hospital-cluster bootstrap (v35)")]
for i, (lab, vals, (d, lo, hi), how) in enumerate(blocks):
    yv = 2 - i; ax.scatter(vals, yv + r.uniform(-0.18, 0.18, len(vals)), s=9, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
    ax.plot([lo, hi], [yv - 0.32, yv - 0.32], color=INK, lw=1.2, solid_capstyle="round"); ax.scatter([d], [yv - 0.32], marker="D", s=16, color=INK, zorder=3)
    ax.text(1.01, yv - 0.32, f"{fmt(d)}\n({fmt(lo)} to {fmt(hi)})", va="center", ha="left", fontsize=6.5, color="#52514e", transform=mtrans.blended_transform_factory(ax.transAxes, ax.transData))
    lab = lab.replace("\n", " "); keep(1, "A", lab + " mean", d, lo, hi, how); [keep(1, "A", f"{lab} point", v) for v in vals]
wv = float(cen[white].d.iloc[0]); ax.scatter([wv], [1], marker="x", s=22, color=INK, linewidths=0.9, zorder=3)
ax.annotate("white group (control capped,\nexcluded from the summary)", xy=(wv, 1), xytext=(-0.036, 0.55), ha="center", va="center", fontsize=6.5, color="#52514e", arrowprops=dict(arrowstyle="-", lw=0.4, color=GREY))
keep(1, "A", "white group (excluded)", wv, source="v5")
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks([2, 1, 0]); ax.set_yticklabels([b[0] for b in blocks]); ax.set_ylim(-0.7, 2.5); ax.set_xlim(-0.048, 0.014)
ax.set_xlabel("Change in the group's held-out AUROC, group withdrawal minus matched random removal (negative values are losses)"); ax.set_title("A", loc="left", fontweight="bold")
# B
ax2 = fig.add_subplot(gs[1, 0])
mf = [("MIMIC-IV,\nretrained", iv(MF13, "MIMIC communities | missed events per 1,000 at matched flag rate | base minus random"), iv(MF13, "MIMIC communities | missed events per 1,000 at matched flag rate | base minus original"), "v13b"),
      ("eICU,\ncentralised", iv(L35, "deaths unflagged per 1,000 | retrained without the consortium minus random removal"), iv(L35, "deaths unflagged per 1,000 | retrained without the consortium minus pooled model"), "v35 landmark"),
      ("eICU,\nfederation", iv(L35, "deaths unflagged per 1,000 | federation without the consortium minus federation after random removal"), iv(L35, "deaths unflagged per 1,000 | federation without the consortium minus full federation"), "v35 landmark")]
for i, (lab, (d, lo, hi), (d2, lo2, hi2), src) in enumerate(mf):
    yv = len(mf) - 1 - i
    ax2.plot([lo2, hi2], [yv + 0.18, yv + 0.18], color=LIGHT, lw=1.4, solid_capstyle="round"); ax2.scatter([d2], [yv + 0.18], marker="s", s=16, color=GREY, zorder=3, edgecolors="white", linewidths=0.6, label="Minus pooled model or full federation" if i == 0 else None)
    ax2.plot([lo, hi], [yv - 0.18, yv - 0.18], color=BLUE, lw=1.4, solid_capstyle="round"); ax2.scatter([d], [yv - 0.18], marker="o", s=18, color=BLUE, zorder=3, edgecolors="white", linewidths=0.6, label="Minus matched random removal" if i == 0 else None)
    keep(1, "B", lab.replace("\n", " ") + ", minus random", d, lo, hi, src); keep(1, "B", lab.replace("\n", " ") + ", minus pooled", d2, lo2, hi2, src)
ax2.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax2.set_yticks(range(len(mf))[::-1]); ax2.set_yticklabels([m[0] for m in mf]); ax2.set_ylim(-0.6, len(mf) - 0.4)
ax2.set_xlabel("Events unflagged per 1,000\nat matched alert counts"); ax2.set_title("B", loc="left", fontweight="bold", pad=24)
ax2.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1.0), handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
# C
ax3 = fig.add_subplot(gs[1, 1]); cs = PC.sort_values("penalty_landmark"); yy = np.arange(len(cs)); lab = [cons_label(f, g) for f, g in zip(cs.family, cs.group)]
for j, (c, f) in enumerate(zip(cs.penalty_landmark, cs.fed_penalty_landmark)): ax3.plot([c, f], [j, j], color=LIGHT, lw=1.2, zorder=1)
ax3.scatter(cs.penalty_landmark, yy, s=16, color=BLUE, label="Centralised", zorder=2, edgecolors="white", linewidths=0.5)
ax3.scatter(cs.fed_penalty_landmark, yy, s=16, color=AQUA, marker="^", label="Federation", zorder=2, edgecolors="white", linewidths=0.5)
for l, c, f in zip(lab, cs.penalty_landmark, cs.fed_penalty_landmark): keep(1, "C", l + " centralised", c, source="v35"); keep(1, "C", l + " federated", f, source="v35")
ax3.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax3.set_yticks(yy); ax3.set_yticklabels(lab); ax3.set_xlabel("Change in AUROC, consortium\nwithdrawal minus random removal")
ax3.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=1, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2); ax3.set_title("C", loc="left", fontweight="bold", pad=24)
# D (and Figure S1 with the Note 2 ratio)
def iso_panel(ax, xcol, xlabel, panel, fig_id):
    D = ISOD.copy(); m = ~D.control_capped.astype(bool); x, y = D[xcol].values, D.penalty.values
    sty = {"clustered": (BLUE, "o", "Clustered"), "census or era": (ORANGE, "s", "Census and era"), "consortium": (AQUA, "^", "eICU consortia")}
    for fg, (c, mk, lb) in sty.items():
        k = (D.family_group == fg).values & m.values; ax.scatter(x[k], y[k], s=12, color=c, marker=mk, alpha=0.75, linewidths=0, label=lb)
    ax.scatter(x[~m.values], y[~m.values], marker="x", s=20, color=INK, linewidths=0.8, label="White (not fitted)")
    b1, b0 = np.polyfit(x[m.values], y[m.values], 1); r = np.random.RandomState(0); xs = np.linspace(x.min(), x.max(), 50); bs = []
    idx = np.where(m.values)[0]
    for _ in range(5000):
        j = r.choice(idx, len(idx), True); bb = np.polyfit(x[j], y[j], 1); bs.append(bb[0] * xs + bb[1])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); ax.fill_between(xs, lo, hi, color=LIGHT, alpha=0.6, lw=0); ax.plot(xs, b1 * xs + b0, color=INK, lw=0.9)
    ax.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_xlabel(xlabel); ax.set_ylabel("Withdrawal minus random removal, AUROC")
    ax.text(0.04, 0.04, f"Slope {fmt(b1)} per unit\n(48 groups)", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color="#52514e")
    for g, xv, yv, f in zip(D.group, x, y, D.family_group): keep(fig_id, panel, f"{f} {g}", yv, source=f"v34; {xcol} {xv:.3f}")
    keep(fig_id, panel, "OLS slope, 48 groups", b1, source="np.polyfit; interval band 5,000 draws over groups")
    return ax
ax4 = fig.add_subplot(gs[1, 2]); iso_panel(ax4, "ratio_equal", "Equal-pool isolation ratio", "D", 1)
ax4.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.08, 1.0), ncol=2, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2, columnspacing=0.8, fontsize=6.5); ax4.set_title("D", loc="left", fontweight="bold", pad=24)
fig.savefig(f"{OUT}/Figure1_v9.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/Figure1_v9.png", dpi=300, bbox_inches="tight"); plt.close(fig)
fig = plt.figure(figsize=(90 * MM, 80 * MM)); ax = fig.add_subplot(111); iso_panel(ax, "ratio_v5", "Isolation ratio (Supplementary Appendix, Note 2)", "S1", "S1")
ax.legend(frameon=False, loc="upper right", fontsize=6.5, handletextpad=0.2, labelspacing=0.2)
fig.savefig(f"{OUT}/FigureS1_v9.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/FigureS1_v9.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ================================================================ Figure 2
I19 = json.load(open(f"{RES}/v19_protocol/intervals.json")); I24 = json.load(open(f"{RES}/v24_symmetric_mimic/intervals.json"))["all"]["results"]
S24 = pd.read_csv(f"{RES}/v24_symmetric_mimic/summary.csv").pivot_table(index="group", columns="arm", values="auroc")
S19 = pd.read_csv(f"{RES}/v19_protocol/summary.csv"); S19 = S19[S19.group.str.startswith("community_trainfit")].pivot_table(index="group", columns="arm", values="auroc")
S16 = pd.read_csv(f"{RES}/v16_strict/summary.csv"); S16 = S16[S16.group.str.startswith("community_trainfit")].pivot_table(index="group", columns="arm", values="auroc")
ctrl = json.load(open(f"{ROOT}/nejmai_build/figure2_control_boot.json"))
groups = list(S19.index); assert len(groups) == 25
ISO = ["community_trainfit/" + c for c in json.load(open(f"{RES}/v14_commonbase_1/sets_B.json"))["most_isolated"]]
chosen = {}
for g in groups:
    ch = [json.load(open(f"{RES}/v19_protocol/runs/{g.replace('/', '__')}__s{s}.json"))["selection"]["chosen"] for s in range(3)]
    top = sorted(set(ch), key=lambda k: (-ch.count(k), k)); chosen[g] = top[0] if ch.count(top[0]) >= 2 else "mixed"
fig = plt.figure(figsize=(180 * MM, 140 * MM)); gs = GridSpec(2, 2, figure=fig, hspace=0.75, wspace=0.45, width_ratios=[1.3, 1], height_ratios=[1, 1.25])
ax = fig.add_subplot(gs[0, 0]); r = np.random.RandomState(4)
arms = [("Base alone (state 3)", S19["strict_base"] - S19["pooled"], iv(I19["all"]["results"], "strict base minus pooled"), GREY, "o", "v19"),
        ("Base + control component", S16["strict_control"] - S16["pooled"], (ctrl["d"], ctrl["ci"][0], ctrl["ci"][1]), ORANGE, "s", "v16 two-stage bootstrap"),
        ("Base + chosen component (state 2)", S19["chosen"] - S19["pooled"], iv(I19["all"]["results"], "chosen component minus pooled"), BLUE, "o", "v19"),
        ("Adapted pooled comparator", S24["pooled_cf_chosen"] - S24["pooled"], iv(I24, "pooled with chosen cross-fitted component minus pooled"), AQUA, "^", "v24")]
for i, (lab, vals, (d, lo, hi), c, mk, src) in enumerate(arms):
    yv = len(arms) - 1 - i; ax.scatter(vals.values, yv + r.uniform(-0.2, 0.2, len(vals)), s=7, color=c, alpha=0.45, linewidths=0, marker=mk)
    ax.plot([lo, hi], [yv - 0.36, yv - 0.36], color=INK, lw=1.1); ax.scatter([d], [yv - 0.36], s=13, color=INK, marker="D", zorder=3)
    keep(2, "A", lab + " mean", d, lo, hi, src); [keep(2, "A", f"{lab} point {g}", v) for g, v in vals.items()]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks(range(len(arms))[::-1]); ax.set_yticklabels([a[0] for a in arms])
ax.set_xlabel("Change in AUROC relative to the pooled model"); ax.set_title("A  MIMIC-IV, 25 clustered groups", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[0, 1])
cb = iv(L35, "AUROC | control component minus strict base"); bp = iv(L35, "AUROC | strict base minus pooled model (centralised)")
arms_e = [("Base alone (state 3)", PL.auroc_base - PL.auroc_pooled, bp, GREY, "o"),
          ("Base + control component", PL.auroc_control - PL.auroc_pooled, (bp[0] + cb[0], np.nan, np.nan), ORANGE, "s"),
          ("Base + chosen component (state 2)", PL.auroc_confined - PL.auroc_pooled, iv(L35, "AUROC | confined (strict) minus pooled model"), BLUE, "o"),
          ("Adapted pooled comparator", PL.auroc_sym - PL.auroc_pooled, iv(L35, "AUROC | adapted pooled comparator minus pooled model"), AQUA, "^")]
for i, (lab, vals, (d, lo, hi), c, mk) in enumerate(arms_e):
    yv = len(arms_e) - 1 - i; ax.scatter(vals.values, yv + r.uniform(-0.18, 0.18, len(vals)), s=10, color=c, alpha=0.6, linewidths=0, marker=mk)
    if np.isfinite(lo): ax.plot([lo, hi], [yv - 0.33, yv - 0.33], color=INK, lw=1.1)
    ax.scatter([d], [yv - 0.33], s=13, color=INK, marker="D", zorder=3); keep(2, "B", lab + " mean", d, lo, hi, "v35 landmark"); [keep(2, "B", f"{lab} point {g}", v) for g, v in vals.items()]
ax.axvline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_yticks(range(len(arms_e))[::-1]); ax.set_yticklabels([]); ax.set_xticks([-0.02, -0.01, 0, 0.005])
ax.set_xlabel("Change in AUROC relative to the pooled model"); ax.set_title("B  eICU centralised, 10 consortia, in the unit at 24 h", loc="left", fontweight="bold")
# C: schematic, enlarged
ax = fig.add_subplot(gs[1, 0]); pc = ax.get_position(); ax.set_position([0.02, pc.y0, pc.x1 + 0.02, pc.height]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.set_title("C  The three permission states", loc="left", fontweight="bold")
def box(x, y, w, h, text, fc="white", ec=INK, ls="-", fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)
ax.text(24, 100, "Shared model", ha="center", va="top", fontsize=7, fontweight="bold"); ax.text(76, 100, "Group's component", ha="center", va="top", fontsize=7, fontweight="bold")
rowsC = [(64, "State 1, pooled", "Trained on the pool\nand the group's data", "#e3edf9", BLUE, "None", "white", GREY, (0, (2, 2))),
         (34, "State 2, confined", "Base trained on the\npool only (provenance\nrecord, checksum)", "#f1f1ef", GREY, "Fitted on the group's data,\ntype chosen by validation,\napplied to its patients only", "#e3edf9", BLUE, "-"),
         (4, "State 3, withdrawn", "Base unchanged\n(same checksum)", "#f1f1ef", GREY, "Deleted", "white", GREY, (0, (2, 2)))]
for y, lab, t1, f1, e1, t2, f2, e2, ls2 in rowsC:
    ax.text(2, y + 22, lab, ha="left", va="bottom", fontsize=7, fontweight="bold")
    box(2, y, 44, 21, t1, fc=f1, ec=e1); box(54, y, 44, 21, t2, fc=f2, ec=e2, ls=ls2)
ax.annotate("", xy=(76, 25.5), xytext=(76, 33.5), arrowprops=dict(arrowstyle="->", lw=0.9, color=INK)); ax.text(78, 29.5, "withdraw: delete", ha="left", va="center", fontsize=6.5)
ax.annotate("", xy=(50, 14), xytext=(50, 74), arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
ax.text(48.3, 44, "withdraw: retrain", ha="center", va="center", fontsize=6.5, rotation=90)
# D: confined minus adapted comparator against the equal-pool ratio
ax = fig.add_subplot(gs[1, 1]); dd = S24["confined_chosen"] - S24["pooled_cf_chosen"]
eq = ISOD[ISOD.family_group == "clustered"].set_index("group").ratio_equal; xi = eq.reindex([g.split("/")[1] for g in dd.index]).values
MK = {"residual": ("o", "Residual component"), "head": ("^", "Representation head"), "finetuned": ("D", "Fine-tuned copy"), "mixed": ("X", "No type in 2 of 3 seeds")}
for kind, (mk, lab) in MK.items():
    m = np.array([chosen[g] == kind for g in dd.index]); iso_m = np.array([g in ISO for g in dd.index])
    ax.scatter(xi[m & ~iso_m], dd.values[m & ~iso_m], s=16, facecolors="none", edgecolors=BLUE, linewidths=0.8, marker=mk, label=lab)
    ax.scatter(xi[m & iso_m], dd.values[m & iso_m], s=18, color=BLUE, marker=mk)
ax.scatter([], [], s=18, color=BLUE, marker="s", label="Filled: five most isolated")
for g, x, v in zip(dd.index, xi, dd.values): keep(2, "D", g, v, source=f"v24 summary; equal-pool ratio {x:.3f}; chosen {chosen[g]}")
ax.axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2))); ax.set_xlabel("Equal-pool isolation ratio")
ax.set_ylabel("AUROC, confined system minus\nadapted pooled comparator"); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, handletextpad=0.3, labelspacing=0.2, columnspacing=0.8, borderaxespad=0.1, fontsize=6.5)
ax.set_title("D  Confined minus adapted comparator", loc="left", fontweight="bold", pad=34)
fig.savefig(f"{OUT}/Figure2_v9.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/Figure2_v9.png", dpi=300, bbox_inches="tight"); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{OUT}/figure_data_v9.csv", index=False); print("written", OUT, len(rows), "values")
