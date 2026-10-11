"""Figure 1 for the manuscript: collective removal versus matched random removal, MIMIC-IV and eICU."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

U = "/mnt/user-data/uploads/mimic/results"
SH = f"{U}/sameheldout_20260926_100534"; FU = f"{U}/fu_20260926_111426"; EI = f"{U}/eicu_20260926_194832"
S = pd.read_csv(f"{SH}/summary.csv"); PJ = json.load(open(f"{SH}/paired.json"))
F = pd.read_csv(f"{FU}/summary.csv"); FP = json.load(open(f"{FU}/paired.json"))
E = pd.read_csv(f"{EI}/summary.csv"); EP = json.load(open(f"{EI}/paired.json"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 8, "axes.titleweight": "bold",
                     "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

def style(ax, zero=True):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    if zero: ax.axhline(0, color=INK2, linewidth=0.6)

W = S.pivot(index="community", columns="model", values="d_auroc"); WE = S.pivot(index="community", columns="model", values="d_ece")
sz = S[S.model == "original"].set_index("community").forget_rows_community; order = sz.sort_values().index; x = np.arange(len(order))
EW = E.pivot(index="hospital", columns="model", values="d_auroc"); esz = E[E.model == "original"].set_index("hospital").forget_stays
eorder = esz.sort_values().index; ex = np.arange(len(eorder))

fig, axs = plt.subplots(2, 2, figsize=(164 / 25.4, 124 / 25.4), dpi=300)
ms = 3.2; lw = 0

# a: MIMIC ΔAUROC per community
ax = axs[0, 0]; style(ax)
ax.plot(x, W.loc[order, "exact_random"], "s", ms=ms, color=ORANGE, mec="white", mew=0.4, label="remove matched random patients")
ax.plot(x, W.loc[order, "exact_random_admatched"], "^", ms=ms, color=AQUA, mec="white", mew=0.4, label="remove matched random admissions")
ax.plot(x, W.loc[order, "exact_community"], "o", ms=ms + 0.4, color=BLUE, mec="white", mew=0.4, label="remove the community")
ax.set_title("a  MIMIC-IV, discrimination", loc="left")
ax.set_ylabel("ΔAUROC, held-out community patients"); ax.set_xticks([]); ax.set_xlabel("25 pseudo-communities, smallest to largest")
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], color="#e34948", linewidth=0.8, linestyle=(0, (3, 2)), label="ε = 0.01 tolerance (b)")); l.append("ε = 0.01 tolerance (b)")
ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.3, columnspacing=1.0)

# b: MIMIC ΔECE per community
ax = axs[0, 1]; style(ax)
ax.axhline(0.01, color="#e34948", linewidth=0.6, linestyle=(0, (3, 2)))
ax.plot(x, WE.loc[order, "exact_random"], "s", ms=ms, color=ORANGE, mec="white", mew=0.4)
ax.plot(x, WE.loc[order, "exact_random_admatched"], "^", ms=ms, color=AQUA, mec="white", mew=0.4)
ax.plot(x, WE.loc[order, "exact_community"], "o", ms=ms + 0.4, color=BLUE, mec="white", mew=0.4)
ax.set_title("b  MIMIC-IV, calibration", loc="left")
ax.set_ylabel("ΔECE, held-out community patients"); ax.set_xticks([]); ax.set_xlabel("25 pseudo-communities, smallest to largest")

# c: eICU ΔAUROC per hospital
ax = axs[1, 0]; style(ax)
ax.plot(ex, EW.loc[eorder, "exact_random"], "s", ms=ms, color=ORANGE, mec="white", mew=0.4, label="remove matched random stays")
ax.plot(ex, EW.loc[eorder, "exact_hospital"], "o", ms=ms + 0.4, color=BLUE, mec="white", mew=0.4, label="remove the hospital")
ax.set_title("c  eICU, discrimination", loc="left")
ax.set_ylabel("ΔAUROC, held-out hospital patients"); ax.set_xticks([]); ax.set_xlabel("64 hospitals, smallest to largest")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, handletextpad=0.3, columnspacing=1.0)

# d: paired contrasts by remedy
ax = axs[1, 1]; style(ax, zero=False); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.5)
rows = [("Exact retrain", FP["exact"], EP["exact_hospital_vs_exact_random"]),
        ("Fine-tune", FP["finetune"], EP["finetune_vs_exact_random"]),
        ("NegGrad+", FP["neggrad"], EP["neggrad_vs_exact_random"]),
        ("Fair Unlearning (race)", FP["fu_race"], None),
        ("Fair Unlearning (community)", FP["fu_community"], None)]
yy = np.arange(len(rows))[::-1]
for y0, (lab, m, e) in zip(yy, rows):
    ax.plot([m["ci"][0], m["ci"][1]], [y0 + 0.12, y0 + 0.12], color=BLUE, linewidth=1.2, solid_capstyle="round")
    ax.plot(m["d_auroc"], y0 + 0.12, "o", ms=ms + 0.6, color=BLUE, mec="white", mew=0.4)
    if e:
        ax.plot([e["ci"][0], e["ci"][1]], [y0 - 0.12, y0 - 0.12], color=ORANGE, linewidth=1.2, solid_capstyle="round")
        ax.plot(e["d_auroc"], y0 - 0.12, "s", ms=ms + 0.2, color=ORANGE, mec="white", mew=0.4)
ax.axvline(0, color=INK2, linewidth=0.6)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
ax.set_xlabel("paired ΔAUROC, group minus matched random (95% CI)")
ax.set_title("d  Cost to the withdrawing group, by remedy", loc="left")
ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="none", ms=ms, label="MIMIC-IV communities"),
                   Line2D([], [], marker="s", color=ORANGE, linestyle="none", ms=ms, label="eICU hospitals")],
          loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.3, columnspacing=1.0)
ax.set_xlim(min(-0.1, ax.get_xlim()[0]), 0.02); ax.set_ylim(-0.6, len(rows) - 0.4)

for ax in axs.flat:
    ax.tick_params(labelsize=6.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.13, wspace=0.55, hspace=0.6)
fig.savefig("/mnt/user-data/outputs/Figure1.png", dpi=300, bbox_inches="tight", pad_inches=0.04); fig.savefig("/mnt/user-data/outputs/Figure1.pdf", bbox_inches="tight", pad_inches=0.04)
print("saved")
