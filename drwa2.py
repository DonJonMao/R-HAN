import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# Data
# =========================

data = [
    # Published MATH cost references
    ["LLM-Debate", "MATH", "Published MATH refs", 3.28, 10.46, 13.73, 5.77, 48.54],
    ["DyLAN",      "MATH", "Published MATH refs", 6.08,  3.30,  9.39, 3.94, 48.63],
    ["MacNet",     "MATH", "Published MATH refs", 7.52,  2.04,  9.57, 4.02, 45.18],
    ["GPTSwarm",   "MATH", "Published MATH refs", 3.11,  0.79,  3.89, 1.64, 47.88],
    ["AFlow",      "MATH", "Published MATH refs", 2.51,  2.15,  4.66, 1.96, 51.28],
    ["MaAS",       "MATH", "Published MATH refs", 1.31,  0.85,  2.16, 0.91, 51.82],

    # Published MMLU token references
    ["MASround=T",   "MMLU", "Published MMLU refs", 2.20, 0.66, 2.86, 1.20, 84.31],
    ["AgentPrune",   "MMLU", "Published MMLU refs", 2.00, 0.67, 2.67, 1.12, 83.66],
    ["AgentDropout", "MMLU", "Published MMLU refs", 1.30, 0.46, 1.76, 0.74, 84.97],

    # R-HAN difficulty scaling
    ["R-HAN", "MATH",      "R-HAN", 1.46, 0.98, 2.44, 1.03, 78.29],
    ["R-HAN", "MMLU",      "R-HAN", 1.38, 0.94, 2.32, 0.97, 88.37],
    ["R-HAN", "MATH-L5",   "R-HAN", 1.76, 1.22, 2.98, 1.25, 68.21],
    ["R-HAN", "MMLU-Pro",  "R-HAN", 1.96, 1.37, 3.33, 1.40, 79.27],
]

df = pd.DataFrame(
    data,
    columns=[
        "Method", "Benchmark", "Group",
        "PromptTok", "CompletionTok", "TotalTok", "RelCost", "Acc"
    ]
)

# =========================
# Style
# =========================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 19,
    "axes.titlesize": 20,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

colors = {
    "Published MATH refs": "#f4a261",
    "Published MMLU refs": "#90be6d",
    "R-HAN": "#4ea8de",
}


# =========================
# Bubble-size scaling
# =========================
# Stronger size separation than before.
# NOTE: current bubble size still reflects TotalTok,
# because the table does not provide wall-clock time.
# If you later provide time, replace TotalTok here with time.
# =========================

tok_min = df["TotalTok"].min()
tok_max = df["TotalTok"].max()

def bubble_size(v):
    # Normalize to [0, 1]
    x = (np.array(v) - tok_min) / (tok_max - tok_min + 1e-9)
    # Stronger nonlinear scaling for clearer separation
    return 260 + 5200 * (x ** 1.25)


# =========================
# Label helpers
# =========================

def pretty_name(row):
    if row["Method"] == "R-HAN":
        return f"R-HAN\n({row['Benchmark']})"
    return row["Method"]

def auto_offset(row, xlim, ylim):
    """
    Put labels close to bubbles and keep them inside axes.
    Returns (dx, dy, ha, va) in 'offset points'.
    """
    x = row["Acc"]
    y = row["TotalTok"]

    x_mid = (xlim[0] + xlim[1]) / 2
    y_mid = (ylim[0] + ylim[1]) / 2

    # horizontal placement
    if x > x_mid + 0.2 * (xlim[1] - xlim[0]):
        dx = -8
        ha = "right"
    elif x < x_mid - 0.2 * (xlim[1] - xlim[0]):
        dx = 8
        ha = "left"
    else:
        dx = 0
        ha = "center"

    # vertical placement
    if y > y_mid + 0.2 * (ylim[1] - ylim[0]):
        dy = -8
        va = "top"
    elif y < y_mid - 0.2 * (ylim[1] - ylim[0]):
        dy = 8
        va = "bottom"
    else:
        dy = 8
        va = "bottom"

    return dx, dy, ha, va


def draw_panel(ax, subdf, title, xlim, ylim):
    # Draw bubbles grouped by color
    for group, sub in subdf.groupby("Group"):
        ax.scatter(
            sub["Acc"],
            sub["TotalTok"],
            s=bubble_size(sub["TotalTok"]),
            c=colors[group],
            alpha=0.62,
            edgecolors="gray",
            linewidths=0.9,
            zorder=2,
        )

    # Center plus marks
    ax.scatter(
        subdf["Acc"],
        subdf["TotalTok"],
        marker="+",
        color="black",
        s=80,
        linewidths=1.2,
        zorder=3,
    )

    # Labels: close to bubbles, boxed, clipped to axes
    for _, row in subdf.iterrows():
        dx, dy, ha, va = auto_offset(row, xlim, ylim)
        ax.annotate(
            pretty_name(row),
            xy=(row["Acc"], row["TotalTok"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=10,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("#Tokens (M)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.28)


# =========================
# Split panels
# =========================

math_panel = pd.concat([
    df[df["Group"] == "Published MATH refs"],
    df[(df["Group"] == "R-HAN") & (df["Benchmark"] == "MATH")]
])

mmlu_panel = pd.concat([
    df[df["Group"] == "Published MMLU refs"],
    df[(df["Group"] == "R-HAN") & (df["Benchmark"] == "MMLU")]
])

rhan_panel = df[df["Group"] == "R-HAN"]


# =========================
# Plot
# =========================

fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.6))

draw_panel(
    axes[0],
    math_panel,
    "MATH Cost References",
    xlim=(44, 80.5),
    ylim=(1.5, 14.8),
)

draw_panel(
    axes[1],
    mmlu_panel,
    "MMLU Token References",
    xlim=(82.8, 89.2),
    ylim=(1.4, 3.1),
)

draw_panel(
    axes[2],
    rhan_panel,
    "R-HAN Difficulty Scaling",
    xlim=(66, 90),
    ylim=(2.1, 3.55),
)

# =========================
# Custom legend
# =========================

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=11,
        markerfacecolor=colors["Published MATH refs"],
        markeredgecolor="gray",
        alpha=0.70,
        label="Published MATH refs",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=11,
        markerfacecolor=colors["Published MMLU refs"],
        markeredgecolor="gray",
        alpha=0.70,
        label="Published MMLU refs",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markersize=11,
        markerfacecolor=colors["R-HAN"],
        markeredgecolor="gray",
        alpha=0.70,
        label="R-HAN",
    ),
    Line2D(
        [0], [0],
        marker="+",
        linestyle="",
        markersize=12,
        color="black",
        label="Method point",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=True,
    columnspacing=1.6,
    handletextpad=0.5,
)

plt.tight_layout(rect=[0, 0, 1, 0.90], w_pad=2.2)

plt.savefig("rhan_cost_analysis_split_bubble_v2.png", bbox_inches="tight")
plt.savefig("rhan_cost_analysis_split_bubble_v2.pdf", bbox_inches="tight")
plt.savefig("rhan_cost_analysis_split_bubble_v2.svg", bbox_inches="tight")

plt.show()