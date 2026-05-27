import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects


# =========================
# 1. Output directory
# =========================

os.makedirs("figures", exist_ok=True)


# =========================
# 2. Estimated plotting data
# =========================
# Columns:
# Category, Method, Dataset, TokenCost_M, Accuracy, TimeCost_rel
#
# TokenCost_M: estimated total token cost in million tokens.
# TimeCost_rel: estimated relative time cost. It is proportional to token cost.

rows = [
    # Single-agent
    ("Single", "Self-Refine", "MMLU", 1.25, 75.44, 0.51),
    ("Single", "Self-Refine", "MMLU-Pro", 1.80, 57.97, 0.74),
    ("Single", "Self-Refine", "MATH", 1.55, 46.10, 0.64),
    ("Single", "Self-Refine", "MATH-Lv5", 1.94, 32.85, 0.80),

    # Pre-defined MAS
    ("Pre-defined", "LLM-Debate", "MMLU", 3.05, 81.04, 2.20),
    ("Pre-defined", "LLM-Debate", "MMLU-Pro", 4.20, 63.28, 3.03),
    ("Pre-defined", "LLM-Debate", "MATH", 13.73, 52.96, 9.90),
    ("Pre-defined", "LLM-Debate", "MATH-Lv5", 16.40, 38.55, 11.83),

    ("Pre-defined", "AgentVerse", "MMLU", 2.55, 78.36, 1.46),
    ("Pre-defined", "AgentVerse", "MMLU-Pro", 3.50, 60.83, 2.01),
    ("Pre-defined", "AgentVerse", "MATH", 5.80, 50.85, 3.33),
    ("Pre-defined", "AgentVerse", "MATH-Lv5", 7.08, 36.85, 4.06),

    ("Pre-defined", "LLM-Blender", "MMLU", 2.20, 81.22, 0.97),
    ("Pre-defined", "LLM-Blender", "MMLU-Pro", 3.05, 63.47, 1.35),
    ("Pre-defined", "LLM-Blender", "MATH", 4.40, 50.34, 1.95),
    ("Pre-defined", "LLM-Blender", "MATH-Lv5", 5.19, 36.16, 2.30),

    ("Pre-defined", "DyLAN", "MMLU", 2.95, 79.96, 1.79),
    ("Pre-defined", "DyLAN", "MMLU-Pro", 4.05, 62.14, 2.46),
    ("Pre-defined", "DyLAN", "MATH", 9.39, 51.12, 5.70),
    ("Pre-defined", "DyLAN", "MATH-Lv5", 11.27, 37.38, 6.84),

    # Automated Agentic Systems
    ("Auto", "H-Swarms", "MMLU", 2.85, 83.68, 1.59),
    ("Auto", "H-Swarms", "MMLU-Pro", 3.95, 69.00, 2.20),
    ("Auto", "H-Swarms", "MATH", 6.20, 65.89, 3.46),
    ("Auto", "H-Swarms", "MATH-Lv5", 7.56, 50.27, 4.21),

    ("Auto", "AFlow", "MMLU", 2.35, 83.10, 1.04),
    ("Auto", "AFlow", "MMLU-Pro", 3.20, 64.35, 1.42),
    ("Auto", "AFlow", "MATH", 4.66, 73.35, 2.06),
    ("Auto", "AFlow", "MATH-Lv5", 5.78, 58.82, 2.56),

    ("Auto", "RouterDC", "MMLU", 1.95, 82.01, 0.70),
    ("Auto", "RouterDC", "MMLU-Pro", 2.65, 63.27, 0.96),
    ("Auto", "RouterDC", "MATH", 2.85, 73.46, 1.03),
    ("Auto", "RouterDC", "MATH-Lv5", 3.55, 58.93, 1.28),

    ("Auto", "A2Flow", "MMLU", 2.30, 83.29, 1.02),
    ("Auto", "A2Flow", "MMLU-Pro", 3.10, 63.42, 1.37),
    ("Auto", "A2Flow", "MATH", 4.20, 58.50, 1.86),
    ("Auto", "A2Flow", "MATH-Lv5", 5.12, 44.76, 2.27),

    ("Auto", "DAAO", "MMLU", 2.05, 84.90, 0.81),
    ("Auto", "DAAO", "MMLU-Pro", 3.00, 65.28, 1.18),
    ("Auto", "DAAO", "MATH", 3.10, 55.37, 1.22),
    ("Auto", "DAAO", "MATH-Lv5", 4.10, 42.57, 1.61),

    ("Auto", "MasRouter", "MMLU", 2.05, 84.25, 0.74),
    ("Auto", "MasRouter", "MMLU-Pro", 2.80, 64.85, 1.01),
    ("Auto", "MasRouter", "MATH", 2.75, 75.42, 0.99),
    ("Auto", "MasRouter", "MATH-Lv5", 3.44, 61.36, 1.24),

    ("Auto", "MaAS", "MMLU", 2.10, 83.01, 0.86),
    ("Auto", "MaAS", "MMLU-Pro", 2.90, 63.68, 1.19),
    ("Auto", "MaAS", "MATH", 2.16, 74.45, 0.89),
    ("Auto", "MaAS", "MATH-Lv5", 2.81, 60.23, 1.15),

    ("Auto", "BiRouter", "MMLU", 2.15, 86.80, 0.81),
    ("Auto", "BiRouter", "MMLU-Pro", 3.00, 66.53, 1.13),
    ("Auto", "BiRouter", "MATH", 2.60, 74.92, 0.98),
    ("Auto", "BiRouter", "MATH-Lv5", 3.25, 60.87, 1.23),

    # Graph-based MAS
    ("Graph", "GPTSwarm", "MMLU", 2.30, 82.80, 1.09),
    ("Graph", "GPTSwarm", "MMLU-Pro", 3.05, 64.19, 1.45),
    ("Graph", "GPTSwarm", "MATH", 3.89, 68.85, 1.85),
    ("Graph", "GPTSwarm", "MATH-Lv5", 4.78, 54.64, 2.27),

    ("Graph", "G-Designer", "MMLU", 2.60, 87.20, 1.28),
    ("Graph", "G-Designer", "MMLU-Pro", 3.55, 66.94, 1.75),
    ("Graph", "G-Designer", "MATH", 3.60, 70.46, 1.77),
    ("Graph", "G-Designer", "MATH-Lv5", 4.50, 55.39, 2.21),

    # Full R-HAN
    ("Graph", "R-HAN", "MMLU", 2.32, 88.37, 0.95),
    ("Graph", "R-HAN", "MMLU-Pro", 3.33, 79.27, 1.36),
    ("Graph", "R-HAN", "MATH", 2.44, 78.29, 1.00),
    ("Graph", "R-HAN", "MATH-Lv5", 2.98, 68.21, 1.22),
]

df = pd.DataFrame(
    rows,
    columns=["Category", "Method", "Dataset", "TokenCost_M", "Accuracy", "TimeCost_rel"],
)


# =========================
# 3. R-HAN ablation point
# =========================
# We only keep one ablation star:
# "w/o Hierarchical Sparse Coordination"
#
# Accuracy:
# Combine the drops from w/o latent memory and w/o global controller:
#   acc_w/o_HSC = full - [(full - w/o_latent_memory) + (full - w/o_global_controller)]
#
# Cost:
# User-specified assumption:
#   w/o self-refine cost decreases.
#   w/o union graph cost decreases.
#   w/o HSC cost increase equals the sum of these two cost-change magnitudes.

datasets = ["MATH", "MATH-Lv5", "MMLU", "MMLU-Pro"]

full_acc = {
    "MMLU": 88.37,
    "MMLU-Pro": 79.27,
    "MATH": 78.29,
    "MATH-Lv5": 68.21,
}

full_cost = {
    "MMLU": 2.32,
    "MMLU-Pro": 3.33,
    "MATH": 2.44,
    "MATH-Lv5": 2.98,
}

full_time = {
    "MMLU": 0.95,
    "MMLU-Pro": 1.36,
    "MATH": 1.00,
    "MATH-Lv5": 1.22,
}

wo_latent_memory_acc = {
    "MMLU": 87.24,
    "MMLU-Pro": 75.42,
    "MATH": 76.15,
    "MATH-Lv5": 63.49,
}

wo_global_controller_acc = {
    "MMLU": 87.85,
    "MMLU-Pro": 76.80,
    "MATH": 77.24,
    "MATH-Lv5": 64.11,
}

wo_hsc_acc = {
    ds: round(
        full_acc[ds]
        - ((full_acc[ds] - wo_latent_memory_acc[ds])
           + (full_acc[ds] - wo_global_controller_acc[ds])),
        2,
    )
    for ds in datasets
}

# Auxiliary cost assumptions used only to estimate the single w/o HSC point.
# These two variants are NOT plotted.
wo_self_refine_cost = {
    "MMLU": 2.02,
    "MMLU-Pro": 2.78,
    "MATH": 1.98,
    "MATH-Lv5": 2.30,
}

wo_union_graph_cost = {
    "MMLU": 2.12,
    "MMLU-Pro": 3.00,
    "MATH": 2.13,
    "MATH-Lv5": 2.55,
}

wo_hsc_cost = {
    ds: round(
        full_cost[ds]
        + (full_cost[ds] - wo_self_refine_cost[ds])
        + (full_cost[ds] - wo_union_graph_cost[ds]),
        2,
    )
    for ds in datasets
}

wo_hsc_time = {
    ds: round(full_time[ds] * (wo_hsc_cost[ds] / full_cost[ds]), 2)
    for ds in datasets
}


# =========================
# 4. Visual style
# =========================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

category_colors = {
    "Single": "#8a8a8a",
    "Pre-defined": "#f0b323",
    "Auto": "#67b7e1",
    "Graph": "#17b978",
}

category_order = ["Single", "Pre-defined", "Auto", "Graph"]

rhan_style = {
    "edgecolor": "#c82333",
    "facecolor": "#e53935",
}

wo_hsc_style = {
    "edgecolor": "#8e44ad",   # purple, clearly separated from other category colors
    "facecolor": "white",
}


def point_area(time_cost: float) -> float:
    """Map relative time cost to marker area."""
    return 220 * (time_cost ** 1.18)


# Manual label offsets to reduce overlap.
# Format: dx, dy, horizontal_alignment
label_offsets = {
    "MATH": {
        "Self-Refine": (6, 4, "left"),
        "LLM-Debate": (8, 4, "left"),
        "DyLAN": (6, 0, "left"),
        "AgentVerse": (-6, 4, "right"),
        "LLM-Blender": (6, -6, "left"),
        "DAAO": (-6, 0, "right"),
        "A2Flow": (6, 4, "left"),
        "H-Swarms": (6, 0, "left"),
        "AFlow": (6, 0, "left"),
        "GPTSwarm": (6, 0, "left"),
        "G-Designer": (6, -6, "left"),
        "RouterDC": (-6, -8, "right"),
        "MasRouter": (6, 0, "left"),
        "MaAS": (6, -8, "left"),
        "BiRouter": (-6, -10, "right"),
    },
    "MATH-Lv5": {
        "Self-Refine": (6, 4, "left"),
        "LLM-Debate": (8, 4, "left"),
        "DyLAN": (-6, -2, "right"),
        "AgentVerse": (-6, 4, "right"),
        "LLM-Blender": (6, -6, "left"),
        "DAAO": (6, 4, "left"),
        "A2Flow": (6, 4, "left"),
        "H-Swarms": (6, 0, "left"),
        "AFlow": (6, 0, "left"),
        "GPTSwarm": (6, 0, "left"),
        "G-Designer": (6, -6, "left"),
        "RouterDC": (-6, -6, "right"),
        "MasRouter": (6, 4, "left"),
        "MaAS": (6, -6, "left"),
        "BiRouter": (-6, -10, "right"),
    },
    "MMLU": {
        "Self-Refine": (6, 4, "left"),
        "LLM-Debate": (-6, -8, "right"),
        "DyLAN": (-6, 4, "right"),
        "AgentVerse": (6, -8, "left"),
        "LLM-Blender": (6, -6, "left"),
        "DAAO": (6, -6, "left"),
        "A2Flow": (6, 4, "left"),
        "H-Swarms": (6, 4, "left"),
        "AFlow": (6, 4, "left"),
        "GPTSwarm": (6, 4, "left"),
        "G-Designer": (6, 4, "left"),
        "RouterDC": (-6, -6, "right"),
        "MasRouter": (-6, -8, "right"),
        "MaAS": (6, -8, "left"),
        "BiRouter": (6, 0, "left"),
    },
    "MMLU-Pro": {
        "Self-Refine": (6, 4, "left"),
        "LLM-Debate": (6, 4, "left"),
        "DyLAN": (-6, 4, "right"),
        "AgentVerse": (6, -4, "left"),
        "LLM-Blender": (-6, -10, "right"),
        "DAAO": (6, -6, "left"),
        "A2Flow": (-6, 4, "right"),
        "H-Swarms": (6, 4, "left"),
        "AFlow": (6, 0, "left"),
        "GPTSwarm": (6, 4, "left"),
        "G-Designer": (6, -10, "left"),
        "RouterDC": (-6, -6, "right"),
        "MasRouter": (-6, -10, "right"),
        "MaAS": (6, -6, "left"),
        "BiRouter": (6, 0, "left"),
    },
}

rhan_label_offsets = {
    "MATH": (8, -6, "left"),
    "MATH-Lv5": (8, -6, "left"),
    "MMLU": (8, -6, "left"),
    "MMLU-Pro": (8, -6, "left"),
}


# =========================
# 5. Plot function
# =========================

def annotate_with_stroke(ax, text, xy, xytext, ha, color="black", fontsize=8.2):
    ann = ax.annotate(
        text,
        xy,
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=8,
    )
    ann.set_path_effects([
        patheffects.withStroke(linewidth=2.4, foreground="white", alpha=0.96)
    ])
    return ann


def draw_pair(
    datasets_pair,
    xlims,
    xticks_list,
    output_prefix,
):
    pair_df = df[df["Dataset"].isin(datasets_pair)].copy()

    all_y = list(pair_df["TokenCost_M"].values)
    all_y += [wo_hsc_cost[ds] for ds in datasets_pair]
    ymax = max(max(all_y) * 1.08, 4.6)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 7.6), dpi=220)
    fig.patch.set_facecolor("white")

    for ax, ds, xlim, xticks in zip(axes, datasets_pair, xlims, xticks_list):
        d = df[df["Dataset"] == ds].copy()

        # Baseline methods, excluding full R-HAN.
        for cat in category_order:
            dc = d[(d["Category"] == cat) & (d["Method"] != "R-HAN")]
            ax.scatter(
                dc["Accuracy"],
                dc["TokenCost_M"],
                s=dc["TimeCost_rel"].map(point_area),
                c=category_colors[cat],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.85,
                zorder=3,
            )

        # Method labels.
        for _, row in d[d["Method"] != "R-HAN"].iterrows():
            dx, dy, ha = label_offsets[ds][row["Method"]]
            annotate_with_stroke(
                ax,
                row["Method"],
                (row["Accuracy"], row["TokenCost_M"]),
                (dx, dy),
                ha,
                color="black",
                fontsize=8.2,
            )

        # w/o Hierarchical Sparse Coordination star.
        # No in-plot text label; explained in the legend.
        # White halo underneath lifts the hollow star out of dense bubble regions.
        hsc_size = point_area(wo_hsc_time[ds]) * 1.55
        ax.scatter(
            [wo_hsc_acc[ds]],
            [wo_hsc_cost[ds]],
            s=[hsc_size * 1.45],
            marker="o",
            facecolors="white",
            edgecolors="white",
            linewidths=0,
            alpha=0.92,
            zorder=6,
        )
        ax.scatter(
            [wo_hsc_acc[ds]],
            [wo_hsc_cost[ds]],
            s=[hsc_size],
            marker="*",
            facecolors=wo_hsc_style["facecolor"],
            edgecolors=wo_hsc_style["edgecolor"],
            linewidths=2.4,
            zorder=7,
        )

        # Full R-HAN star.
        ax.scatter(
            [full_acc[ds]],
            [full_cost[ds]],
            s=[point_area(full_time[ds]) * 1.18],
            marker="*",
            facecolors=rhan_style["facecolor"],
            edgecolors=rhan_style["edgecolor"],
            linewidths=1.9,
            zorder=7,
        )

        dx, dy, ha = rhan_label_offsets[ds]
        annotate_with_stroke(
            ax,
            "R-HAN",
            (full_acc[ds], full_cost[ds]),
            (dx, dy),
            ha,
            color=rhan_style["edgecolor"],
            fontsize=8.3,
        )

        ax.set_title(ds, fontsize=16, fontweight="bold", pad=8)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(0, ymax)

        if ymax > 10:
            ax.set_yticks(np.arange(0, ymax + 0.01, 2.5))
        else:
            ax.set_yticks(np.arange(0, ymax + 0.01, 0.5))

        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_ylabel("Total Token Cost (M)", fontsize=12)

        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35, zorder=0)
        ax.tick_params(labelsize=10.5)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    fig.suptitle("Cost–Accuracy–Time Comparison", fontsize=21, fontweight="bold", y=0.985)

    # Category legend.
    category_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=category_colors[c],
            markeredgecolor="black",
            markersize=8,
            label=c,
        )
        for c in category_order
    ]

    leg1 = fig.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.29, 0.942),
        ncol=4,
        frameon=False,
        fontsize=10.8,
        title="Method category",
        title_fontsize=11,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    # Star legend only. No bubble-size legend.
    star_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=wo_hsc_style["facecolor"],
            markeredgecolor=wo_hsc_style["edgecolor"],
            markersize=12,
            label="w/o Hierarchical Sparse Coordination",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=rhan_style["facecolor"],
            markeredgecolor=rhan_style["edgecolor"],
            markersize=12,
            label="R-HAN",
        ),
    ]

    leg2 = fig.legend(
        handles=star_handles,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.942),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        title="R-HAN variants",
        title_fontsize=11,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    fig.add_artist(leg1)
    fig.add_artist(leg2)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.885])

    pdf_path = f"figures/{output_prefix}.pdf"
    png_path = f"figures/{output_prefix}.png"

    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# =========================
# 6. Generate figures
# =========================

# Figure A: MATH and MATH-Lv5 side by side.
# MATH x-axis starts from 40 as requested.
draw_pair(
    datasets_pair=["MATH", "MATH-Lv5"],
    xlims=[(40, 81), (30, 81)],
    xticks_list=[
        [40, 50, 60, 70, 80],
        [30, 40, 50, 60, 70, 80],
    ],
    output_prefix="figure1_math_pair_hsc_only",
)

# Figure B: MMLU and MMLU-Pro side by side.
# MMLU x-axis starts from 70.
# MMLU-Pro x-axis ends at 80.
draw_pair(
    datasets_pair=["MMLU", "MMLU-Pro"],
    xlims=[(70, 90), (57, 80)],
    xticks_list=[
        [70, 75, 80, 85, 90],
        [60, 65, 70, 75, 80],
    ],
    output_prefix="figure1_mmlu_pair_hsc_only",
)