import os
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D


# ============================================================
# 0. Global knobs
# ============================================================

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# This is tuned for ACL two-column figure* with width=\textwidth.
# If you still feel the fonts are small, increase this to 1.10 or 1.20.
FONT_SCALE = 1.08

TITLE_FONT = 14.0 * FONT_SCALE
AXIS_FONT = 11.0 * FONT_SCALE
TICK_FONT = 9.5 * FONT_SCALE
LABEL_FONT = 10.0 * FONT_SCALE
RHAN_LABEL_FONT = 10.8 * FONT_SCALE

# Marker size. Increase these if you want larger points.
BASE_POINT_SCALE = 145
RHAN_STAR_SCALE = 1.45
HSC_STAR_SCALE = 2.35

# Purple star style.
PURPLE_EDGE = "#8e44ad"
PURPLE_FILL = "#f4e4ff"
RHAN_EDGE = "#c82333"
RHAN_FILL = "#e53935"

# Category colors.
CATEGORY_COLORS = {
    "Single": "#8a8a8a",
    "Pre-defined": "#f0b323",
    "Auto": "#67b7e1",
    "Graph": "#17b978",
}

CATEGORY_ORDER = ["Single", "Pre-defined", "Auto", "Graph"]


# ============================================================
# 1. Data
# ============================================================

CSV_DATA = """Category,Method,Dataset,TokenCost_M,Accuracy,TimeCost_rel
Single,Self-Refine,MMLU,1.25,75.44,0.51
Single,Self-Refine,MMLU-Pro,1.80,57.97,0.74
Single,Self-Refine,MATH,1.55,46.10,0.64
Single,Self-Refine,MATH-Lv5,1.94,32.85,0.80
Pre-defined,LLM-Debate,MMLU,3.05,81.04,2.20
Pre-defined,LLM-Debate,MMLU-Pro,4.20,63.28,3.03
Pre-defined,LLM-Debate,MATH,13.73,52.96,9.90
Pre-defined,LLM-Debate,MATH-Lv5,16.40,38.55,11.83
Pre-defined,AgentVerse,MMLU,2.55,78.36,1.46
Pre-defined,AgentVerse,MMLU-Pro,3.50,60.83,2.01
Pre-defined,AgentVerse,MATH,5.80,50.85,3.33
Pre-defined,AgentVerse,MATH-Lv5,7.08,36.85,4.06
Pre-defined,LLM-Blender,MMLU,2.20,81.22,0.97
Pre-defined,LLM-Blender,MMLU-Pro,3.05,63.47,1.35
Pre-defined,LLM-Blender,MATH,4.40,50.34,1.95
Pre-defined,LLM-Blender,MATH-Lv5,5.19,36.16,2.30
Pre-defined,DyLAN,MMLU,2.95,79.96,1.79
Pre-defined,DyLAN,MMLU-Pro,4.05,62.14,2.46
Pre-defined,DyLAN,MATH,9.39,51.12,5.70
Pre-defined,DyLAN,MATH-Lv5,11.27,37.38,6.84
Auto,H-Swarms,MMLU,2.85,83.68,1.59
Auto,H-Swarms,MMLU-Pro,3.95,69.00,2.20
Auto,H-Swarms,MATH,6.20,65.89,3.46
Auto,H-Swarms,MATH-Lv5,7.56,50.27,4.21
Auto,AFlow,MMLU,2.35,83.10,1.04
Auto,AFlow,MMLU-Pro,3.20,64.35,1.42
Auto,AFlow,MATH,4.66,73.35,2.06
Auto,AFlow,MATH-Lv5,5.78,58.82,2.56
Graph,GPTSwarm,MMLU,2.30,82.80,1.09
Graph,GPTSwarm,MMLU-Pro,3.05,64.19,1.45
Graph,GPTSwarm,MATH,3.89,68.85,1.85
Graph,GPTSwarm,MATH-Lv5,4.78,54.64,2.27
Auto,RouterDC,MMLU,1.95,82.01,0.70
Auto,RouterDC,MMLU-Pro,2.65,63.27,0.96
Auto,RouterDC,MATH,2.85,73.46,1.03
Auto,RouterDC,MATH-Lv5,3.55,58.93,1.28
Auto,A2Flow,MMLU,2.30,83.29,1.02
Auto,A2Flow,MMLU-Pro,3.10,63.42,1.37
Auto,A2Flow,MATH,4.20,58.50,1.86
Auto,A2Flow,MATH-Lv5,5.12,44.76,2.27
Auto,DAAO,MMLU,2.05,84.90,0.81
Auto,DAAO,MMLU-Pro,3.00,65.28,1.18
Auto,DAAO,MATH,3.10,55.37,1.22
Auto,DAAO,MATH-Lv5,4.10,42.57,1.61
Auto,MasRouter,MMLU,2.05,84.25,0.74
Auto,MasRouter,MMLU-Pro,2.80,64.85,1.01
Auto,MasRouter,MATH,2.75,75.42,0.99
Auto,MasRouter,MATH-Lv5,3.44,61.36,1.24
Auto,MaAS,MMLU,2.10,83.01,0.86
Auto,MaAS,MMLU-Pro,2.90,63.68,1.19
Auto,MaAS,MATH,2.16,74.45,0.89
Auto,MaAS,MATH-Lv5,2.81,60.23,1.15
Graph,G-Designer,MMLU,2.60,87.20,1.28
Graph,G-Designer,MMLU-Pro,3.55,66.94,1.75
Graph,G-Designer,MATH,3.60,70.46,1.77
Graph,G-Designer,MATH-Lv5,4.50,55.39,2.21
Auto,BiRouter,MMLU,2.15,86.80,0.81
Auto,BiRouter,MMLU-Pro,3.00,66.53,1.13
Auto,BiRouter,MATH,2.60,74.92,0.98
Auto,BiRouter,MATH-Lv5,3.25,60.87,1.23
Graph,R-HAN,MMLU,2.32,88.37,0.95
Graph,R-HAN,MMLU-Pro,3.33,79.27,1.36
Graph,R-HAN,MATH,2.44,78.29,1.00
Graph,R-HAN,MATH-Lv5,2.98,68.21,1.22
"""

df = pd.read_csv(StringIO(CSV_DATA))


# ============================================================
# 2. R-HAN and ablation data
# ============================================================

DATASETS = ["MATH", "MATH-Lv5", "MMLU", "MMLU-Pro"]

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

# From Table 2.
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

# Combine the two drops into one setting:
# w/o Hierarchical Sparse Coordination.
wo_hsc_acc = {
    ds: round(
        full_acc[ds]
        - (
            (full_acc[ds] - wo_latent_memory_acc[ds])
            + (full_acc[ds] - wo_global_controller_acc[ds])
        ),
        2,
    )
    for ds in DATASETS
}

# Cost assumptions:
# w/o self-refine and w/o union graph reduce cost;
# w/o HSC increases cost by the sum of those two reductions.
# These two are not plotted; they are only used to estimate w/o HSC cost.
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
    for ds in DATASETS
}

wo_hsc_time = {
    ds: round(full_time[ds] * (wo_hsc_cost[ds] / full_cost[ds]), 2)
    for ds in DATASETS
}


# ============================================================
# 3. Matplotlib style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def point_area(time_cost: float) -> float:
    """Bubble area proportional to estimated relative time cost."""
    return BASE_POINT_SCALE * (time_cost ** 1.18)


def add_text(ax, text, xy, xytext, ha, color="black", fontsize=LABEL_FONT):
    """Text with white stroke, so labels remain readable."""
    ann = ax.annotate(
        text,
        xy,
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=30,
    )
    ann.set_path_effects([
        patheffects.withStroke(linewidth=2.8, foreground="white", alpha=0.97)
    ])
    return ann


# Manual offsets to avoid overlap. Adjust here if needed.
label_offsets = {
    "MATH": {
        "Self-Refine": (5, 5, "left"),
        "LLM-Debate": (6, 5, "left"),
        "DyLAN": (6, 0, "left"),
        "AgentVerse": (-5, 6, "right"),
        "LLM-Blender": (5, -6, "left"),
        "DAAO": (-5, 0, "right"),
        "A2Flow": (5, 4, "left"),
        "H-Swarms": (6, 0, "left"),
        "AFlow": (6, 0, "left"),
        "GPTSwarm": (6, 3, "left"),
        "G-Designer": (5, -7, "left"),
        "RouterDC": (-6, -9, "right"),
        "MasRouter": (6, 1, "left"),
        "MaAS": (6, -9, "left"),
        "BiRouter": (-6, -11, "right"),
    },
    "MATH-Lv5": {
        "Self-Refine": (5, 5, "left"),
        "LLM-Debate": (6, 5, "left"),
        "DyLAN": (-6, -2, "right"),
        "AgentVerse": (-6, 5, "right"),
        "LLM-Blender": (5, -6, "left"),
        "DAAO": (5, 4, "left"),
        "A2Flow": (5, 4, "left"),
        "H-Swarms": (6, 0, "left"),
        "AFlow": (6, 0, "left"),
        "GPTSwarm": (6, 3, "left"),
        "G-Designer": (6, -7, "left"),
        "RouterDC": (-6, -8, "right"),
        "MasRouter": (6, 4, "left"),
        "MaAS": (6, -6, "left"),
        "BiRouter": (-6, -10, "right"),
    },
    "MMLU": {
        "Self-Refine": (5, 5, "left"),
        "LLM-Debate": (-6, -8, "right"),
        "DyLAN": (-6, 5, "right"),
        "AgentVerse": (5, -8, "left"),
        "LLM-Blender": (5, -6, "left"),
        "DAAO": (5, -6, "left"),
        "A2Flow": (5, 4, "left"),
        "H-Swarms": (5, 4, "left"),
        "AFlow": (5, 4, "left"),
        "GPTSwarm": (5, 4, "left"),
        "G-Designer": (5, 4, "left"),
        "RouterDC": (-5, -6, "right"),
        "MasRouter": (-5, -8, "right"),
        "MaAS": (5, -8, "left"),
        "BiRouter": (5, 0, "left"),
    },
    "MMLU-Pro": {
        "Self-Refine": (5, 5, "left"),
        "LLM-Debate": (5, 5, "left"),
        "DyLAN": (-5, 5, "right"),
        "AgentVerse": (5, -4, "left"),
        "LLM-Blender": (-5, -9, "right"),
        "DAAO": (5, -6, "left"),
        "A2Flow": (-5, 4, "right"),
        "H-Swarms": (5, 4, "left"),
        "AFlow": (5, 0, "left"),
        "GPTSwarm": (5, 4, "left"),
        "G-Designer": (5, -9, "left"),
        "RouterDC": (-5, -6, "right"),
        "MasRouter": (-5, -9, "right"),
        "MaAS": (5, -6, "left"),
        "BiRouter": (5, 0, "left"),
    },
}

rhan_label_offsets = {
    "MATH": (7, -7, "left"),
    "MATH-Lv5": (7, -7, "left"),
    "MMLU": (7, -7, "left"),
    "MMLU-Pro": (7, -7, "left"),
}


# ============================================================
# 4. Panel drawing
# ============================================================

def draw_panel(ax, dataset, xlim, xticks, show_ylabel=True):
    d = df[df["Dataset"] == dataset].copy()

    # Baseline points, excluding R-HAN.
    for cat in CATEGORY_ORDER:
        dc = d[(d["Category"] == cat) & (d["Method"] != "R-HAN")]
        ax.scatter(
            dc["Accuracy"],
            dc["TokenCost_M"],
            s=dc["TimeCost_rel"].map(point_area),
            c=CATEGORY_COLORS[cat],
            edgecolors="black",
            linewidths=0.55,
            alpha=0.88,
            zorder=5,
        )

    # Baseline labels.
    for _, row in d[d["Method"] != "R-HAN"].iterrows():
        dx, dy, ha = label_offsets[dataset][row["Method"]]
        add_text(
            ax,
            row["Method"],
            (row["Accuracy"], row["TokenCost_M"]),
            (dx, dy),
            ha,
            fontsize=LABEL_FONT,
        )

    # w/o Hierarchical Sparse Coordination: emphasized purple star.
    x_p = wo_hsc_acc[dataset]
    y_p = wo_hsc_cost[dataset]
    t_p = wo_hsc_time[dataset]

    # Big white halo first.
    ax.scatter(
        [x_p],
        [y_p],
        s=[point_area(t_p) * HSC_STAR_SCALE * 1.35],
        marker="*",
        facecolors="white",
        edgecolors="white",
        linewidths=5.0,
        zorder=40,
    )

    # Purple outline.
    ax.scatter(
        [x_p],
        [y_p],
        s=[point_area(t_p) * HSC_STAR_SCALE],
        marker="*",
        facecolors=PURPLE_FILL,
        edgecolors=PURPLE_EDGE,
        linewidths=2.8,
        zorder=41,
    )

    # Small inner purple core, still visually hollow-ish.
    ax.scatter(
        [x_p],
        [y_p],
        s=[point_area(t_p) * HSC_STAR_SCALE * 0.46],
        marker="*",
        facecolors="white",
        edgecolors=PURPLE_EDGE,
        linewidths=1.5,
        zorder=42,
    )

    # Full R-HAN: red solid star.
    x_r = full_acc[dataset]
    y_r = full_cost[dataset]
    t_r = full_time[dataset]

    ax.scatter(
        [x_r],
        [y_r],
        s=[point_area(t_r) * RHAN_STAR_SCALE],
        marker="*",
        facecolors=RHAN_FILL,
        edgecolors=RHAN_EDGE,
        linewidths=1.8,
        zorder=43,
    )

    dx, dy, ha = rhan_label_offsets[dataset]
    add_text(
        ax,
        "R-HAN",
        (x_r, y_r),
        (dx, dy),
        ha,
        color=RHAN_EDGE,
        fontsize=RHAN_LABEL_FONT,
    )

    # Axes.
    ax.set_title(dataset, fontsize=TITLE_FONT, fontweight="bold", pad=4)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)

    y_values = d["TokenCost_M"].tolist() + [wo_hsc_cost[dataset], full_cost[dataset]]
    ymax = max(max(y_values) * 1.10, 4.6)
    ax.set_ylim(0, ymax)

    if ymax > 10:
        ax.set_yticks(np.arange(0, ymax + 0.01, 2.5))
    else:
        ax.set_yticks(np.arange(0, ymax + 0.01, 0.5))

    ax.set_xlabel("Accuracy (%)", fontsize=AXIS_FONT)
    if show_ylabel:
        ax.set_ylabel("Total Token Cost (M)", fontsize=AXIS_FONT)
    else:
        ax.set_ylabel("")

    ax.tick_params(labelsize=TICK_FONT, width=0.8, length=3.0)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.34, zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


# ============================================================
# 5. Figure functions
# ============================================================

LEGEND_HEADER_FONT = 11.5 * FONT_SCALE
LEGEND_ITEM_FONT = 10.5 * FONT_SCALE


def add_top_legend(fig):
    """Two grouped legends at the top of the figure: Method category + R-HAN variants."""
    cat_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="",
            markerfacecolor=CATEGORY_COLORS[c],
            markeredgecolor="black", markeredgewidth=0.6,
            markersize=11, label=c,
        )
        for c in CATEGORY_ORDER
    ]
    rhan_handles = [
        Line2D(
            [0], [0],
            marker="*", linestyle="",
            markerfacecolor=PURPLE_FILL,
            markeredgecolor=PURPLE_EDGE, markeredgewidth=1.6,
            markersize=18, label="w/o Hierarchical Sparse Coordination",
        ),
        Line2D(
            [0], [0],
            marker="*", linestyle="",
            markerfacecolor=RHAN_FILL,
            markeredgecolor=RHAN_EDGE, markeredgewidth=1.2,
            markersize=18, label="R-HAN",
        ),
    ]

    leg_cat = fig.legend(
        handles=cat_handles,
        title="Method category",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995),
        ncol=len(cat_handles),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.4,
        fontsize=LEGEND_ITEM_FONT,
        title_fontsize=LEGEND_HEADER_FONT,
    )
    leg_cat._legend_box.align = "left"
    leg_cat.get_title().set_fontweight("bold")
    fig.add_artist(leg_cat)

    leg_rhan = fig.legend(
        handles=rhan_handles,
        title="R-HAN variants",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        ncol=len(rhan_handles),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.4,
        fontsize=LEGEND_ITEM_FONT,
        title_fontsize=LEGEND_HEADER_FONT,
    )
    leg_rhan._legend_box.align = "left"
    leg_rhan.get_title().set_fontweight("bold")
    fig.add_artist(leg_rhan)


def save_pair_figure(datasets_pair, xlims, xticks_list, filename_prefix):
    """
    Pair figure designed to be used as:
    \\includegraphics[width=\\textwidth]{...}
    in an ACL figure*.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.2),  # extra height to host the top legend
        dpi=300,
        constrained_layout=False,
    )

    draw_panel(
        axes[0],
        datasets_pair[0],
        xlim=xlims[0],
        xticks=xticks_list[0],
        show_ylabel=True,
    )
    draw_panel(
        axes[1],
        datasets_pair[1],
        xlim=xlims[1],
        xticks=xticks_list[1],
        show_ylabel=True,
    )

    plt.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.165, wspace=0.30)
    add_top_legend(fig)

    pdf_path = os.path.join(OUT_DIR, f"{filename_prefix}.pdf")
    png_path = os.path.join(OUT_DIR, f"{filename_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def save_2x2_figure(filename_prefix):
    """
    Recommended for ACL if you want all four panels in one Figure*.
    It avoids shrinking two separate pair figures side by side.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 10.0),
        dpi=300,
        constrained_layout=False,
    )

    draw_panel(axes[0, 0], "MATH", xlim=(40, 81), xticks=[40, 50, 60, 70, 80], show_ylabel=True)
    draw_panel(axes[0, 1], "MATH-Lv5", xlim=(30, 81), xticks=[30, 40, 50, 60, 70, 80], show_ylabel=True)
    draw_panel(axes[1, 0], "MMLU", xlim=(70, 90), xticks=[70, 75, 80, 85, 90], show_ylabel=True)
    draw_panel(axes[1, 1], "MMLU-Pro", xlim=(57, 80), xticks=[60, 65, 70, 75, 80], show_ylabel=True)

    plt.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.075, wspace=0.30, hspace=0.40)
    add_top_legend(fig)

    pdf_path = os.path.join(OUT_DIR, f"{filename_prefix}.pdf")
    png_path = os.path.join(OUT_DIR, f"{filename_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


# ============================================================
# 6. Generate outputs
# ============================================================

if __name__ == "__main__":
    # Pair 1: MATH and MATH-Lv5.
    save_pair_figure(
        datasets_pair=["MATH", "MATH-Lv5"],
        xlims=[(40, 81), (30, 81)],
        xticks_list=[
            [40, 50, 60, 70, 80],
            [30, 40, 50, 60, 70, 80],
        ],
        filename_prefix="figure1_math_pair_final_largefont",
    )

    # Pair 2: MMLU and MMLU-Pro.
    save_pair_figure(
        datasets_pair=["MMLU", "MMLU-Pro"],
        xlims=[(70, 90), (57, 80)],
        xticks_list=[
            [70, 75, 80, 85, 90],
            [60, 65, 70, 75, 80],
        ],
        filename_prefix="figure1_mmlu_pair_final_largefont",
    )

    # Recommended all-in-one 2x2 version.
    save_2x2_figure(
        filename_prefix="figure1_cost_accuracy_final_largefont_2x2"
    )