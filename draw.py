import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# Data from your two tables
# =========================

# Table 4: Sensitivity to the number of fused topologies K
K_labels = ["1", "2", "3", "4", "5"]
K_acc = np.array([83.49, 85.30, 86.37, 85.97, 85.56])
K_cost = np.array([0.78, 0.91, 1.00, 1.16, 1.31])

# Table 5: Sensitivity to risk_std_penalty
risk_labels = ["0.00", "0.10", "0.30", "0.50", "0.70", "1.00"]
risk_acc = np.array([85.92, 86.19, 86.37, 86.22, 85.96, 85.52])
risk_cost = np.array([1.02, 1.018, 1.013, 0.997, 0.981, 0.976])


# =========================
# Plot style
# =========================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def plot_dual_axis(ax, x_labels, acc, cost, xlabel, acc_ylim=None, cost_ylim=None):
    """
    Draw one dual-axis sensitivity panel.
    Left y-axis: average accuracy / performance.
    Right y-axis: relative token cost.
    """
    x = np.arange(len(x_labels))

    perf_color = "#9d1028"
    cost_color = "#1f77b4"

    # Left axis: performance
    ax.plot(
        x,
        acc,
        color=perf_color,
        marker="*",
        markersize=18,
        linewidth=2.5,
        zorder=3,
    )

    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Avg. Accuracy", color=perf_color)
    ax.tick_params(axis="y", labelcolor=perf_color)

    if acc_ylim is not None:
        ax.set_ylim(*acc_ylim)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, color="#cccccc")

    # Right axis: relative token cost
    ax2 = ax.twinx()
    ax2.plot(
        x,
        cost,
        color=cost_color,
        marker="s",
        markersize=10,
        linewidth=2.5,
        zorder=3,
    )

    ax2.set_ylabel("Relative Token Cost", color=cost_color)
    ax2.tick_params(axis="y", labelcolor=cost_color)

    if cost_ylim is not None:
        ax2.set_ylim(*cost_ylim)

    return ax, ax2


# =========================
# Draw figure
# =========================

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))

plot_dual_axis(
    axes[0],
    K_labels,
    K_acc,
    K_cost,
    xlabel=r"The number of fused topologies $K$",
    acc_ylim=(83.0, 87.0),
    cost_ylim=(0.7, 1.4),
)

plot_dual_axis(
    axes[1],
    risk_labels,
    risk_acc,
    risk_cost,
    xlabel=r"Risk std penalty $\lambda$",
    acc_ylim=(85.3, 86.6),
    cost_ylim=(0.94, 1.06),
)


# =========================
# Global legend above plots
# =========================

legend_handles = [
    Line2D(
        [0],
        [0],
        color="#9d1028",
        marker="*",
        markersize=18,
        linewidth=2.5,
        label="Performance (Left Axis)",
    ),
    Line2D(
        [0],
        [0],
        color="#1f77b4",
        marker="s",
        markersize=10,
        linewidth=2.5,
        label="Cost per query (Right Axis)",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=2,
    frameon=True,
    framealpha=1.0,
    borderpad=0.5,
    columnspacing=2.5,
    handlelength=2.2,
)

# Leave space for the global legend
plt.tight_layout(rect=[0, 0, 1, 0.90], w_pad=2.4)


# =========================
# Save files
# =========================

plt.savefig("rhan_sensitivity_two_panels.png", bbox_inches="tight")
plt.savefig("rhan_sensitivity_two_panels.pdf", bbox_inches="tight")
plt.savefig("rhan_sensitivity_two_panels.svg", bbox_inches="tight")
plt.savefig("rhan_sensitivity_two_panels.eps", format="eps", bbox_inches="tight")

plt.show()