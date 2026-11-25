"""Publication-style figure for the FMD filtering-strength comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/section_4_3_unified/filtering_strength.csv"
OUT = ROOT / "output/section_4_3_unified/filtering_strength_performance"


def main():
    df = pd.read_csv(SOURCE)
    order = ["loose", "default", "strict"]
    df["strength"] = pd.Categorical(df["strength"], categories=order, ordered=True)
    df = df.sort_values("strength")
    x = np.arange(len(df))
    labels = ["Loose", "Default", "Strict"]

    metrics = [
        ("precision", "Precision", "#3569A8", "o"),
        ("recall", "Recall", "#E47B24", "s"),
        ("f1", "F1-score", "#278C68", "^"),
        ("f2", "F2-score", "#8B63A9", "D"),
        ("auroc", "AUROC", "#6D7788", "v"),
        ("auprc_ap", "AP", "#BF4D59", "P"),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3, "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.8),
        gridspec_kw={"width_ratios": [1.65, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.81, wspace=0.32)
    score_ax, candidate_ax, error_ax = axes

    for key, name, color, marker in metrics:
        score_ax.plot(
            x, df[key], label=name, color=color, marker=marker,
            linewidth=1.55, markersize=4.2, markeredgecolor="white",
            markeredgewidth=0.5, zorder=3,
        )
    score_ax.set_ylim(0.65, 1.01)
    score_ax.yaxis.set_major_locator(MultipleLocator(0.05))
    score_ax.set_ylabel("Score", labelpad=4)
    score_ax.set_xticks(x, labels)
    score_ax.set_title("(a) Detection metrics", loc="left", fontweight="bold", pad=7)

    width = 0.34
    candidate_pass = candidate_ax.bar(
        x - width / 2, df["candidate_count"], width,
        label="Passing candidates", color="#7896B5", zorder=3,
    )
    candidate_selected = candidate_ax.bar(
        x + width / 2, df["selected_count"], width,
        label="Selected samples", color="#D98B75", zorder=3,
    )
    candidate_ax.set_ylim(0, 1300)
    candidate_ax.yaxis.set_major_locator(MultipleLocator(250))
    candidate_ax.set_ylabel("Samples", labelpad=4)
    candidate_ax.set_xticks(x, labels)
    candidate_ax.set_title("(b) Candidate selection", loc="left", fontweight="bold", pad=7)
    candidate_ax.bar_label(candidate_pass, padding=2, fontsize=6.8, fontweight="bold")
    candidate_ax.bar_label(candidate_selected, padding=2, fontsize=6.8, fontweight="bold")
    candidate_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=1,
        frameon=False, handlelength=1.1, labelspacing=0.2,
    )

    fp = error_ax.bar(x - width / 2, df["fp"], width, label="FP", color="#7896B5", zorder=3)
    fn = error_ax.bar(x + width / 2, df["fn"], width, label="FN", color="#D98B75", zorder=3)
    error_ax.set_ylim(0, 130)
    error_ax.yaxis.set_major_locator(MultipleLocator(25))
    error_ax.set_ylabel("Errors", labelpad=4)
    error_ax.set_xticks(x, labels)
    error_ax.set_title("(c) FP and FN", loc="left", fontweight="bold", pad=7)
    error_ax.bar_label(fp, padding=2, fontsize=6.8, fontweight="bold")
    error_ax.bar_label(fn, padding=2, fontsize=6.8, fontweight="bold")
    error_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2,
        frameon=False, handlelength=1.0, columnspacing=0.8,
    )

    for ax in axes:
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8A929B")
        ax.tick_params(length=2.5, width=0.65, color="#8A929B", pad=2)

    fig.legend(
        *score_ax.get_legend_handles_labels(), loc="upper left",
        bbox_to_anchor=(0.065, 0.985), ncol=6, frameon=False,
        handlelength=1.5, columnspacing=1.15, handletextpad=0.4,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
