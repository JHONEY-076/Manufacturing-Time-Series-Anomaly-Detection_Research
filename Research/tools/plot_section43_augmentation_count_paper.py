"""Render the section 4.3.2 augmentation-count results as a paper-ready figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/masking_count_unified/masking_count_unified.csv"
OUT = ROOT / "output/section_4_3_unified/fmd_augmented_count_performance"


def main():
    df = pd.read_csv(SOURCE)
    x = np.arange(len(df))
    counts = df["generated_count"].astype(int).tolist()
    metrics = [
        ("precision", "Precision", "#3569A8", "o"),
        ("recall", "Recall", "#E47B24", "s"),
        ("f1", "F1-score", "#278C68", "^"),
        ("f2", "F2-score", "#8B63A9", "D"),
        ("auroc", "AUROC", "#6D7788", "v"),
        ("auprc_ap", "AP", "#BF4D59", "P"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })

    fig, (ax, err_ax) = plt.subplots(
        1, 2, figsize=(7.6, 3.65), gridspec_kw={"width_ratios": [1.9, 1.0]}
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.78, wspace=0.22)

    for key, label, color, marker in metrics:
        ax.plot(
            x, df[key], label=label, color=color, marker=marker,
            linewidth=1.5, markersize=3.8, markeredgecolor="white",
            markeredgewidth=0.55, zorder=3,
        )
    ax.set_xlim(-0.2, len(x) - 0.8)
    ax.set_ylim(0.50, 1.02)
    ax.set_xticks(x, [f"{n:,}" for n in counts])
    ax.set_xlabel("Added generated anomaly samples", labelpad=6)
    ax.set_ylabel("Score", labelpad=5)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A929B")
    ax.tick_params(length=3, width=0.7, color="#8A929B", pad=3)
    ax.set_title("(a) Detection performance", loc="left", fontweight="bold", pad=8)

    bar_width = 0.34
    fp_color, fn_color = "#7896B5", "#D98B75"
    fp = err_ax.bar(x - bar_width / 2, df["fp"], bar_width, label="FP", color=fp_color, zorder=3)
    fn = err_ax.bar(x + bar_width / 2, df["fn"], bar_width, label="FN", color=fn_color, zorder=3)
    err_ax.set_xlim(-0.55, len(x) - 0.45)
    err_ax.set_ylim(0, 158)
    err_ax.set_xticks(x, [f"{n:,}" for n in counts])
    err_ax.set_xlabel("Added samples", labelpad=6)
    err_ax.set_ylabel("Number of errors", labelpad=5)
    err_ax.yaxis.set_major_locator(MultipleLocator(25))
    err_ax.grid(axis="y", color="#D9DDE2", linewidth=0.65)
    err_ax.set_axisbelow(True)
    err_ax.spines[["top", "right"]].set_visible(False)
    err_ax.spines[["left", "bottom"]].set_color("#8A929B")
    err_ax.tick_params(length=3, width=0.7, color="#8A929B", pad=3)
    err_ax.set_title("(b) Classification errors", loc="left", fontweight="bold", pad=8)
    err_ax.bar_label(fp, padding=2, fontsize=6.8, fontweight="bold", color="#30343B")
    err_ax.bar_label(fn, padding=2, fontsize=6.8, fontweight="bold", color="#30343B")

    fig.legend(
        *ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.075, 0.975),
        ncol=6, frameon=False, handlelength=1.8, columnspacing=1.25,
        handletextpad=0.45, borderaxespad=0,
    )
    err_ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, 1.19), ncol=2, frameon=False,
        handlelength=1.2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
