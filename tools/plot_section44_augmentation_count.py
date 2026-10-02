"""Publication figures from the stored Research08 augmentation experiment."""

from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


SOURCE = ROOT / "data/research08/results/research08_augmentation_saturation.csv"
OUT = ROOT / "output/section_4_4_figures"
COUNTS = [50, 100, 200, 500, 750, 1000]
STYLES = {
    "Filtered Masking Diffusion": ("FMD", "#A92434", "o", "-"),
    "Masking Diffusion": ("Masking Diffusion", "#1874A5", "s", "-"),
    "ScoreFiltered Masking Diffusion": ("Score-filtered MD", "#9160A4", "D", "--"),
    "Magnitude warping": ("Magnitude warping", "#007E71", "^", "--"),
    "Noise injection": ("Noise injection", "#D27900", "v", "-"),
    "Frequency domain": ("Frequency domain", "#668722", "P", "--"),
    "Time warping": ("Time warping", "#6F777E", "X", ":"),
    "Comprehensive": ("Comprehensive", "#AD683F", "h", "-."),
}
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("auprc", "AUPRC (AP)")]


def configure_axis(ax):
    ax.set_xlim(10, 1040)
    ax.set_xticks(COUNTS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    ax.set_ylim(0.40, 1.025)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ADB5BD")
    ax.tick_params(length=3, color="#ADB5BD", labelsize=9)
    ax.set_xlabel("Added anomaly windows", labelpad=7)


def save_figure(fig, stem):
    paths = []
    for extension in ["png", "svg"]:
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=350, facecolor="white", bbox_inches="tight", pad_inches=0.12)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)
    assert set(df["method"]) == set(STYLES)
    assert not df.duplicated(["method", "augmentation_count"]).any()
    assert (df["normal_train_used"] == 5000).all()
    assert (df["real_anomaly_seed_used"] == 247).all()
    for method in STYLES:
        assert sorted(df.loc[df.method.eq(method), "augmentation_count"].tolist()) == COUNTS
    metric_columns = [key for key, _ in METRICS]
    assert np.isfinite(df[metric_columns].to_numpy()).all()
    assert ((df[metric_columns] >= 0) & (df[metric_columns] <= 1)).all().all()
    np.testing.assert_allclose(df.f1, 2 * df.precision * df.recall / (df.precision + df.recall), atol=1e-12)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 12,
        "legend.fontsize": 9, "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), sharex=True, sharey=True)
    handles = {}
    for panel, (ax, (key, label)) in enumerate(zip(axes.flat, METRICS)):
        for method in reversed(list(STYLES)):
            display, color, marker, linestyle = STYLES[method]
            group = df[df.method.eq(method)].sort_values("augmentation_count")
            is_fmd = method == "Filtered Masking Diffusion"
            line, = ax.plot(
                group.augmentation_count, group[key], label=display,
                color=color, marker=marker, linestyle=linestyle,
                linewidth=2.65 if is_fmd else 1.45,
                markersize=6.0 if is_fmd else 4.3,
                markeredgecolor="white", markeredgewidth=0.5,
                alpha=1 if is_fmd else 0.90, zorder=10 if is_fmd else 3,
            )
            handles[method] = line
        configure_axis(ax)
        ax.tick_params(labelbottom=True)
        ax.set_title(f"({chr(97 + panel)}) {label}", loc="left", fontweight="bold", pad=10)
        ax.set_ylabel("Score")
    fig.suptitle("RandomForest performance by augmentation count", fontsize=16, fontweight="bold", y=0.99)
    fig.legend(
        [handles[m] for m in STYLES], [STYLES[m][0] for m in STYLES],
        loc="upper center", bbox_to_anchor=(0.5, 0.951), ncol=4,
        frameon=False, columnspacing=2.0, handlelength=2.8,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.825, bottom=0.12, hspace=0.36, wspace=0.16)
    fig.text(0.07, 0.045, "5,000 real normal + 247 real anomaly windows; threshold selected by validation F1.", fontsize=9, color="#4D5963")
    fig.text(0.07, 0.022, "Stored Research08 results: one run per condition; lines connect observed points, with no uncertainty estimates.", fontsize=8.5, color="#4D5963")
    outputs = save_figure(fig, "rf_metrics_by_augmentation_count")

    fmd = df[df.method.eq("Filtered Masking Diffusion")].sort_values("augmentation_count")
    fig, ax = plt.subplots(figsize=(9.7, 5.9))
    colors = ["#1874A5", "#007E71", "#A92434", "#9160A4"]
    for (key, label), color, marker in zip(METRICS, colors, ["s", "^", "o", "D"]):
        ax.plot(fmd.augmentation_count, fmd[key], color=color, marker=marker,
                linewidth=2.4 if key == "f1" else 1.9, markersize=6,
                markeredgecolor="white", markeredgewidth=0.6, label=label)
    configure_axis(ax)
    ax.set_ylim(0.50, 1.025)
    ax.set_ylabel("Score")
    peak = fmd.loc[fmd.f1.idxmax()]
    ax.annotate(
        f"Highest observed F1\n{int(peak.augmentation_count):,} windows: {peak.f1:.3f}",
        xy=(peak.augmentation_count, peak.f1), xytext=(430, 0.968),
        ha="center", va="center", fontsize=9.5, color="#A92434",
        arrowprops={"arrowstyle": "->", "color": "#A92434", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#E6CCD0"},
    )
    fig.suptitle("FMD: detection performance by augmentation count", fontsize=15, fontweight="bold", y=0.985)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.82, bottom=0.19)
    fig.text(0.085, 0.070, "Default filter; 50-1,000 samples selected from one saved 1,000-window candidate pool.", fontsize=9, color="#4D5963")
    fig.text(0.085, 0.036, "Single stored run per condition; the observed maximum does not establish an optimal augmentation count.", fontsize=8.5, color="#4D5963")
    outputs += save_figure(fig, "fmd_metrics_by_augmentation_count")

    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "methods": list(STYLES), "counts": COUNTS, "metrics": metric_columns,
        "outputs": outputs, "number_of_source_rows": len(df),
        "number_of_score_values_in_comparison_figure": len(df) * len(METRICS),
        "data_transform": "None. Direct source values; straight lines between observations.",
        "auc_definition": "Stored auprc is sklearn average_precision_score, labeled AUPRC (AP).",
        "uncertainty": "Single run per condition; no confidence intervals or significance estimates.",
        "experimental_caveats": [
            "RF seed and normal-sample seed vary across methods and counts in Research08.",
            "Normal count is fixed, so augmentation count also changes the class ratio.",
            "Method labels mean augmentation methods; the downstream estimator is RandomForest in every condition.",
            "Recorded source-row overlap in the existing splits must be resolved before final independent-test claims.",
        ],
        "fmd_rows": fmd.to_dict(orient="records"),
    }
    (OUT / "augmentation_count_figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in outputs:
        print(path)
    print("Validated 48 condition rows and 192 plotted scores against the source CSV.")


if __name__ == "__main__":
    main()
