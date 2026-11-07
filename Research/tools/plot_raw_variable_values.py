"""Plot raw process variables, preserving values and chronological order."""

from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


RAW = ROOT / "data" / "raw_data"
OUT = ROOT / "pictures" / "raw_variable_values"
FEATURES = [
    "RealPower", "GateOnTime", "SetPower", "Speed", "Length",
    "SetFrequency", "SetDuty",
]
CONSTANTS = {"SetFrequency": 1000, "SetDuty": 100}
BLUE = "#216E9D"
ORANGE = "#CC6536"
INK = "#222B34"
GRID = "#E5EAF0"
EDGE = "#AAB4BE"


def number(value):
    return f"{value:,.1f}".rstrip("0").rstrip(".")


def read_source(path):
    data = pd.read_csv(path)
    times = pd.to_datetime(data["WorkingTime"].astype(str).str.strip(), errors="raise")
    if times.isna().any():
        raise ValueError(f"Missing WorkingTime in {path.name}")
    data = data.assign(WorkingTime=times).sort_values("WorkingTime", kind="stable")
    data = data.reset_index(drop=True)
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite process value in {path.name}")
    return data


def inspect_sources():
    records = []
    for path in sorted(RAW.glob("*/*.csv")):
        data = read_source(path)
        for col, expected in CONSTANTS.items():
            if not data[col].eq(expected).all():
                raise ValueError(f"{path.name}: {col} is not constant at {expected}")
        records.append({
            "file": str(path.relative_to(ROOT)),
            "observations": len(data),
            "first_time": data["WorkingTime"].iloc[0].isoformat(),
            "last_time": data["WorkingTime"].iloc[-1].isoformat(),
            "features": {
                name: {
                    "min": float(data[name].min()),
                    "max": float(data[name].max()),
                    "unique_values": int(data[name].nunique()),
                } for name in FEATURES
            },
        })
    return records


def configure_axes(ax, feature, all_values):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(EDGE)
        ax.spines[side].set_linewidth(0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="both", labelsize=9, width=0.6, length=3, color=EDGE)
    ax.tick_params(axis="y", pad=5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: number(y)))
    low, high = float(all_values.min()), float(all_values.max())
    span = high - low
    pad = max(abs(low) * 0.045, 1) if span == 0 else span * 0.13
    ax.set_ylim(low - pad, high + pad)
    if feature in CONSTANTS:
        ax.set_yticks([low])
        ax.text(
            0.015, 0.83, f"Constant = {number(low)}",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=10, color=INK,
        )
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.set_axisbelow(True)


def plot_combined(train):
    full_x = np.arange(1, len(train) + 1)
    detail_n = min(90, len(train))
    fig, axes = plt.subplots(
        len(FEATURES), 2, figsize=(13.8, 12.2),
        sharex="col", gridspec_kw={"width_ratios": [1.12, 1]},
    )
    for row, feature in enumerate(FEATURES):
        values = train[feature].to_numpy()
        color = ORANGE if feature in CONSTANTS else BLUE
        for column, ax in enumerate(axes[row]):
            configure_axes(ax, feature, values)
            n = len(train) if column == 0 else detail_n
            line, = ax.plot(
                full_x[:n], values[:n], color=color,
                linewidth=0.45 if column == 0 else 1.0,
                marker=None if column == 0 else "o",
                markersize=2.2, markeredgewidth=0,
                rasterized=column == 0 and feature not in CONSTANTS,
            )
            ax.set_xlim(1, n)
            if not np.array_equal(line.get_ydata(), values[:n]):
                raise AssertionError("Plotted values do not match the source")
        axes[row, 0].set_ylabel(
            f"({chr(97 + row)}) {feature}", rotation=0,
            ha="right", va="center", labelpad=20,
            fontsize=10.5, fontweight="medium",
        )
        if feature not in CONSTANTS:
            axes[row, 0].text(
                0.985, 0.86,
                f"Range: {number(values.min())} to {number(values.max())}",
                transform=axes[row, 0].transAxes, ha="right", va="center",
                fontsize=8.5, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.5},
            )

    axes[0, 0].set_title(f"Full normal training data (n = {len(train):,})", fontsize=11, pad=15)
    axes[0, 1].set_title(f"Detail: first {detail_n} observations", fontsize=11, pad=15)
    axes[-1, 0].set_xticks([1, 30000, 60000, 90000, 120000, len(train)])
    axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[-1, 1].set_xticks([1, 15, 30, 45, 60, 75, detail_n])
    for ax in axes[-1]:
        ax.set_xlabel("Observation index after sorting by WorkingTime", fontsize=10, labelpad=10)

    fig.suptitle("Raw process-variable values", fontsize=17, fontweight="semibold", y=0.979)
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, linewidth=2, label="Model input (5 variables)"),
            Line2D([0], [0], color=ORANGE, linewidth=2, label="Excluded constant (2 variables)"),
        ],
        loc="upper center", bbox_to_anchor=(0.565, 0.952),
        ncol=2, frameon=False, fontsize=10, handlelength=2.5, columnspacing=3,
    )
    fig.text(
        0.177, 0.035,
        "Source: Training_Data.csv. Values are shown before standardization; each row has its own y-axis scale.",
        ha="left", va="center", fontsize=9, color="#505C66",
    )
    fig.subplots_adjust(left=0.177, right=0.983, top=0.877, bottom=0.099, hspace=0.30, wspace=0.20)
    for extension in ("png", "svg"):
        destination = OUT / f"raw_process_variable_values.{extension}"
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)


def plot_value_distributions(train):
    """Plot empirical values from all 135,759 observations."""
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.8), constrained_layout=True)
    axes = axes.ravel()

    for ax, feature in zip(axes[:7], FEATURES):
        values = train[feature].astype(float)
        unique = np.sort(values.unique())
        color = ORANGE if feature in CONSTANTS else BLUE

        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(EDGE)
            ax.spines[side].set_linewidth(0.7)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        if len(unique) <= 15:
            counts = values.value_counts().sort_index()
            positions = np.arange(len(counts))
            ax.bar(
                positions, counts.to_numpy(),
                color=color, width=0.68,
                edgecolor="none", zorder=3,
            )
            ax.set_xticks(positions, [number(value) for value in counts.index])
            ax.tick_params(axis="x", labelrotation=35 if len(unique) > 5 else 25)
            ax.margins(x=0.08)
        else:
            ax.hist(values.to_numpy(), bins=35, color=color, edgecolor="white", linewidth=0.35, zorder=3)

        status = "Excluded: constant" if feature in CONSTANTS else "Model input"
        ax.set_title(f"{feature}\n{status}", fontsize=11, fontweight="medium")
        ax.set_xlabel("Observed value", fontsize=9)
        ax.set_ylabel("Number of observations", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        ax.text(
            0.98, 0.95,
            f"min {number(values.min())} | max {number(values.max())}\nunique values {len(unique):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.3, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.6},
        )

    axes[7].axis("off")
    axes[7].legend(
        handles=[
            Line2D([0], [0], color=BLUE, linewidth=8, label="Model input"),
            Line2D([0], [0], color=ORANGE, linewidth=8, label="Excluded constant"),
        ],
        loc="center", frameon=False, fontsize=11,
    )
    axes[7].text(
        0.5, 0.27,
        "Source: Training_Data.csv\nAll 135,759 observations\nRaw values before standardization",
        ha="center", va="center", fontsize=10, color="#505C66", linespacing=1.5,
    )
    fig.suptitle("Empirical distribution of raw process variables", fontsize=16, fontweight="semibold")
    for extension in ("png", "svg"):
        destination = OUT / f"raw_process_variable_distributions.{extension}"
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)


def plot_chronological_segment(train, observations=300):
    """Plot a readable, unaggregated chronological segment of the actual rows."""
    segment = train.iloc[:observations].copy()
    x = np.arange(1, len(segment) + 1)
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(13.5, 11.5), sharex=True)

    for row, (ax, feature) in enumerate(zip(axes, FEATURES)):
        values = segment[feature].to_numpy(dtype=float)
        color = ORANGE if feature in CONSTANTS else BLUE
        ax.plot(x, values, color=color, linewidth=0.85, marker="o", markersize=1.7, markeredgewidth=0)
        ax.set_ylabel(f"({chr(97 + row)}) {feature}", rotation=0, ha="right", va="center", labelpad=18)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(EDGE)
            ax.spines[side].set_linewidth(0.7)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: number(y)))
        low, high = float(values.min()), float(values.max())
        span = high - low
        pad = max(abs(low) * 0.04, 1) if span == 0 else span * 0.12
        ax.set_ylim(low - pad, high + pad)
        if span == 0:
            ax.set_yticks([low])
            ax.text(0.99, 0.78, f"constant = {number(low)}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=8.5, color=INK)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))

    axes[-1].set_xlabel("Observation index after sorting by WorkingTime", fontsize=10)
    axes[-1].set_xticks([1, 50, 100, 150, 200, 250, observations])
    fig.suptitle(
        f"Raw process variables in chronological order: first {observations} observations",
        fontsize=16, fontweight="semibold", y=0.985,
    )
    fig.text(
        0.145, 0.025,
        "Source: Training_Data.csv. No standardization, smoothing, aggregation, or generated values.",
        ha="left", fontsize=9, color="#505C66",
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.942, bottom=0.075, hspace=0.30)
    for extension in ("png", "svg"):
        destination = OUT / f"raw_process_variable_timeseries_first_{observations}.{extension}"
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)


def plot_requested_first_90(train):
    """Standalone seven-panel figure matching the requested layout."""
    observations = 90
    segment = train.iloc[:observations].copy()
    x = np.arange(1, observations + 1)
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(13.0, 17.0), sharex=True)

    for row, (ax, feature) in enumerate(zip(axes, FEATURES)):
        values = segment[feature].to_numpy(dtype=float)
        color = ORANGE if feature in CONSTANTS else BLUE
        plotted, = ax.plot(
            x, values, color=color, linewidth=3.0,
            marker="o", markersize=5.0, markeredgewidth=0,
        )
        if not np.array_equal(plotted.get_ydata(), values):
            raise AssertionError(f"{feature}: plotted values differ from the CSV")

        ax.set_ylabel(
            feature, rotation=0, ha="right", va="center", labelpad=36,
            fontsize=18, fontweight="semibold",
        )
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(EDGE)
            ax.spines[side].set_linewidth(1.6)
        ax.grid(axis="y", color=GRID, linewidth=1.3)
        ax.tick_params(axis="both", labelsize=16, width=1.4, length=6, color=EDGE)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: number(y)))

        low, high = float(values.min()), float(values.max())
        span = high - low
        pad = max(abs(low) * 0.045, 1) if span == 0 else span * 0.13
        ax.set_ylim(low - pad, high + pad)
        if span == 0:
            ax.set_yticks([low])
            ax.text(
                0.015, 0.82, f"Constant = {number(low)}",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=17, fontweight="semibold", color=INK,
            )
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))

    axes[0].set_title(
        "Detail: first 90 observations", fontsize=23,
        fontweight="semibold", pad=16,
    )
    axes[-1].set_xlim(1, observations)
    axes[-1].set_xticks([1, 15, 30, 45, 60, 75, 90])
    axes[-1].set_xlabel(
        "Observation index after sorting by WorkingTime",
        fontsize=18, fontweight="medium", labelpad=14,
    )
    fig.text(
        0.25, 0.014,
        "Source: Training_Data.csv. Raw observations without standardization or smoothing.",
        ha="left", fontsize=13, color="#505C66",
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.96, bottom=0.075, hspace=0.44)

    for extension in ("png", "svg"):
        destination = OUT / f"raw_process_variable_timeseries_first_{observations}_requested.{extension}"
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)


def style_group_axis(ax, feature, values, color):
    """Apply the larger, publication-readable style for grouped figures."""
    x = np.arange(1, len(values) + 1)
    plotted, = ax.plot(
        x, values, color=color, linewidth=3.0,
        marker="o", markersize=5.0, markeredgewidth=0,
    )
    if not np.array_equal(plotted.get_ydata(), values):
        raise AssertionError(f"{feature}: plotted values differ from the CSV")

    ax.set_ylabel(feature, rotation=0, ha="right", va="center", labelpad=34,
                  fontsize=18, fontweight="semibold")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(EDGE)
        ax.spines[side].set_linewidth(1.6)
    ax.grid(axis="y", color=GRID, linewidth=1.3)
    ax.tick_params(axis="both", labelsize=16, width=1.4, length=6, color=EDGE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: number(y)))

    low, high = float(values.min()), float(values.max())
    span = high - low
    pad = max(abs(low) * 0.06, 1) if span == 0 else span * 0.14
    ax.set_ylim(low - pad, high + pad)
    if span == 0:
        ax.set_yticks([low])
        ax.text(
            0.018, 0.78, f"Constant = {number(low)}",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=17, fontweight="semibold", color=INK,
        )
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))


def plot_separated_variable_groups(train):
    """Create separate figures for five model inputs and two excluded constants."""
    observations = 90
    segment = train.iloc[:observations].copy()
    x_ticks = [1, 15, 30, 45, 60, 75, 90]

    input_features = FEATURES[:5]
    fig, axes = plt.subplots(len(input_features), 1, figsize=(13.0, 13.0), sharex=True)
    for ax, feature in zip(axes, input_features):
        style_group_axis(ax, feature, segment[feature].to_numpy(dtype=float), BLUE)
    axes[0].set_title("Model input variables: first 90 observations", fontsize=23,
                      fontweight="semibold", pad=14)
    axes[-1].set_xlim(1, observations)
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xlabel("Observation index after sorting by WorkingTime", fontsize=18,
                        fontweight="medium", labelpad=13)
    fig.text(
        0.25, 0.024,
        "Source: Training_Data.csv. Raw observations without standardization or smoothing.",
        ha="left", fontsize=13, color="#505C66",
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.94, bottom=0.09, hspace=0.42)
    for extension in ("png", "svg"):
        destination = OUT / f"raw_model_input_variables_first_{observations}.{extension}"
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)

    constant_features = FEATURES[5:]
    fig, axes = plt.subplots(len(constant_features), 1, figsize=(13.0, 6.7), sharex=True)
    for ax, feature in zip(axes, constant_features):
        style_group_axis(ax, feature, segment[feature].to_numpy(dtype=float), ORANGE)
    axes[0].set_title("Excluded constant variables: first 90 observations", fontsize=23,
                      fontweight="semibold", pad=14)
    axes[-1].set_xlim(1, observations)
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xlabel("Observation index after sorting by WorkingTime", fontsize=18,
                        fontweight="medium", labelpad=13)
    fig.text(
        0.25, 0.045,
        "Source: Training_Data.csv. Each variable has one unique value across all 135,759 observations.",
        ha="left", fontsize=13, color="#505C66",
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.87, bottom=0.18, hspace=0.48)
    for extension in ("png", "svg"):
        destination = OUT / f"raw_constant_variables_first_{observations}.{extension}"
        fig.savefig(destination, dpi=300, facecolor="white")
        print(destination)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "path",
        "axes.formatter.useoffset": False,
    })
    checks = inspect_sources()
    source = RAW / "train" / "Training_Data.csv"
    train = read_source(source)
    if len(train) != 135759:
        raise ValueError("Training source size differs from the manuscript")
    report = {
        "plotted_source": str(source.relative_to(ROOT)),
        "ordering": "Stable ascending sort by parsed WorkingTime within each source file",
        "full_panel": "All original observations; no sampling or averaging",
        "detail_panel": "First 90 chronological observations",
        "normalization": "None: original raw values",
        "model_input_order": FEATURES[:5],
        "constant_verification": checks,
        "note": "Labeled and unlabeled versions were checked separately, never pooled or counted twice in the plot.",
    }
    (OUT / "source_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_combined(train)
    plot_value_distributions(train)
    plot_chronological_segment(train, observations=300)
    plot_requested_first_90(train)
    plot_separated_variable_groups(train)
    print("All seven raw CSV files verified: SetFrequency = 1000; SetDuty = 100.")


if __name__ == "__main__":
    main()
