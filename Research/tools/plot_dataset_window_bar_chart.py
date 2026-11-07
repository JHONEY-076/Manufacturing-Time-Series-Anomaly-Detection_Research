"""Create grouped bar charts of the experimental dataset window counts."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np


TEXT = {
    "ko": {
        "pool": "정상 학습 후보",
        "sets": ["불량 학습 시드", "검증 데이터", "최종 평가 데이터"],
        "panel_a": "(a) 정상 학습 후보",
        "panel_b": "(b) 불량 시드 및 평가 데이터",
        "normal": "정상",
        "defect": "불량",
        "ylabel": "윈도우 수 (개)",
        "scale_note": "두 패널은 서로 다른 세로축 범위를 사용하며, 막대 위 숫자는 실제 윈도우 수를 나타냄.",
        "pool_note": "정상 학습 후보 135,730개 중 주요 분류 실험에는 5,000개를 사용함.",
    },
    "en": {
        "pool": "Normal training pool",
        "sets": ["Anomaly training seeds", "Validation set", "Final test set"],
        "panel_a": "(a) Normal training pool",
        "panel_b": "(b) Anomaly seeds and evaluation sets",
        "normal": "Normal",
        "defect": "Anomaly",
        "ylabel": "Number of windows",
        "scale_note": "Panels use different y-axis ranges. Values above the bars indicate window counts.",
        "pool_note": "The main classification experiments used 5,000 windows from the normal training pool of 135,730.",
    },
}


def main(language: str = "ko") -> None:
    text = TEXT[language]
    with (ROOT / "data/augmentation_split/split_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        records = {row["split"]: row for row in csv.DictReader(stream)}

    for row in records.values():
        if int(row["normal"]) + int(row["anomaly"]) != int(row["windows"]):
            raise ValueError("Dataset totals do not match class counts.")

    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
            "font.size": 12,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "path",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 5.9),
        gridspec_kw={"width_ratios": [1.1, 3.0], "wspace": 0.30},
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.11)
    groups = [
        (["normal_train"], [text["pool"]], text["panel_a"], 40_000),
        (
            ["anomaly_train_seed", "validation", "final_test"],
            text["sets"],
            text["panel_b"], 500,
        ),
    ]
    colors = ["#216B99", "#CB6033"]
    width = 0.32

    for ax, (names, labels, title, tick_step) in zip(axes, groups):
        x = np.arange(len(names))
        maximum = 0
        for index, (column, legend) in enumerate((("normal", text["normal"]), ("anomaly", text["defect"]))):
            counts = [int(records[name][column]) for name in names]
            maximum = max(maximum, max(counts))
            centers = x + (index - 0.5) * width
            bars = ax.bar(
                centers, counts, width=width, label=legend,
                color=colors[index], edgecolor="white", linewidth=0.8, zorder=3,
            )
            ax.bar_label(
                bars, labels=[f"{value:,}" for value in counts],
                padding=6, fontsize=12, fontweight="bold", color="#25313B",
            )
            for center, count in zip(centers, counts):
                if count == 0:
                    ax.plot(
                        [center - width / 2, center + width / 2], [0, 0],
                        color=colors[index], linewidth=2, zorder=4, clip_on=False,
                    )

        ax.set_ylim(0, math.ceil(maximum * 1.14 / tick_step) * tick_step)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.7, len(names) - 0.3)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=17)
        ax.set_ylabel(text["ylabel"], labelpad=10)
        ax.tick_params(axis="both", length=0, pad=10)
        ax.grid(axis="y", color="#E1E6EB", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#ADB7C0")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.875),
        ncol=2, frameon=False, handlelength=1.6, columnspacing=1.4,
    )
    output = ROOT / "pictures"
    output.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        suffix = "_en" if language == "en" else ""
        path = output / f"dataset_window_bar_chart{suffix}_clean.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(TEXT), default="ko")
    main(parser.parse_args().language)
