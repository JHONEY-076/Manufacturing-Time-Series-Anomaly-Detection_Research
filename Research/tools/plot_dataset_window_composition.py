"""Plot dataset counts and class composition for manuscript Section 4.1."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.patches import Patch
import numpy as np


def main() -> None:
    source = ROOT / "data/augmentation_split/split_summary.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["split"]: row for row in csv.DictReader(stream)}

    order = ["normal_train", "anomaly_train_seed", "validation", "final_test"]
    labels = ["정상 학습 후보", "불량 학습 시드", "검증 데이터", "최종 평가 데이터"]
    normal = np.array([int(rows[name]["normal"]) for name in order])
    defect = np.array([int(rows[name]["anomaly"]) for name in order])
    total = np.array([int(rows[name]["windows"]) for name in order])
    if not np.array_equal(normal + defect, total):
        raise ValueError("Class counts do not match dataset totals.")

    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "hatch.linewidth": 0.45,
            "savefig.facecolor": "white",
        }
    )

    colors = {"total": "#70808E", "normal": "#176995", "defect": "#BD5427"}
    fig, (ax_count, ax_class) = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.9),
        gridspec_kw={"width_ratios": [1.0, 1.62], "wspace": 0.16},
        sharey=True,
    )
    fig.subplots_adjust(left=0.145, right=0.978, top=0.79, bottom=0.22)
    y = np.arange(len(order))
    bar_height = 0.60

    for ax in (ax_count, ax_class):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E4E7EB", linewidth=0.75)
        ax.tick_params(axis="both", length=0, pad=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B9C1C9")
        ax.set_ylim(len(order) - 0.40, -0.60)

    ax_count.barh(y, total, height=bar_height, color=colors["total"], zorder=3)
    ax_count.set_xscale("log")
    ax_count.set_xlim(1, 600_000)
    ax_count.set_xticks([1, 100, 10_000, 100_000])
    ax_count.set_xticklabels(["1", "100", "10,000", "100,000"])
    ax_count.xaxis.set_minor_locator(ticker.NullLocator())
    ax_count.set_yticks(y, labels)
    ax_count.set_xlabel("전체 윈도우 수 (로그 축)", labelpad=12)
    ax_count.set_title("(a) 데이터셋별 전체 개수", loc="left", pad=18, fontweight="bold")
    for row_index, count in enumerate(total):
        ax_count.annotate(
            f"{count:,}",
            (count, row_index),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#26323C",
        )

    normal_pct = normal / total * 100
    defect_pct = defect / total * 100
    ax_class.barh(
        y, normal_pct, height=bar_height, color=colors["normal"],
        edgecolor="white", linewidth=1.0, zorder=3,
    )
    ax_class.barh(
        y, defect_pct, left=normal_pct, height=bar_height,
        color=colors["defect"], edgecolor="white", linewidth=1.0,
        hatch="//", zorder=3,
    )
    ax_class.set_xlim(0, 100)
    ax_class.set_xticks(np.arange(0, 101, 20))
    ax_class.set_xlabel("정상·불량 윈도우 구성비 (%)", labelpad=12)
    ax_class.set_title("(b) 클래스별 구성", loc="left", pad=18, fontweight="bold")
    ax_class.tick_params(axis="y", labelleft=False)

    for row_index in y:
        left = 0.0
        for counts, proportions in ((normal, normal_pct), (defect, defect_pct)):
            count = counts[row_index]
            pct = proportions[row_index]
            if count:
                percentage = "100%" if pct == 100 else f"{pct:.1f}%"
                ax_class.text(
                    left + pct / 2,
                    row_index,
                    f"{count:,}\n({percentage})",
                    ha="center", va="center", color="white",
                    fontsize=9.5 if pct < 20 else 11,
                    fontweight="bold", linespacing=1.6,
                )
            left += pct

    fig.legend(
        handles=[
            Patch(facecolor=colors["normal"], label="정상"),
            Patch(facecolor=colors["defect"], edgecolor="white", hatch="//", label="불량"),
        ],
        loc="upper right", bbox_to_anchor=(0.983, 0.976),
        ncol=2, frameon=False, handlelength=1.8, handleheight=1.0, columnspacing=1.5,
    )
    fig.text(
        0.145, 0.945, "실험 데이터셋의 윈도우 구성",
        fontsize=15, fontweight="bold", color="#24313B", va="center",
    )
    fig.text(
        0.145, 0.084,
        "막대 내부: 윈도우 수 (해당 데이터셋 내 비율). 윈도우 길이 30, 이동 간격 1.",
        fontsize=10, color="#4B5560",
    )
    fig.text(
        0.145, 0.035,
        "정상 학습 후보 135,730개 중 주요 분류 실험에 사용한 정상 윈도우는 5,000개임.",
        fontsize=10, color="#4B5560",
    )

    output = ROOT / "pictures"
    output.mkdir(exist_ok=True)
    stem = output / "dataset_window_composition_ko"
    for extension in ("png", "svg", "pdf"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
