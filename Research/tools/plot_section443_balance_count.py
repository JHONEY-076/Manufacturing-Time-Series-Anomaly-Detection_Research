"""Publication figures and tables from matched section 4.4.3 results."""
from pathlib import Path
import json
import os
import hashlib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

OUT = ROOT / "output/section_4_4_3"
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("auprc_ap", "AUPRC (AP)")]
STYLES = {"Original only": ("#69737D", "D", "--"), "Masking Diffusion": ("#287AA8", "s", "-"), "FMD": ("#B63C48", "o", "-")}


def save(fig, stem):
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=320, bbox_inches="tight", facecolor="white", pad_inches=0.14)
    plt.close(fig)


def main():
    source = OUT / "balance_count_results.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    counts = data["count_experiment"]["generated_counts"]
    normals = data["balance_experiment"]["normal_counts"]
    crows = data["count_experiment"]["results"]
    brows = data["balance_experiment"]["results"]
    baseline = next(r for r in crows if r["method"] == "Original only")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
                         "legend.fontsize": 10, "svg.fonttype": "none", "axes.unicode_minus": False})
    def format_ax(ax, key, label, panel):
        ax.set_title(f"({chr(97 + panel)}) {label}", loc="left", fontweight="bold", pad=10)
        ax.set_ylim(0.30, 1.015)
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.set_ylabel("Score")
        ax.grid(axis="y", color="#DFE4E8", linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#ADB5BD")
        ax.tick_params(labelsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    handles = {}
    for panel, (ax, (key, label)) in enumerate(zip(axes.flat, METRICS)):
        format_ax(ax, key, label, panel)
        handles["Original only"] = ax.axhline(baseline[key], color=STYLES["Original only"][0], ls="--", lw=1.6)
        for method in ["Masking Diffusion", "FMD"]:
            rows = [baseline] + sorted([r for r in crows if r["method"] == method], key=lambda r: r["generated_count"])
            assert [r["generated_count"] for r in rows] == counts
            color, marker, linestyle = STYLES[method]
            handles[method], = ax.plot(counts, [r[key] for r in rows], color=color, marker=marker,
                                       ls=linestyle, lw=2, ms=5.5, mfc="white", mew=1.4)
        ax.set_xlim(-25, 1025)
        ax.set_xticks(counts, [f"{n:,}" for n in counts], rotation=45, ha="right")
        ax.set_xlabel("Added generated anomaly windows")
    fig.suptitle("Effect of augmentation count", y=.99, fontsize=17, fontweight="bold")
    fig.legend([handles[m] for m in STYLES], ["Original only (baseline)", "Masking Diffusion", "FMD"],
               loc="upper center", bbox_to_anchor=(.5, .948), ncol=3, frameon=False)
    fig.subplots_adjust(left=.075, right=.98, top=.855, bottom=.18, hspace=.48, wspace=.22)
    fig.text(.075, .045, "Fixed RF training data: 5,000 real normal + 247 real anomaly windows; nested generated subsets.", fontsize=9, color="#4E5964")
    fig.text(.075, .020, "One saved candidate pool and RF seed; validation-selected thresholds. At 1,000, MD and FMD use identical data.", fontsize=9, color="#4E5964")
    save(fig, "augmentation_count_matched")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    handles = {}
    for panel, (ax, (key, label)) in enumerate(zip(axes.flat, METRICS)):
        format_ax(ax, key, label, panel)
        for method, (color, marker, linestyle) in STYLES.items():
            rows = sorted([r for r in brows if r["method"] == method], key=lambda r: r["normal_count"])
            assert [r["normal_count"] for r in rows] == normals
            handles[method], = ax.plot(normals, [r[key] for r in rows], color=color, marker=marker,
                                       ls=linestyle, lw=2 if method != "Original only" else 1.6,
                                       ms=5.5, mfc="white", mew=1.4)
        ax.set_xscale("log", base=2)
        ax.set_xlim(870, 23000)
        ax.set_xticks(normals, [f"{n:,}\n({n / 997:.2f}:1)" for n in normals])
        ax.set_xlabel("Real normal training windows")
    fig.suptitle("Effect of RF training class composition", y=.99, fontsize=17, fontweight="bold")
    fig.legend([handles[m] for m in STYLES], ["Original only (0 generated)", "Masking Diffusion (+750)", "FMD (+750)"],
               loc="upper center", bbox_to_anchor=(.5, .948), ncol=3, frameon=False)
    fig.subplots_adjust(left=.075, right=.98, top=.855, bottom=.18, hspace=.46, wspace=.22)
    fig.text(.075, .055, "247 real anomalies throughout; MD/FMD add a fixed 750-window subset. Normal subsets are nested.", fontsize=9, color="#4E5964")
    fig.text(.075, .029, "Ratios in parentheses apply to MD/FMD. RF class weights: balanced_subsample. Generator/filter references stay fixed.", fontsize=8.8, color="#4E5964")
    save(fig, "class_balance_matched")

    fmd = [baseline] + sorted([r for r in crows if r["method"] == "FMD"], key=lambda r: r["generated_count"])
    lines = ["**표 C. 정상 5,000개를 고정했을 때 FMD 증강량에 따른 RandomForest 성능.**", "",
             "| 생성 수 | 전체 불량 수 | 정상:불량 비율 | Precision | Recall | F1 | AUPRC(AP) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fmd:
        lines.append(f"| {r['generated_count']:,} | {r['total_anomaly_count']:,} | {r['normal_to_anomaly_ratio']:.2f}:1 | "
                     + " | ".join(f"{r[k]:.4f}" for k, _ in METRICS) + " |")
    lines += ["", "주: 전체 불량 수는 실제 247개와 생성 수의 합이다. 0개는 원본만 학습한 기준선이며, 1,000개는 후보 전체를 사용하는 조건으로 Masking Diffusion과 동일하다."]
    lines += ["", "**표 D. 생성 불량 750개를 고정했을 때 정상 학습 데이터 수와 클래스 비율에 따른 성능.**", "",
              "| 정상 수 | 증강 후 정상:불량 비율 | 원본만 F1 | MD F1 | FMD F1 | FMD Recall | FMD AUPRC(AP) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for n in normals:
        group = {r["method"]: r for r in brows if r["normal_count"] == n}
        r = group["FMD"]
        lines.append(f"| {n:,} | {n / 997:.2f}:1 | {group['Original only']['f1']:.4f} | "
                     f"{group['Masking Diffusion']['f1']:.4f} | {r['f1']:.4f} | {r['recall']:.4f} | {r['auprc_ap']:.4f} |")
    lines += ["", "주: MD는 Masking Diffusion이다. 표의 비율은 실제 불량 247개와 생성 불량 750개를 합한 997개를 분모로 하며 MD와 FMD에만 적용된다. 원본만 조건은 동일한 정상 수와 실제 불량 247개를 사용한다.",
              "", "두 표 모두 검증 집합에서 임계값을 선택하고 정상 2,785개·불량 295개의 공통 최종 평가 집합에서 계산한 단일 RF 실행 결과이다. AUPRC는 average precision으로 계산하였다."]
    (OUT / "balance_count_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "count_figure_run_ids": [r["run_id"] for r in crows], "balance_figure_run_ids": [r["run_id"] for r in brows],
                "plotted_metrics": [k for k, _ in METRICS], "class_balance_x_scale": "log2 of normal training count",
                "y_limits": [0.30, 1.015], "uncertainty": "No error bars: one RF seed and saved generation pool.",
                "interpolation": "Straight segments connect measured conditions only.",
                "outputs": [f"{stem}.{ext}" for stem in ["augmentation_count_matched", "class_balance_matched"] for ext in ["png", "svg"]]}
    (OUT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
