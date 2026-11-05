from __future__ import annotations

import argparse
from pathlib import Path


OUT = Path(__file__).resolve().parent

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PNG_OUT = OUT / "Supplementary_Figure_1.png"
PDF_OUT = OUT / "Supplementary_Figure_1.pdf"
SVG_OUT = OUT / "Supplementary_Figure_1.svg"

WIDTH_MM = 183
HEIGHT_MM = 154
X_LIM = (-25.0, 45.0)
INK = "#1F1F1F"
MID_GRAY = "#555555"
LIGHT_GRAY = "#D7D7D7"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.9,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_and_validate(task1_results: Path, task2_results: Path, task3_results: Path) -> pd.DataFrame:
    task1 = pd.read_csv(task1_results / "model_comparisons.csv")
    task2 = pd.read_csv(task2_results / "model_comparisons.csv")
    task3 = pd.read_csv(task3_results / "model_comparisons.csv")
    model = pd.concat(
        [
            pd.DataFrame({
                "Task": "Task 1",
                "Direction": task1["Comparison"],
                "Difference_pp": task1["Difference_pp"],
                "CI95_Lower_pp": task1["CI95_Lower_pp"],
                "CI95_Upper_pp": task1["CI95_Upper_pp"],
                "Adjusted_P": task1["Holm_Adjusted_P"],
            }),
            pd.DataFrame({
                "Task": "Task 2",
                "Direction": task2["Comparison"],
                "Difference_pp": task2["Difference_pp"],
                "CI95_Lower_pp": task2["CI95_Lower_pp"],
                "CI95_Upper_pp": task2["CI95_Upper_pp"],
                "Adjusted_P": task2["Holm_Adjusted_P"],
            }),
            pd.DataFrame({
                "Task": "Task 3",
                "Direction": task3["Model_A"].astype(str) + " minus " + task3["Model_B"].astype(str),
                "Difference_pp": task3["Difference_A_minus_B_pp"],
                "CI95_Lower_pp": task3["CI_Low_pp"],
                "CI95_Upper_pp": task3["CI_High_pp"],
                "Adjusted_P": task3["Holm_Adjusted_P_Value"],
            }),
        ],
        ignore_index=True,
    )
    if len(model) != 9:
        raise ValueError("Expected exactly nine frozen between-model comparisons")
    required = ["Task", "Direction", "Difference_pp", "CI95_Lower_pp", "CI95_Upper_pp", "Adjusted_P"]
    if model[required].isna().any().any():
        raise ValueError("Missing frozen between-model comparison value")
    return model


def prepare_rows(model: pd.DataFrame) -> list[dict]:
    order = {
        "Task 1": ["ChatGPT minus DeepSeek", "ChatGPT minus Gemini", "DeepSeek minus Gemini"],
        "Task 2": ["DeepSeek minus Gemini", "DeepSeek minus ChatGPT", "Gemini minus ChatGPT"],
        "Task 3": ["ChatGPT minus DeepSeek", "ChatGPT minus Gemini", "DeepSeek minus Gemini"],
    }
    positions = {
        "Task 1": [10.35, 9.35, 8.35],
        "Task 2": [6.20, 5.20, 4.20],
        "Task 3": [2.05, 1.05, 0.05],
    }
    rows: list[dict] = []
    for task in ["Task 1", "Task 2", "Task 3"]:
        subset = model.loc[model["Task"] == task]
        for direction, y in zip(order[task], positions[task]):
            row = subset.loc[subset["Direction"] == direction]
            if len(row) != 1:
                raise ValueError(f"Expected one frozen row for {task}: {direction}")
            rows.append({**row.iloc[0].to_dict(), "y": y})
    return rows


def signed(value: float) -> str:
    value = float(value)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:+.2f}"


def effect_text(row: dict) -> str:
    return (
        f"{signed(row['Difference_pp'])} "
        f"({signed(row['CI95_Lower_pp'])} to {signed(row['CI95_Upper_pp'])})"
    )


def p_text(value: float) -> str:
    p = float(value)
    if p < 0.001:
        return f"{p:.6f}"
    return f"{p:.6f}".rstrip("0").rstrip(".")


def build_figure(task1_results: Path, task2_results: Path, task3_results: Path) -> None:
    configure_style()
    rows = prepare_rows(load_and_validate(task1_results, task2_results, task3_results))

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.55, 2.85, 2.50],
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.115,
        wspace=0.04,
    )
    label_ax = fig.add_subplot(grid[0, 0])
    forest_ax = fig.add_subplot(grid[0, 1])
    stat_ax = fig.add_subplot(grid[0, 2])
    ylim = (-0.45, 12.20)

    for ax in (label_ax, stat_ax):
        ax.set_axis_off()
        ax.set_ylim(*ylim)

    forest_ax.set_xlim(*X_LIM)
    forest_ax.set_ylim(*ylim)
    forest_ax.set_yticks([])
    forest_ax.set_xticks([-20, 0, 20, 40])
    forest_ax.set_xlabel("Difference in accuracy (percentage points)", labelpad=5)
    forest_ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.45, alpha=0.8)
    forest_ax.set_axisbelow(True)
    forest_ax.axvline(0, color=MID_GRAY, linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
    forest_ax.spines["left"].set_visible(False)
    forest_ax.spines["bottom"].set_color(MID_GRAY)

    stat_ax.text(
        0.00,
        1.015,
        "Difference (95% CI), pp",
        transform=stat_ax.transAxes,
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    stat_ax.text(
        1.00,
        1.015,
        "Holm-adjusted P",
        transform=stat_ax.transAxes,
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        ha="right",
        va="bottom",
    )

    groups = [
        ("Task 1", 11.65, 11.23, "Evidence completeness", "Joint Success"),
        ("Task 2", 7.50, 7.08, "Evidence consensus", "Overall controversy classification accuracy"),
        ("Task 3", 3.35, 2.93, "Evidence applicability", "Applicability verdict accuracy"),
    ]
    for task, y_head, y_outcome, title, outcome in groups:
        label_ax.text(
            0.0,
            y_head,
            f"{task}  {title}",
            color=INK,
            fontsize=7.6,
            fontweight="bold",
            va="center",
        )
        label_ax.text(0.0, y_outcome, outcome, color=MID_GRAY, fontsize=6.6, va="center")

    for y in [7.90, 3.75]:
        for ax in (label_ax, forest_ax, stat_ax):
            ax.axhline(y, color=LIGHT_GRAY, linewidth=0.55, zorder=0)

    for row in rows:
        estimate = float(row["Difference_pp"])
        low = float(row["CI95_Lower_pp"])
        high = float(row["CI95_Upper_pp"])
        y = float(row["y"])
        forest_ax.errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="o",
            markersize=5.0,
            markerfacecolor=INK,
            markeredgecolor=INK,
            markeredgewidth=0.6,
            ecolor=INK,
            elinewidth=1.25,
            capsize=2.6,
            capthick=1.0,
            zorder=3,
        )
        label_ax.text(
            0.0,
            y,
            str(row["Direction"]).replace(" minus ", " - "),
            fontsize=7.1,
            color=INK,
            va="center",
        )
        stat_ax.text(0.0, y, effect_text(row), fontsize=6.8, color=INK, va="center")
        stat_ax.text(1.0, y, p_text(row["Adjusted_P"]), fontsize=6.8, color=INK, ha="right", va="center")

    fig.savefig(PNG_OUT, dpi=600)
    fig.savefig(PDF_OUT)
    fig.savefig(SVG_OUT)
    plt.close(fig)
    print(f"Created monochrome forest plot with {len(rows)} frozen comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Supplementary Figure 1 from analysis-generated CSV summaries.")
    parser.add_argument("--task1-results", type=Path, required=True)
    parser.add_argument("--task2-results", type=Path, required=True)
    parser.add_argument("--task3-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    globals()["OUT"] = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    globals()["PNG_OUT"] = args.output_dir / "Supplementary_Figure_1.png"
    globals()["PDF_OUT"] = args.output_dir / "Supplementary_Figure_1.pdf"
    globals()["SVG_OUT"] = args.output_dir / "Supplementary_Figure_1.svg"
    build_figure(args.task1_results, args.task2_results, args.task3_results)
