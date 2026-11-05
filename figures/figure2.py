from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure2_data as base


OUT = Path(__file__).resolve().parent
MODELS = base.MODELS
COLORS = base.COLORS

POINT_SIZE = 4.5
CI_WIDTH = 1.0
CAP_SIZE = 2.3
CAP_THICKNESS = 0.85
GRID_COLOR = "#D9D9D9"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.6,
            "axes.titlesize": 8.2,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def style_main_axis(ax: plt.Axes, ymax: float, ylabel: bool) -> None:
    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(3), MODELS)
    ax.set_ylabel("Accuracy (%)" if ylabel else "")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.34, alpha=0.55)
    ax.set_axisbelow(True)


def draw_single_metric(ax: plt.Axes, frame: pd.DataFrame, ymax: float, *, ylabel: bool) -> None:
    pct = base.percent_frame(frame)
    for x, row in pct.iterrows():
        model = str(row["Model"])
        value = float(row["Value"])
        low = float(row["Low"])
        high = float(row["High"])
        ax.errorbar(
            x,
            value,
            yerr=np.array([[value - low], [high - value]]),
            fmt="o",
            markersize=POINT_SIZE,
            color=COLORS[model],
            markerfacecolor=COLORS[model],
            markeredgewidth=0.65,
            elinewidth=CI_WIDTH,
            capsize=CAP_SIZE,
            capthick=CAP_THICKNESS,
            zorder=3,
        )
        ax.text(x, high + 0.027 * ymax, f"{value:.1f}", ha="center", va="bottom", fontsize=5.8)
    style_main_axis(ax, ymax, ylabel)


def draw_task3(ax: plt.Axes, verdict: pd.DataFrame, reason: pd.DataFrame) -> None:
    v = base.percent_frame(verdict)
    r = base.percent_frame(reason)
    for x, model in enumerate(MODELS):
        vr = v.loc[v["Model"] == model].iloc[0]
        rr = r.loc[r["Model"] == model].iloc[0]
        color = COLORS[model]
        xv, xr = x - 0.105, x + 0.105
        ax.errorbar(
            xv,
            vr["Value"],
            yerr=np.array([[vr["Value"] - vr["Low"]], [vr["High"] - vr["Value"]]]),
            fmt="o",
            markersize=POINT_SIZE,
            color=color,
            markerfacecolor=color,
            markeredgewidth=0.65,
            elinewidth=CI_WIDTH,
            capsize=CAP_SIZE,
            capthick=CAP_THICKNESS,
            zorder=3,
        )
        ax.errorbar(
            xr,
            rr["Value"],
            yerr=np.array([[rr["Value"] - rr["Low"]], [rr["High"] - rr["Value"]]]),
            fmt="D",
            markersize=4.6,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.0,
            elinewidth=CI_WIDTH,
            capsize=CAP_SIZE,
            capthick=CAP_THICKNESS,
            zorder=3,
        )
        ax.text(
            xv,
            float(vr["High"]) + 2.2,
            f"{vr['Value']:.1f}",
            ha="center",
            va="bottom",
            color=color,
            fontsize=5.7,
        )
        ax.text(
            xr,
            float(rr["Low"]) - 2.2,
            f"{rr['Value']:.1f}",
            ha="center",
            va="top",
            color=color,
            fontsize=5.7,
        )
    style_main_axis(ax, 100, ylabel=False)

    outcome_legend = [
        Line2D([0], [0], marker="o", color="#444444", markerfacecolor="#444444", lw=0, markersize=4.2, label="Verdict"),
        Line2D([0], [0], marker="D", color="#444444", markerfacecolor="white", lw=0, markersize=4.3, label="Key reason"),
    ]
    ax.legend(
        handles=outcome_legend,
        loc="lower left",
        ncol=2,
        handletextpad=0.32,
        columnspacing=0.72,
        borderaxespad=0.2,
        fontsize=5.65,
    )


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.5f}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def draw_contrast(ax: plt.Axes, row: pd.Series, name: str, xlim: tuple[float, float]) -> None:
    estimate = float(row["Difference_pp"])
    low = float(row["CI95_Lower_pp"])
    high = float(row["CI95_Upper_pp"])
    p_value = float(row["Raw_P"])

    ax.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.65, zorder=1)
    ax.errorbar(
        estimate,
        0.38,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="o",
        markersize=POINT_SIZE,
        markerfacecolor="white",
        markeredgecolor="#303030",
        markeredgewidth=1.0,
        color="#303030",
        elinewidth=CI_WIDTH,
        capsize=CAP_SIZE,
        capthick=CAP_THICKNESS,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.32, alpha=0.58)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Difference (percentage points)", labelpad=1.5, fontsize=5.8)
    ax.text(0.0, 1.26, "Exploratory contrast", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color="#6A6A6A")
    ax.text(0.0, 1.06, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.75)
    ax.text(1.0, 1.06, f"P = {format_p(p_value)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.65)
    ax.text(
        0.5,
        0.67,
        f"{estimate:+.1f} pp (95% CI {low:.1f}–{high:.1f})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.35,
        color="#444444",
    )


def draw_reliability(ax: plt.Axes, data: pd.DataFrame, metric: str, xlabel: str, xlim: tuple[float, float]) -> None:
    labels = [
        "Task 1 · Joint-success status",
        "Task 2 · Controversy classification",
        "Task 3 · Applicability verdict",
        "Task 3 · Reason-quality rating",
    ]
    y_base = np.arange(4)[::-1].astype(float)
    offsets = {"ChatGPT": 0.17, "DeepSeek": 0.0, "Gemini": -0.17}
    if metric == "Cohen kappa":
        ax.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.65, zorder=1)

    for i, label in enumerate(labels):
        block = data.loc[(data["Outcome"] == label) & (data["Metric"] == metric)]
        for model in MODELS:
            row = block.loc[block["Model"] == model].iloc[0]
            estimate = float(row["Estimate"])
            low = row["CI_Low"]
            high = row["CI_High"]
            interval: dict = {}
            if pd.notna(low) and pd.notna(high):
                interval["xerr"] = np.array([[estimate - float(low)], [float(high) - estimate]])
            ax.errorbar(
                estimate,
                y_base[i] + offsets[model],
                fmt="o",
                markersize=POINT_SIZE,
                color=COLORS[model],
                markerfacecolor=COLORS[model],
                markeredgewidth=0.65,
                elinewidth=CI_WIDTH,
                capsize=CAP_SIZE,
                capthick=CAP_THICKNESS,
                zorder=3,
                **interval,
            )
    ax.set_yticks(y_base, labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.45, 3.45)
    ax.set_xlabel(xlabel, labelpad=3)
    ax.grid(axis="x", color="#DEDEDE", linewidth=0.34, alpha=0.58)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=4)


def build_figure(data: dict[str, pd.DataFrame]) -> plt.Figure:
    # Formal width: 180 mm; height: 145 mm.
    fig = plt.figure(figsize=(7.086614, 5.708661), facecolor="white")

    top = fig.add_gridspec(
        1,
        3,
        left=0.073,
        right=0.985,
        bottom=0.49,
        top=0.90,
        wspace=0.36,
    )
    panels: list[tuple[plt.Axes, plt.Axes]] = []
    for col in range(3):
        inner = top[0, col].subgridspec(2, 1, height_ratios=[2.62, 0.94], hspace=0.36)
        panels.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))

    ax_a, ax_ac = panels[0]
    ax_b, ax_bc = panels[1]
    ax_c, ax_cc = panels[2]

    draw_single_metric(ax_a, data["task1"], 40, ylabel=True)
    draw_single_metric(ax_b, data["task2"], 100, ylabel=False)
    draw_task3(ax_c, data["task3_verdict"], data["task3_reason"])

    titles = ["Evidence completeness", "Evidence consensus", "Evidence applicability"]
    subtitles = [
        "Joint Q1–Q2 success (170 items)",
        "Classification accuracy (36 questions)",
        "Accuracy (60 scenarios)",
    ]
    for ax, title, subtitle in zip([ax_a, ax_b, ax_c], titles, subtitles):
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=6.15)

    contrast = data["contrasts"].set_index("Task")
    draw_contrast(ax_ac, contrast.loc["Task 1"], "PubMed-linked − unlinked", (-5, 23))
    draw_contrast(ax_bc, contrast.loc["Task 2"], "Low − high controversy", (-5, 50))
    draw_contrast(ax_cc, contrast.loc["Task 3"], "Verdict − key-reason accuracy", (-5, 23))

    bottom = fig.add_gridspec(
        1,
        2,
        left=0.215,
        right=0.985,
        bottom=0.095,
        top=0.372,
        wspace=0.18,
    )
    ax_da = fig.add_subplot(bottom[0, 0])
    ax_dk = fig.add_subplot(bottom[0, 1])
    draw_reliability(ax_da, data["reliability"], "Observed agreement", "Observed agreement (%)", (0, 100))
    draw_reliability(ax_dk, data["reliability"], "Cohen kappa", "Unweighted Cohen’s κ", (-0.4, 1.0))
    ax_dk.set_yticklabels([])

    panel_letter_size = 8.5
    for letter, ax in zip(["a", "b", "c"], [ax_a, ax_b, ax_c]):
        box = ax.get_position()
        fig.text(box.x0 - 0.028, box.y1 + 0.043, letter, fontsize=panel_letter_size, fontweight="bold", ha="left", va="top")
    fig.text(0.073, 0.401, "d", fontsize=panel_letter_size, fontweight="bold", ha="left", va="bottom")
    fig.text(0.104, 0.401, "Repeated-response reliability", fontsize=8.2, fontweight="bold", ha="left", va="bottom")

    model_handles = [
        Line2D([0], [0], marker="o", color=COLORS[m], markerfacecolor=COLORS[m], lw=0, markersize=4.6, label=m)
        for m in MODELS
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.25,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2 from analysis-generated CSV summaries.")
    parser.add_argument("--task1-results", type=Path, required=True)
    parser.add_argument("--task2-results", type=Path, required=True)
    parser.add_argument("--task3-results", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["public", "full"],
        default="full",
        help="Public mode exports all available source data without inventing restricted Task 3 results.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    global OUT
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    data, missing = base.load_analysis_outputs(
        args.task1_results,
        args.task2_results,
        args.task3_results,
    )
    base.export_source_data(data, OUT, missing)
    if missing:
        message = "Complete Figure 2 requires restricted Task 3 human reason ratings: " + "; ".join(missing)
        if args.mode == "full":
            raise RuntimeError(message)
        print(message)
        print("Public-data source tables and a machine-readable availability report were created; no incomplete figure was drawn.")
        return
    set_style()
    fig = build_figure(data)
    fig.savefig(OUT / "Figure2.svg", facecolor="white")
    fig.savefig(OUT / "Figure2.pdf", facecolor="white")
    fig.savefig(OUT / "Figure2.png", dpi=600, facecolor="white")
    fig.savefig(OUT / "Figure2.tiff", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
