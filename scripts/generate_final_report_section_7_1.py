from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRWISE_SUMMARY_PATH = REPO_ROOT / "outputs" / "pairwise_deepseek_feature_all200" / "summary.csv"
SCENARIO_OVERVIEW_PATH = REPO_ROOT / "outputs" / "pairwise_deepseek_feature_all200_scenario" / "scenario_overview.csv"
SCENARIO_SUMMARY_PATH = REPO_ROOT / "outputs" / "pairwise_deepseek_feature_all200_scenario" / "summary.csv"
OUTPUT_DIR = REPO_ROOT / "final_report_figures"

FEATURE_COLOR = "#4C78A8"
FUSION_COLOR = "#F58518"
TIE_COLOR = "#BAB0AC"
GRID_COLOR = "#D9D9D9"

PAIR_LABELS = {
    "feature_vs_early": "Feature vs Early fusion",
    "feature_vs_late": "Feature vs Late fusion",
}
SCENARIO_ORDER = ["emotion_dominant", "hybrid_emotion_task", "task_dominant"]
SCENARIO_LABELS = {
    "emotion_dominant": "Emotion-dominant",
    "hybrid_emotion_task": "Hybrid emotion-task",
    "task_dominant": "Task-dominant",
}
DIMENSION_LABELS = {
    "D1": "D1 Attunement",
    "D2": "D2 Strategy Fit",
    "D3": "D3 Usefulness",
    "D4": "D4 Clarity",
    "D5": "D5 Safety",
    "D6": "D6 Signal Use",
}


def apply_academic_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
        }
    )


def add_segment_labels(ax: plt.Axes, containers: Iterable, threshold: float = 8.0) -> None:
    for container in containers:
        labels = []
        for patch in container:
            value = patch.get_height()
            labels.append(f"{value:.1f}%" if value >= threshold else "")
        ax.bar_label(container, labels=labels, label_type="center", color="white", fontsize=9, fontweight="bold")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise_summary = pd.read_csv(PAIRWISE_SUMMARY_PATH, encoding="utf-8-sig")
    scenario_overview = pd.read_csv(SCENARIO_OVERVIEW_PATH, encoding="utf-8-sig")
    scenario_summary = pd.read_csv(SCENARIO_SUMMARY_PATH, encoding="utf-8-sig")
    return pairwise_summary, scenario_overview, scenario_summary


def build_overall_summary(pairwise_summary: pd.DataFrame) -> pd.DataFrame:
    overall = pairwise_summary[pairwise_summary["dimension"] == "OVERALL_D1_D5"].copy()
    overall["Comparison"] = overall["pair"].map(PAIR_LABELS)
    overall["Feature win %"] = overall["a_win_rate"].round(1)
    overall["Fusion win %"] = overall["b_win_rate"].round(1)
    overall["Tie %"] = overall["tie_rate"].round(1)
    overall["Main interpretation"] = overall["pair"].map(
        {
            "feature_vs_early": "Feature system shows a clearer overall advantage",
            "feature_vs_late": "Systems are close overall",
        }
    )
    return overall[["Comparison", "Feature win %", "Fusion win %", "Tie %", "Main interpretation"]]


def plot_overall_pairwise_outcome(overall_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=False)
    x = np.arange(len(overall_table))
    feature = overall_table["Feature win %"].to_numpy()
    fusion = overall_table["Fusion win %"].to_numpy()
    tie = overall_table["Tie %"].to_numpy()

    bars_feature = ax.bar(x, feature, width=0.62, color=FEATURE_COLOR, label="Feature win")
    bars_fusion = ax.bar(x, fusion, width=0.62, bottom=feature, color=FUSION_COLOR, label="Fusion win")
    bars_tie = ax.bar(x, tie, width=0.62, bottom=feature + fusion, color=TIE_COLOR, label="Tie")

    add_segment_labels(ax, [bars_feature, bars_fusion, bars_tie], threshold=8.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_xticks(x, overall_table["Comparison"])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Overall Pairwise Outcome", fontsize=14, fontweight="bold", y=0.965)
    fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.925), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, output_path)


def build_dimension_summary(pairwise_summary: pd.DataFrame) -> pd.DataFrame:
    quality = pairwise_summary[pairwise_summary["dimension"].isin(DIMENSION_LABELS)].copy()
    quality["delta"] = (quality["avg_score_a"] - quality["avg_score_b"]).round(3)

    early = quality[quality["pair"] == "feature_vs_early"].set_index("dimension")["delta"]
    late = quality[quality["pair"] == "feature_vs_late"].set_index("dimension")["delta"]

    interpretations = {
        "D1": "Fusion advantage in emotional attunement.",
        "D2": "Fusion advantage in strategy fit.",
        "D3": "Feature advantage in task usefulness.",
        "D4": "Feature advantage in clarity and cognitive load fit.",
        "D5": "Nearly identical safety and risk calibration.",
    }

    rows = []
    for dimension in ["D1", "D2", "D3", "D4", "D5"]:
        rows.append(
            {
                "Dimension": DIMENSION_LABELS[dimension],
                "Feature vs Early delta": round(float(early.loc[dimension]), 3),
                "Feature vs Late delta": round(float(late.loc[dimension]), 3),
                "Interpretation": interpretations[dimension],
            }
        )
    return pd.DataFrame(rows)


def plot_dimension_average_scores(pairwise_summary: pd.DataFrame, output_path: Path) -> None:
    quality = pairwise_summary[pairwise_summary["dimension"].isin(DIMENSION_LABELS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), sharey=True, constrained_layout=False)
    width = 0.34
    legend_handles = None
    legend_labels = None

    for ax, pair_name, panel_title in zip(
        axes,
        ["feature_vs_early", "feature_vs_late"],
        ["Panel A: Feature vs Early Fusion", "Panel B: Feature vs Late Fusion"],
    ):
        subset = (
            quality[quality["pair"] == pair_name]
            .set_index("dimension")
            .loc[["D1", "D2", "D3", "D4", "D5", "D6"]]
            .reset_index()
        )
        x = np.arange(len(subset))
        feature_scores = subset["avg_score_a"].to_numpy()
        fusion_scores = subset["avg_score_b"].to_numpy()

        bars_feature = ax.bar(x - width / 2, feature_scores, width=width, color=FEATURE_COLOR, label="Feature system")
        bars_fusion = ax.bar(x + width / 2, fusion_scores, width=width, color=FUSION_COLOR, label="Fusion system")
        ax.bar_label(bars_feature, labels=[f"{v:.3f}" for v in feature_scores], padding=3, fontsize=9)
        ax.bar_label(bars_fusion, labels=[f"{v:.3f}" for v in fusion_scores], padding=3, fontsize=9)

        ax.set_xticks(x, [DIMENSION_LABELS[item] for item in subset["dimension"]])
        ax.set_title(panel_title, fontweight="bold")
        ax.set_ylim(0, 5.35)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Average rubric score")
    fig.suptitle("Dimension-wise Average Scores", fontweight="bold", fontsize=14, y=0.97)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            ncols=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, output_path)


def build_scenario_summary(scenario_overview: pd.DataFrame) -> pd.DataFrame:
    def format_pattern(row: pd.Series) -> str:
        return (
            f"Feature {row['unweighted_a_win_rate']:.1f}%, "
            f"Fusion {row['unweighted_b_win_rate']:.1f}%, "
            f"Tie {row['unweighted_tie_rate']:.1f}%"
        )

    scenario_map = {
        "emotion_dominant": "mixed or close outcome",
        "hybrid_emotion_task": "feature system is stronger, especially against early fusion",
        "task_dominant": "fusion system is more competitive and often stronger by primary win rate",
    }

    rows = []
    for scenario in SCENARIO_ORDER:
        early_row = scenario_overview[
            (scenario_overview["pair"] == "feature_vs_early") & (scenario_overview["scope"] == scenario)
        ].iloc[0]
        late_row = scenario_overview[
            (scenario_overview["pair"] == "feature_vs_late") & (scenario_overview["scope"] == scenario)
        ].iloc[0]
        rows.append(
            {
                "Scenario category": SCENARIO_LABELS[scenario],
                "Feature vs Early pattern": format_pattern(early_row),
                "Feature vs Late pattern": format_pattern(late_row),
                "Main interpretation": scenario_map[scenario],
            }
        )
    return pd.DataFrame(rows)


def plot_scenario_wise_outcome(scenario_overview: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, constrained_layout=False)

    for ax, pair_name, panel_title in zip(
        axes,
        ["feature_vs_early", "feature_vs_late"],
        ["Panel A: Feature vs Early fusion", "Panel B: Feature vs Late fusion"],
    ):
        subset = (
            scenario_overview[scenario_overview["pair"] == pair_name]
            .set_index("scope")
            .loc[SCENARIO_ORDER]
            .reset_index()
        )
        x = np.arange(len(subset))
        feature = subset["unweighted_a_win_rate"].to_numpy()
        fusion = subset["unweighted_b_win_rate"].to_numpy()
        tie = subset["unweighted_tie_rate"].to_numpy()

        bars_feature = ax.bar(x, feature, width=0.62, color=FEATURE_COLOR, label="Feature win")
        bars_fusion = ax.bar(x, fusion, width=0.62, bottom=feature, color=FUSION_COLOR, label="Fusion win")
        bars_tie = ax.bar(x, tie, width=0.62, bottom=feature + fusion, color=TIE_COLOR, label="Tie")

        add_segment_labels(ax, [bars_feature, bars_fusion, bars_tie], threshold=9.0)
        ax.set_title(panel_title)
        ax.set_xticks(x, [SCENARIO_LABELS[item] for item in subset["scope"]])
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Win rate (%)")
    fig.suptitle("Scenario-wise Pairwise Outcome", fontsize=14, y=0.97, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 0.03), frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    save_figure(fig, output_path)


def build_d6_summary(pairwise_summary: pd.DataFrame) -> pd.DataFrame:
    d6 = pairwise_summary[pairwise_summary["dimension"] == "D6"].copy()
    d6["Comparison"] = d6["pair"].map(PAIR_LABELS)
    d6["Feature win %"] = d6["a_win_rate"].round(1)
    d6["Fusion win %"] = d6["b_win_rate"].round(1)
    d6["Tie %"] = d6["tie_rate"].round(1)
    d6["Interpretation"] = [
        "Fusion systems make substantially stronger use of the emotion signal; D6 should be reported separately from D1-D5 quality outcomes."
        for _ in range(len(d6))
    ]
    return d6[["Comparison", "Feature win %", "Fusion win %", "Tie %", "Interpretation"]]


def plot_emotion_signal_utilisation(d6_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    x = np.arange(len(d6_table))
    feature = d6_table["Feature win %"].to_numpy()
    fusion = d6_table["Fusion win %"].to_numpy()
    tie = d6_table["Tie %"].to_numpy()

    bars_feature = ax.bar(x, feature, width=0.62, color=FEATURE_COLOR, label="Feature win")
    bars_fusion = ax.bar(x, fusion, width=0.62, bottom=feature, color=FUSION_COLOR, label="Fusion win")
    bars_tie = ax.bar(x, tie, width=0.62, bottom=feature + fusion, color=TIE_COLOR, label="Tie")

    add_segment_labels(ax, [bars_feature, bars_fusion, bars_tie], threshold=6.0)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, d6_table["Comparison"])
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Emotion Signal Utilisation", pad=10, fontweight="bold")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.20), frameon=False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.text(
        0.02,
        -0.24,
        "Note: D6 is a mechanism dimension and is reported separately from the D1-D5 overall quality result.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, output_path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    apply_academic_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_summary, scenario_overview, scenario_summary = load_inputs()
    _ = scenario_summary  # Loaded to keep the script aligned with the available scenario summary outputs.

    overall_table = build_overall_summary(pairwise_summary)
    dimension_table = build_dimension_summary(pairwise_summary)
    scenario_table = build_scenario_summary(scenario_overview)
    d6_table = build_d6_summary(pairwise_summary)

    saved_files = []

    overall_fig = OUTPUT_DIR / "Figure_7_1_overall_pairwise_outcome.png"
    overall_csv = OUTPUT_DIR / "table_7_1_overall_summary.csv"
    plot_overall_pairwise_outcome(overall_table, overall_fig)
    write_csv(overall_table, overall_csv)
    saved_files.extend([overall_fig, overall_csv])

    dimension_fig = OUTPUT_DIR / "Figure_7_2_dimension_average_scores.png"
    dimension_csv = OUTPUT_DIR / "table_7_1_dimension_summary.csv"
    plot_dimension_average_scores(pairwise_summary, dimension_fig)
    write_csv(dimension_table, dimension_csv)
    saved_files.extend([dimension_fig, dimension_csv])

    scenario_fig = OUTPUT_DIR / "Figure_7_3_scenario_wise_outcome.png"
    scenario_csv = OUTPUT_DIR / "table_7_2_scenario_summary.csv"
    plot_scenario_wise_outcome(scenario_overview, scenario_fig)
    write_csv(scenario_table, scenario_csv)
    saved_files.extend([scenario_fig, scenario_csv])

    d6_fig = OUTPUT_DIR / "Figure_7_4_emotion_signal_utilisation.png"
    d6_csv = OUTPUT_DIR / "table_7_3_d6_summary.csv"
    plot_emotion_signal_utilisation(d6_table, d6_fig)
    write_csv(d6_table, d6_csv)
    saved_files.extend([d6_fig, d6_csv])

    print("Saved final report assets:")
    for path in saved_files:
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
