from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from generate_final_report_section_7_1 import (
    DIMENSION_LABELS,
    FEATURE_COLOR,
    FUSION_COLOR,
    OUTPUT_DIR,
    PAIR_LABELS,
    TIE_COLOR,
    apply_academic_style,
    build_overall_summary,
    load_inputs,
)


DIMENSION_TICK_LABELS = {
    "D1": "D1\nAttunement",
    "D2": "D2\nStrategy Fit",
    "D3": "D3\nUsefulness",
    "D4": "D4\nClarity",
    "D5": "D5\nSafety",
    "D6": "D6\nSignal Use",
}

CAPTION = (
    "Blind pairwise evaluation results for the feature-based system and the two fusion variants. "
    "Panel A shows the overall pairwise win rates. Panel B shows the dimension-wise average rubric "
    "scores for Feature vs Early Fusion and Feature vs Late Fusion."
)


def save_figure_variants(fig: plt.Figure, *paths: Path) -> None:
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_stacked_bar_labels(ax: plt.Axes, feature: np.ndarray, fusion: np.ndarray, tie: np.ndarray) -> None:
    x = np.arange(len(feature))
    for idx, (f_val, fu_val, t_val) in enumerate(zip(feature, fusion, tie)):
        segments = [
            (f_val / 2, f_val, f"{f_val:.1f}%"),
            (f_val + fu_val / 2, fu_val, f"{fu_val:.1f}%"),
            (f_val + fu_val + t_val / 2, t_val, f"{t_val:.1f}%"),
        ]
        for y_pos, value, label in segments:
            if value >= 8.0:
                ax.text(
                    x[idx],
                    y_pos,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9.5,
                    fontweight="bold",
                )


def plot_combined_panel_figure(output_png: Path, output_pdf: Path) -> None:
    apply_academic_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_summary, _, _ = load_inputs()
    overall_table = build_overall_summary(pairwise_summary)
    quality = pairwise_summary[pairwise_summary["dimension"].isin(DIMENSION_LABELS)].copy()

    fig = plt.figure(figsize=(10.8, 11.0), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.45], hspace=0.42, wspace=0.18)

    ax_top = fig.add_subplot(gs[0, :])
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1], sharey=ax_b1)

    x = np.arange(len(overall_table))
    feature = overall_table["Feature win %"].to_numpy()
    fusion = overall_table["Fusion win %"].to_numpy()
    tie = overall_table["Tie %"].to_numpy()

    ax_top.bar(x, feature, width=0.58, color=FEATURE_COLOR)
    ax_top.bar(x, fusion, width=0.58, bottom=feature, color=FUSION_COLOR)
    ax_top.bar(x, tie, width=0.58, bottom=feature + fusion, color=TIE_COLOR)
    add_stacked_bar_labels(ax_top, feature, fusion, tie)

    ax_top.set_title("(A) Overall pairwise outcome", loc="left", pad=10, fontweight="bold")
    ax_top.set_ylabel("Win rate (%)")
    ax_top.set_xticks(x, overall_table["Comparison"])
    ax_top.set_ylim(0, 100)
    ax_top.grid(axis="y")
    ax_top.grid(axis="x", visible=False)

    bottom_specs = [
        (ax_b1, "feature_vs_early", "(B1) Feature vs Early Fusion"),
        (ax_b2, "feature_vs_late", "(B2) Feature vs Late Fusion"),
    ]
    width = 0.33

    for ax, pair_name, title in bottom_specs:
        subset = (
            quality[quality["pair"] == pair_name]
            .set_index("dimension")
            .loc[["D1", "D2", "D3", "D4", "D5", "D6"]]
            .reset_index()
        )
        x = np.arange(len(subset))
        feature_scores = subset["avg_score_a"].to_numpy()
        fusion_scores = subset["avg_score_b"].to_numpy()

        feature_bars = ax.bar(x - width / 2, feature_scores, width=width, color=FEATURE_COLOR)
        fusion_bars = ax.bar(x + width / 2, fusion_scores, width=width, color=FUSION_COLOR)

        ax.bar_label(feature_bars, labels=[f"{value:.2f}" for value in feature_scores], padding=2, fontsize=8.5)
        ax.bar_label(fusion_bars, labels=[f"{value:.2f}" for value in fusion_scores], padding=2, fontsize=8.5)

        ax.set_title(title, fontsize=11.5, loc="left", pad=8, fontweight="bold")
        ax.set_xticks(x, [DIMENSION_TICK_LABELS[dimension] for dimension in subset["dimension"]])
        ax.set_ylim(0, 5.35)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    ax_b1.set_ylabel("Average rubric score")
    ax_b2.tick_params(axis="y", labelleft=False)

    ax_b1.text(
        0.0,
        1.14,
        "(B) Dimension-wise average scores",
        transform=ax_b1.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    legend_handles = [
        Patch(facecolor=FEATURE_COLOR, label="Feature system / Feature win"),
        Patch(facecolor=FUSION_COLOR, label="Fusion system / Fusion win"),
        Patch(facecolor=TIE_COLOR, label="Tie"),
    ]
    fig.legend(
        handles=legend_handles,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )

    fig.suptitle("Pairwise Evaluation Results", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, CAPTION, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.08, right=0.985, hspace=0.62, wspace=0.18)

    save_figure_variants(fig, output_png, output_pdf)


def main() -> None:
    output_png = OUTPUT_DIR / "Figure_7_1_7_2_combined_panel.png"
    output_pdf = OUTPUT_DIR / "Figure_7_1_7_2_combined_panel.pdf"
    plot_combined_panel_figure(output_png=output_png, output_pdf=output_pdf)

    print("Saved combined panel figure:")
    print(f"- {output_png.relative_to(OUTPUT_DIR.parent)}")
    print(f"- {output_pdf.relative_to(OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()
