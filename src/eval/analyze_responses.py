from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Emotion-aware weighting
# ---------------------------------------------------------------------------

# Weights for D1-D5 (must sum to 1.0 per category)
EMOTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    # High-arousal negative → emotional attunement + strategy fit are critical
    "angry": {
        "D1": 0.30, "D2": 0.25, "D3": 0.15, "D4": 0.15, "D5": 0.15
    },
    "fearful": {
        "D1": 0.25, "D2": 0.25, "D3": 0.15, "D4": 0.15, "D5": 0.20
    },
    "disgust": {
        "D1": 0.25, "D2": 0.25, "D3": 0.20, "D4": 0.15, "D5": 0.15
    },
    # Low-arousal negative → empathy and safety
    "sad": {
        "D1": 0.25, "D2": 0.30, "D3": 0.15, "D4": 0.15, "D5": 0.15
    },
    # Neutral / calm → usefulness and clarity primary
    "neutral": {
        "D1": 0.10, "D2": 0.15, "D3": 0.35, "D4": 0.25, "D5": 0.15
    },
    "calm": {
        "D1": 0.10, "D2": 0.15, "D3": 0.35, "D4": 0.25, "D5": 0.15
    },
    # Positive / high-arousal positive → task help with warmth
    "happy": {
        "D1": 0.15, "D2": 0.20, "D3": 0.30, "D4": 0.20, "D5": 0.15
    },
    "surprise": {
        "D1": 0.20, "D2": 0.20, "D3": 0.25, "D4": 0.20, "D5": 0.15
    },
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "D1": 0.20, "D2": 0.20, "D3": 0.20, "D4": 0.20, "D5": 0.20
}

QUALITY_DIMS = ["D1", "D2", "D3", "D4", "D5"]


def get_weights(emotion: Optional[str]) -> Dict[str, float]:
    """Return emotion-specific D1-D5 weights, defaulting to equal weights."""
    if emotion is None:
        return DEFAULT_WEIGHTS
    return EMOTION_WEIGHTS.get(emotion.lower().strip(), DEFAULT_WEIGHTS)


def compute_weighted_score(row: pd.Series, weights: Dict[str, float]) -> float:
    """Compute weighted overall score from D1-D5 for a single row."""
    total_w = 0.0
    wsum = 0.0
    for dim, w in weights.items():
        val = row.get(dim)
        if val is not None and not pd.isna(val):
            # Scores are already on 1-5 scale in the CSV
            wsum += float(val) * w
            total_w += w
    return wsum / total_w if total_w > 0 else float("nan")


def add_weighted_scores(df: pd.DataFrame, emotion_col: str = "predicted_label") -> pd.DataFrame:
    """Add 'weighted_overall' column using emotion-aware D1-D5 weights."""
    df = df.copy()
    scores = []
    for _, row in df.iterrows():
        emotion = row.get(emotion_col)
        weights = get_weights(emotion)
        scores.append(compute_weighted_score(row, weights))
    df["weighted_overall"] = scores
    return df


# ---------------------------------------------------------------------------
# Single-score analysis
# ---------------------------------------------------------------------------

def analyze_single_scores(
    scores_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Analyze single-score CSV and produce per-strategy and per-emotion summaries.

    Args:
        scores_csv:  Path to the single-scoring CSV from judge.py.
        output_dir:  Directory for output CSVs and plots.

    Returns:
        DataFrame with weighted scores added.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scores_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df)} scored rows from {scores_csv}")

    # Add weighted overall
    df = add_weighted_scores(df, emotion_col="predicted_label")

    # ---- Per-strategy summary ----
    _print_header("Per-Prompt-Strategy Summary")
    strategy_cols = QUALITY_DIMS + ["D6", "overall_d1_d5", "weighted_overall"]
    strategy_cols = [c for c in strategy_cols if c in df.columns]
    strategy_summary = (
        df.groupby("prompt_strategy")[strategy_cols]
        .agg(["mean", "std", "count"])
    )
    strategy_summary.columns = ["_".join(c) for c in strategy_summary.columns]
    strategy_summary = strategy_summary.sort_values("weighted_overall_mean", ascending=False)
    print(strategy_summary.round(3).to_string())

    out_path = output_dir / "strategy_summary.csv"
    strategy_summary.to_csv(out_path, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")

    # ---- Per-emotion summary ----
    _print_header("Per-Predicted-Emotion Summary")
    emotion_summary = (
        df.groupby("predicted_label")[["weighted_overall", "overall_d1_d5", "D6"]]
        .agg(["mean", "std", "count"])
    )
    emotion_summary.columns = ["_".join(c) for c in emotion_summary.columns]
    print(emotion_summary.round(3).to_string())
    emotion_summary.to_csv(output_dir / "emotion_summary.csv", encoding="utf-8-sig")

    # ---- Overall mean per dimension ----
    _print_header("Overall Dimension Means (all strategies combined)")
    for dim in QUALITY_DIMS + ["D6"]:
        if dim in df.columns:
            m = df[dim].mean()
            s = df[dim].std()
            print(f"  {dim}: {m:.2f} ± {s:.2f}")

    # ---- Plots ----
    if PLOT_AVAILABLE:
        _plot_strategy_bars(df, output_dir)
        _plot_dimension_heatmap(df, output_dir)

    return df


# ---------------------------------------------------------------------------
# Pairwise analysis
# ---------------------------------------------------------------------------

def analyze_pairwise(
    pairwise_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Analyze pairwise comparison CSV and report win rates.

    Args:
        pairwise_csv: Path to pairwise CSV from judge.py score_pairwise.
        output_dir:   Directory for output CSVs and plots.

    Returns:
        The loaded pairwise DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pairwise_csv, encoding="utf-8-sig")
    print(f"Loaded {len(df)} pairwise rows from {pairwise_csv}")

    if df.empty:
        print("No data to analyze.")
        return df

    strategy_a = df["strategy_a"].iloc[0]
    strategy_b = df["strategy_b"].iloc[0]
    comparison_type = df["comparison_type"].iloc[0] if "comparison_type" in df.columns else "unknown"

    _print_header(f"Pairwise: {strategy_a} vs {strategy_b} ({comparison_type})")

    # ---- Overall win rates ----
    total = len(df)
    wins_a = (df["winner_overall"] == "A").sum()
    wins_b = (df["winner_overall"] == "B").sum()
    ties   = (df["winner_overall"] == "tie").sum()

    print(f"Total pairs:  {total}")
    print(f"  {strategy_a} wins: {wins_a} ({100*wins_a/total:.1f}%)")
    print(f"  {strategy_b} wins: {wins_b} ({100*wins_b/total:.1f}%)")
    print(f"  ties:        {ties} ({100*ties/total:.1f}%)")
    print(f"\nOverall mean scores (0-100 scale):")
    print(f"  {strategy_a}: {df['overall_a'].mean():.2f} ± {df['overall_a'].std():.2f}")
    print(f"  {strategy_b}: {df['overall_b'].mean():.2f} ± {df['overall_b'].std():.2f}")
    print(f"  delta (B-A): {df['delta_overall'].mean():.2f} ± {df['delta_overall'].std():.2f}")

    # Save win rate summary
    win_summary = pd.DataFrame([{
        "comparison_type": comparison_type,
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "total_pairs": total,
        f"wins_{strategy_a}": wins_a,
        f"wins_{strategy_b}": wins_b,
        "ties": ties,
        f"win_rate_{strategy_a}": round(wins_a / total, 4),
        f"win_rate_{strategy_b}": round(wins_b / total, 4),
        "tie_rate": round(ties / total, 4),
        f"mean_overall_{strategy_a}": round(df["overall_a"].mean(), 3),
        f"mean_overall_{strategy_b}": round(df["overall_b"].mean(), 3),
        "mean_delta": round(df["delta_overall"].mean(), 3),
    }])
    out_path = output_dir / f"win_rates_{strategy_a}_vs_{strategy_b}.csv"
    win_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")

    # ---- Per-emotion win rates ----
    if "predicted_label" in df.columns:
        _print_header("Win Rates by Predicted Emotion")
        emotion_groups = []
        for emotion, group in df.groupby("predicted_label"):
            n = len(group)
            wa = (group["winner_overall"] == "A").sum()
            wb = (group["winner_overall"] == "B").sum()
            wt = (group["winner_overall"] == "tie").sum()
            emotion_groups.append({
                "predicted_label": emotion,
                "n": n,
                f"wins_{strategy_a}": wa,
                f"wins_{strategy_b}": wb,
                "ties": wt,
                f"win_rate_{strategy_b}": round(wb / n, 3),
                "mean_delta": round(group["delta_overall"].mean(), 3),
            })
        emotion_df = pd.DataFrame(emotion_groups).sort_values(
            f"win_rate_{strategy_b}", ascending=False
        )
        print(emotion_df.to_string(index=False))
        emotion_df.to_csv(
            output_dir / f"emotion_win_rates_{strategy_a}_vs_{strategy_b}.csv",
            index=False, encoding="utf-8-sig",
        )

    # ---- D6 signal utilization comparison ----
    if "A_D6_signal" in df.columns and "B_D6_signal" in df.columns:
        _print_header("D6 Signal Utilization (mechanism metric)")
        print(f"  {strategy_a} D6: {df['A_D6_signal'].mean():.2f}")
        print(f"  {strategy_b} D6: {df['B_D6_signal'].mean():.2f}")

    # ---- Plots ----
    if PLOT_AVAILABLE:
        _plot_pairwise_win_bars(df, strategy_a, strategy_b, output_dir)
        _plot_score_distributions(df, strategy_a, strategy_b, output_dir)

    return df


# ---------------------------------------------------------------------------
# Full combined report
# ---------------------------------------------------------------------------

def generate_full_report(
    scores_csv: Optional[Path],
    pairwise_csvs: List[Path],
    output_dir: Path,
) -> None:
    """Generate a comprehensive text report combining single and pairwise analyses.

    Args:
        scores_csv:    Path to single-score CSV (optional).
        pairwise_csvs: List of pairwise CSV paths.
        output_dir:    Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("RAVDESS AV-FUSION LLM EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # ---- Single scores section ----
    if scores_csv is not None and scores_csv.exists():
        df = pd.read_csv(scores_csv, encoding="utf-8-sig")
        df = add_weighted_scores(df, "predicted_label")

        lines.append("1. SINGLE RESPONSE SCORES")
        lines.append("-" * 80)
        lines.append(f"Total scored responses: {len(df)}")
        lines.append(f"Models: {df.get('llm_model', pd.Series()).unique().tolist()}")
        lines.append(f"Judge:  {df.get('judge_model', pd.Series()).unique().tolist()}")
        lines.append("")

        lines.append("Per-strategy weighted overall (D1-D5, emotion-aware weights):")
        for strat, grp in df.groupby("prompt_strategy"):
            m = grp["weighted_overall"].mean()
            s = grp["weighted_overall"].std()
            n = len(grp)
            lines.append(f"  {strat:<12s}: {m:.3f} ± {s:.3f}  (n={n})")
        lines.append("")

        lines.append("Per-strategy unweighted overall (D1-D5, equal weights):")
        if "overall_d1_d5" in df.columns:
            for strat, grp in df.groupby("prompt_strategy"):
                m = grp["overall_d1_d5"].mean()
                lines.append(f"  {strat:<12s}: {m:.2f}")
        lines.append("")

        lines.append("D6 Signal Utilization by strategy:")
        if "D6" in df.columns:
            for strat, grp in df.groupby("prompt_strategy"):
                m = grp["D6"].mean()
                lines.append(f"  {strat:<12s}: {m:.2f} / 5")
        lines.append("")

    # ---- Pairwise section ----
    if pairwise_csvs:
        lines.append("2. PAIRWISE COMPARISON RESULTS")
        lines.append("-" * 80)
        for pw_path in pairwise_csvs:
            if not pw_path.exists():
                continue
            df = pd.read_csv(pw_path, encoding="utf-8-sig")
            if df.empty:
                continue
            strategy_a = df["strategy_a"].iloc[0]
            strategy_b = df["strategy_b"].iloc[0]
            total = len(df)
            wa = (df["winner_overall"] == "A").sum()
            wb = (df["winner_overall"] == "B").sum()
            wt = (df["winner_overall"] == "tie").sum()
            delta = df["delta_overall"].mean()

            lines.append(f"\n  {strategy_a} vs {strategy_b}:")
            lines.append(f"    Total pairs: {total}")
            lines.append(f"    {strategy_a} wins: {wa} ({100*wa/total:.1f}%)")
            lines.append(f"    {strategy_b} wins: {wb} ({100*wb/total:.1f}%)")
            lines.append(f"    Ties:        {wt} ({100*wt/total:.1f}%)")
            lines.append(f"    Mean delta (B-A): {delta:+.2f} (positive = B better)")

            # Winner
            if wb > wa:
                winner_label = f"{strategy_b} is better overall"
            elif wa > wb:
                winner_label = f"{strategy_a} is better overall"
            else:
                winner_label = "No clear winner (equal wins)"
            lines.append(f"    Conclusion: {winner_label}")
        lines.append("")

    # ---- Recommendations section ----
    lines.append("3. RECOMMENDATIONS")
    lines.append("-" * 80)
    if scores_csv is not None and scores_csv.exists():
        df = pd.read_csv(scores_csv, encoding="utf-8-sig")
        df = add_weighted_scores(df, "predicted_label")
        best_strat = df.groupby("prompt_strategy")["weighted_overall"].mean().idxmax()
        best_score = df.groupby("prompt_strategy")["weighted_overall"].mean().max()
        lines.append(f"Best prompt strategy (by weighted overall): {best_strat} ({best_score:.3f})")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved: {report_path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(title)
    print("="*60)


def _plot_strategy_bars(df: pd.DataFrame, output_dir: Path) -> None:
    if not PLOT_AVAILABLE or "prompt_strategy" not in df.columns:
        return

    dims = [d for d in QUALITY_DIMS if d in df.columns]
    if not dims:
        return

    strategies = sorted(df["prompt_strategy"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 5), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    for i, dim in enumerate(dims):
        means = [df[df["prompt_strategy"] == s][dim].mean() for s in strategies]
        stds  = [df[df["prompt_strategy"] == s][dim].std()  for s in strategies]
        axes[i].bar(strategies, means, yerr=stds, capsize=5, alpha=0.8)
        axes[i].set_title(dim, fontsize=11, fontweight="bold")
        axes[i].set_ylim(1, 5)
        axes[i].set_ylabel("Mean Score (1-5)" if i == 0 else "")
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("D1-D5 Scores by Prompt Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "strategy_dimension_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")


def _plot_dimension_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    if not PLOT_AVAILABLE or "prompt_strategy" not in df.columns:
        return

    dims = [d for d in QUALITY_DIMS + ["D6"] if d in df.columns]
    pivot = df.groupby("prompt_strategy")[dims].mean()

    fig, ax = plt.subplots(figsize=(max(6, len(dims) * 1.2), max(3, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5, ax=ax, cbar_kws={"label": "Mean Score (1-5)"},
    )
    ax.set_title("Mean Dimension Scores by Strategy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Prompt Strategy")

    plt.tight_layout()
    out = output_dir / "strategy_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")


def _plot_pairwise_win_bars(
    df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    output_dir: Path,
) -> None:
    if not PLOT_AVAILABLE:
        return

    total = len(df)
    wa = (df["winner_overall"] == "A").sum()
    wb = (df["winner_overall"] == "B").sum()
    wt = (df["winner_overall"] == "tie").sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{strategy_a}\nwins", "tie", f"{strategy_b}\nwins"]
    values = [wa / total, wt / total, wb / total]
    colors = ["#2196F3", "#9E9E9E", "#F44336"]
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_title(f"Pairwise Win Rates: {strategy_a} vs {strategy_b}", fontweight="bold")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"win_rates_{strategy_a}_vs_{strategy_b}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")


def _plot_score_distributions(
    df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    output_dir: Path,
) -> None:
    if not PLOT_AVAILABLE:
        return

    if "overall_a" not in df.columns or "overall_b" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["overall_a"], bins=15, alpha=0.6, label=strategy_a, color="#2196F3")
    ax.hist(df["overall_b"], bins=15, alpha=0.6, label=strategy_b, color="#F44336")
    ax.set_xlabel("Overall Score (0-100, D1-D5)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution: {strategy_a} vs {strategy_b}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"score_dist_{strategy_a}_vs_{strategy_b}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")
