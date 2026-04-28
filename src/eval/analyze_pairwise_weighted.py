from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAIR_CONFIGS: List[Dict[str, str]] = [
    {
        "name": "early_vs_baseline",
        "file_a": "outputs/llm_responses/responses_early_noppg_fusion.jsonl",
        "file_b": "outputs/llm_responses/responses_baseline.jsonl",
        "label_a": "early fusion (no PPG in classifier) + PPG in prompt",
        "label_b": "text-only baseline",
    },
    {
        "name": "late_vs_baseline",
        "file_a": "outputs/llm_responses/responses_late_noppg_fusion.jsonl",
        "file_b": "outputs/llm_responses/responses_baseline.jsonl",
        "label_a": "late fusion (no PPG in classifier) + PPG in prompt",
        "label_b": "text-only baseline",
    },
    {
        "name": "early_vs_late",
        "file_a": "outputs/llm_responses/responses_early_noppg_fusion.jsonl",
        "file_b": "outputs/llm_responses/responses_late_noppg_fusion.jsonl",
        "label_a": "early fusion (no PPG in classifier) + PPG in prompt",
        "label_b": "late fusion (no PPG in classifier) + PPG in prompt",
    },
]

QUALITY_DIMS = ["D1", "D2", "D3", "D4", "D5"]
SCENARIO_ORDER = ["all", "task_dominant", "emotion_dominant", "hybrid_emotion_task"]
SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "task_dominant": {"D1": 0.10, "D2": 0.20, "D3": 0.35, "D4": 0.20, "D5": 0.15},
    "emotion_dominant": {"D1": 0.30, "D2": 0.25, "D3": 0.10, "D4": 0.15, "D5": 0.20},
    "hybrid_emotion_task": {"D1": 0.20, "D2": 0.20, "D3": 0.25, "D4": 0.15, "D5": 0.20},
}

TASK_CUES = [
    "outline", "rewrite", "explain", "compare", "paragraph", "report", "presentation",
    "structure", "draft", "email", "message", "checklist", "plan", "troubleshooting",
    "debug", "design", "evaluation", "limitations", "academic", "wording", "strategy",
    "steps", "priorit", "clarify", "interview", "question", "rule", "routine", "summary",
]
EMOTION_CUES = [
    "feel", "feeling", "felt", "worried", "worry", "anxious", "panic", "panicking",
    "nervous", "stress", "stressed", "overwhelmed", "frustrated", "annoyed",
    "disappointed", "upset", "angry", "sad", "guilty", "ashamed", "relief", "relieved",
    "scared", "afraid", "draining", "exhausted", "tense", "hurt", "lighter", "understood",
    "supported", "burdened", "self-doubt", "confidence", "discouraged",
]
SUPPORT_CUES = [
    "being heard", "understood", "emotional space", "less self-anger", "more mentally breathable",
    "feel lighter", "support when i need it", "emotional steadiness", "without making me feel worse",
]


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s\-]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _matched_cues(text: str, cues: List[str]) -> List[str]:
    normalized = _normalize_text(text)
    hits: List[str] = []
    for cue in cues:
        if cue.endswith(" "):
            continue
        if cue in normalized:
            hits.append(cue)
    return sorted(set(hits))


def classify_scenario(user_text: str) -> Dict[str, Any]:
    normalized = _normalize_text(user_text)
    task_hits = _matched_cues(normalized, TASK_CUES)
    emotion_hits = _matched_cues(normalized, EMOTION_CUES)
    support_hits = _matched_cues(normalized, SUPPORT_CUES)

    task_score = len(task_hits)
    emotion_score = len(emotion_hits) + (2 if support_hits else 0)

    if task_score >= emotion_score + 2:
        scenario = "task_dominant"
        rule = "task_score >= emotion_score + 2"
    elif emotion_score >= task_score + 3 and task_score <= 2:
        scenario = "emotion_dominant"
        rule = "emotion_score >= task_score + 3 and task_score <= 2"
    else:
        scenario = "hybrid_emotion_task"
        rule = "mixed task and emotion cues"

    return {
        "scenario": scenario,
        "scenario_task_score": task_score,
        "scenario_emotion_score": emotion_score,
        "scenario_task_hits": ", ".join(task_hits),
        "scenario_emotion_hits": ", ".join(emotion_hits),
        "scenario_support_hits": ", ".join(support_hits),
        "scenario_rule": rule,
    }


def _canonical_av_key(raw_key: Any) -> str:
    text = str(raw_key).strip()
    match = re.fullmatch(r"(\d+)_(\d+)", text)
    if match:
        return f"{int(match.group(1))}_{int(match.group(2))}"
    return text


def _load_user_texts(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    if not path.is_file():
        return rows

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                rows[_canonical_av_key(row["av_key"])] = str(row.get("user_text", ""))
        return rows

    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        key_column = next((name for name in ("av_key", "stem") if name in df.columns), None)
        text_column = next((name for name in ("user_text", "text") if name in df.columns), None)
        if key_column and text_column:
            for _, row in df[[key_column, text_column]].dropna(subset=[key_column]).iterrows():
                rows[_canonical_av_key(row[key_column])] = str(row.get(text_column, ""))
        return rows

    logger.warning("Unsupported text source format for scenario analysis: %s", path)
    return rows


def _winner_to_side_scores(winner: str) -> tuple[float, float]:
    if winner == "A":
        return 1.0, 0.0
    if winner == "B":
        return 0.0, 1.0
    return 0.5, 0.5


def _win_rates(df: pd.DataFrame, winner_column: str) -> Dict[str, float]:
    if df.empty:
        return {"A": 0.0, "B": 0.0, "tie": 0.0}
    counts = df[winner_column].value_counts()
    total = len(df)
    return {
        "A": round(counts.get("A", 0) / total * 100.0, 1),
        "B": round(counts.get("B", 0) / total * 100.0, 1),
        "tie": round(counts.get("tie", 0) / total * 100.0, 1),
    }


def _weighted_winner(weighted_a: float, weighted_b: float, tolerance: float = 1e-9) -> str:
    if abs(weighted_a - weighted_b) <= tolerance:
        return "tie"
    return "A" if weighted_a > weighted_b else "B"


def annotate_pair_results(pair_config: Dict[str, str], input_dir: Path, output_dir: Path) -> pd.DataFrame:
    results_path = input_dir / pair_config["name"] / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing pairwise results: {results_path}")

    df = pd.read_csv(results_path, encoding="utf-8-sig")
    if df.empty:
        return df

    text_index = _load_user_texts(REPO_ROOT / pair_config["file_a"])
    text_index.update(_load_user_texts(REPO_ROOT / pair_config["file_b"]))

    scenario_columns = {
        "user_text": [],
        "scenario": [],
        "scenario_task_score": [],
        "scenario_emotion_score": [],
        "scenario_task_hits": [],
        "scenario_emotion_hits": [],
        "scenario_support_hits": [],
        "scenario_rule": [],
        "weighted_A": [],
        "weighted_B": [],
        "weighted_winner": [],
        "weighted_margin_A_minus_B": [],
    }

    for _, row in df.iterrows():
        av_key = str(row["av_key"])
        user_text = text_index.get(av_key, "")
        scenario_info = classify_scenario(user_text)
        weights = SCENARIO_WEIGHTS[scenario_info["scenario"]]

        weighted_a = 0.0
        weighted_b = 0.0
        for dimension in QUALITY_DIMS:
            left_score, right_score = _winner_to_side_scores(str(row[f"winner_{dimension}"]))
            weighted_a += weights[dimension] * left_score
            weighted_b += weights[dimension] * right_score

        scenario_columns["user_text"].append(user_text)
        for key in (
            "scenario",
            "scenario_task_score",
            "scenario_emotion_score",
            "scenario_task_hits",
            "scenario_emotion_hits",
            "scenario_support_hits",
            "scenario_rule",
        ):
            scenario_columns[key].append(scenario_info[key])
        scenario_columns["weighted_A"].append(round(weighted_a, 4))
        scenario_columns["weighted_B"].append(round(weighted_b, 4))
        scenario_columns["weighted_winner"].append(_weighted_winner(weighted_a, weighted_b))
        scenario_columns["weighted_margin_A_minus_B"].append(round(weighted_a - weighted_b, 4))

    annotated = df.assign(**scenario_columns)
    pair_dir = output_dir / pair_config["name"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = pair_dir / "weighted_results.csv"
    annotated.to_csv(annotated_path, index=False, encoding="utf-8-sig")
    logger.info("[%s] Wrote annotated weighted results to %s", pair_config["name"], annotated_path)
    return annotated


def build_summary(all_results: Dict[str, pd.DataFrame], pair_configs: List[Dict[str, str]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for pair_config in pair_configs:
        df = all_results.get(pair_config["name"])
        if df is None or df.empty:
            continue

        for scope in SCENARIO_ORDER:
            subset = df if scope == "all" else df[df["scenario"] == scope]
            if subset.empty:
                continue

            unweighted_rates = _win_rates(subset, "winner_overall")
            weighted_rates = _win_rates(subset, "weighted_winner")
            d6_rates = _win_rates(subset, "winner_D6")

            records.append(
                {
                    "pair": pair_config["name"],
                    "scope": scope,
                    "label_a": pair_config["label_a"],
                    "label_b": pair_config["label_b"],
                    "n": len(subset),
                    "unweighted_a_win_rate": unweighted_rates["A"],
                    "unweighted_b_win_rate": unweighted_rates["B"],
                    "unweighted_tie_rate": unweighted_rates["tie"],
                    "weighted_a_win_rate": weighted_rates["A"],
                    "weighted_b_win_rate": weighted_rates["B"],
                    "weighted_tie_rate": weighted_rates["tie"],
                    "mean_weighted_a": round(subset["weighted_A"].mean(), 4),
                    "mean_weighted_b": round(subset["weighted_B"].mean(), 4),
                    "mean_weighted_margin_a_minus_b": round(subset["weighted_margin_A_minus_B"].mean(), 4),
                    "d6_a_win_rate": d6_rates["A"],
                    "d6_b_win_rate": d6_rates["B"],
                    "d6_tie_rate": d6_rates["tie"],
                }
            )

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["pair", "scope"], ignore_index=True)
    return summary_df


def write_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    lines: List[str] = []
    lines.append("PAIRWISE POST-HOC SCENARIO-WEIGHTED ANALYSIS")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("Primary overall result = original winner_overall from D1-D5.")
    lines.append("Secondary weighted result = scenario-aware weighting over D1-D5 winners.")
    lines.append("D6 remains separate and unweighted.")
    lines.append("")

    for pair_name in summary_df["pair"].unique():
        pair_rows = summary_df[summary_df["pair"] == pair_name]
        if pair_rows.empty:
            continue

        base = pair_rows[pair_rows["scope"] == "all"].iloc[0]
        lines.append(pair_name)
        lines.append(f"A: {base['label_a']}")
        lines.append(f"B: {base['label_b']}")
        lines.append("")

        for scope in SCENARIO_ORDER:
            scope_rows = pair_rows[pair_rows["scope"] == scope]
            if scope_rows.empty:
                continue
            row = scope_rows.iloc[0]
            lines.append(f"{scope} (n={int(row['n'])})")
            lines.append(
                f"  primary overall: A {row['unweighted_a_win_rate']:.1f}% | "
                f"B {row['unweighted_b_win_rate']:.1f}% | tie {row['unweighted_tie_rate']:.1f}%"
            )
            lines.append(
                f"  weighted overall: A {row['weighted_a_win_rate']:.1f}% | "
                f"B {row['weighted_b_win_rate']:.1f}% | tie {row['weighted_tie_rate']:.1f}%"
            )
            lines.append(
                f"  mean weighted score: A {row['mean_weighted_a']:.4f} | "
                f"B {row['mean_weighted_b']:.4f} | "
                f"margin A-B {row['mean_weighted_margin_a_minus_b']:.4f}"
            )
            lines.append(
                f"  D6 mechanism: A {row['d6_a_win_rate']:.1f}% | "
                f"B {row['d6_b_win_rate']:.1f}% | tie {row['d6_tie_rate']:.1f}%"
            )
            lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Wrote report to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc scenario-aware weighted analysis on existing pairwise results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/pairwise",
        help="Directory containing pairwise/{pair_name}/results.csv outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pairwise_weighted",
        help="Directory for weighted summaries and annotated pairwise results.",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        choices=[config["name"] for config in PAIR_CONFIGS],
        default=None,
        help="Optional subset of pairs to analyze.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = REPO_ROOT / args.input_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    active_pairs = PAIR_CONFIGS
    if args.pairs:
        requested = set(args.pairs)
        active_pairs = [config for config in PAIR_CONFIGS if config["name"] in requested]

    all_results: Dict[str, pd.DataFrame] = {}
    for pair_config in active_pairs:
        logger.info("Analyzing pair: %s", pair_config["name"])
        all_results[pair_config["name"]] = annotate_pair_results(pair_config, input_dir, output_dir)

    summary_df = build_summary(all_results, active_pairs)
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.txt"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)
    logger.info("Wrote summary CSV to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
