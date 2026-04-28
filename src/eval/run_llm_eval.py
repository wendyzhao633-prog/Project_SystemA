from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="run_llm_eval",
    help="RAVDESS AV-Fusion LLM evaluation pipeline.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@app.command()
def generate(
    predictions: Path = typer.Option(
        ...,
        help="Early fusion predictions JSONL.",
        exists=True, file_okay=True, dir_okay=False,
    ),
    labels: Path = typer.Option(
        "dataset_200_labeled.csv",
        help="Path to dataset_200_labeled.csv.",
    ),
    ppg_root: Path = typer.Option(
        "data/raw/custom/ppg",
        help="Directory containing ppg_P_NN.csv files.",
    ),
    output: Path = typer.Option(
        ...,
        help="Output JSONL path for generated responses.",
    ),
    model: str = typer.Option(
        "gpt-5.4",
        help="OpenAI model name (e.g. gpt-5.4, gpt-4o).",
    ),
    strategies: str = typer.Option(
        "explicit",
        help="Comma-separated prompt strategies to run.",
    ),
    temperature: float = typer.Option(0.2, help="Sampling temperature."),
    max_tokens: int = typer.Option(
        1200,
        help="Max completion tokens per response, including reasoning tokens.",
    ),
    reasoning_effort: str = typer.Option(
        "none",
        help="GPT-5 reasoning effort: none, low, medium, high, or xhigh.",
    ),
    concurrency: int = typer.Option(5, help="Max concurrent API calls."),
    max_calls: Optional[int] = typer.Option(None, help="Budget cap (None = unlimited)."),
    ppg_cache: Optional[Path] = typer.Option(
        None,
        help="Directory to cache extracted PPG features as JSON (optional).",
    ),
    fusion_method: str = typer.Option(
        "early_fusion",
        help="Label written to output rows (e.g. early_fusion_3cnn_gated_ppg).",
    ),
) -> None:
    """Generate LLM responses from early fusion predictions."""
    _check_api_key()

    strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
    valid = {"empathy_then_help", "text_only_baseline"}
    bad = set(strategy_list) - valid
    if bad:
        typer.echo(f"ERROR: Unknown strategies: {bad}. Valid: {valid}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Predictions: {predictions}")
    typer.echo(f"Labels CSV:  {labels}")
    typer.echo(f"PPG root:    {ppg_root}")
    typer.echo(f"Output:      {output}")
    typer.echo(f"Model:       {model}")
    typer.echo(f"Strategies:  {strategy_list}")
    typer.echo(f"Reasoning:   {reasoning_effort}")
    typer.echo(f"Concurrency: {concurrency}")
    if max_calls:
        typer.echo(f"Budget cap:  {max_calls} calls")

    from src.eval.llm_response import generate_responses
    generate_responses(
        predictions_jsonl=predictions,
        labels_csv=labels,
        ppg_root=ppg_root,
        output_jsonl=output,
        model=model,
        prompt_strategies=strategy_list,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        concurrency=concurrency,
        max_calls=max_calls,
        ppg_cache_dir=ppg_cache,
        fusion_method=fusion_method,
    )
    typer.echo(f"\nDone. Responses written to: {output}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

@app.command()
def score(
    input: Path = typer.Option(
        ...,
        help="Input JSONL with generated responses.",
        exists=True, file_okay=True, dir_okay=False,
    ),
    output: Path = typer.Option(
        ...,
        help="Output CSV path for scores.",
    ),
    mode: str = typer.Option(
        "single",
        help="Scoring mode: 'single' or 'pairwise'.",
    ),
    judge_model: str = typer.Option(
        "gpt-5.4",
        help="OpenAI model to use as judge.",
    ),
    concurrency: int = typer.Option(5, help="Max concurrent judge calls."),
    max_calls: Optional[int] = typer.Option(None, help="Budget cap."),
    strategy_a: str = typer.Option(
        "baseline",
        help="[pairwise only] First strategy.",
    ),
    strategy_b: str = typer.Option(
        "soft",
        help="[pairwise only] Second strategy.",
    ),
    comparison_type: str = typer.Option(
        "prompt",
        help="[pairwise only] Label for the comparison (e.g. prompt, fusion_method).",
    ),
    tie_margin: float = typer.Option(
        3.0,
        help="[pairwise only] Score difference below which we declare a tie (0-100 scale).",
    ),
) -> None:
    """Score responses with LLM-as-judge."""
    _check_api_key()

    if mode not in ("single", "pairwise"):
        typer.echo(f"ERROR: Unknown mode '{mode}'. Use 'single' or 'pairwise'.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Mode:        {mode}")
    typer.echo(f"Input:       {input}")
    typer.echo(f"Output:      {output}")
    typer.echo(f"Judge model: {judge_model}")

    if mode == "single":
        from src.eval.judge import score_responses
        df = score_responses(
            responses_jsonl=input,
            output_csv=output,
            judge_model=judge_model,
            concurrency=concurrency,
            max_calls=max_calls,
        )
        typer.echo(f"\nScored {len(df)} responses. Saved to: {output}")

    else:  # pairwise
        typer.echo(f"Comparing:   {strategy_a} vs {strategy_b}")
        from src.eval.judge import score_pairwise
        df = score_pairwise(
            responses_jsonl=input,
            output_csv=output,
            comparison_type=comparison_type,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            judge_model=judge_model,
            concurrency=concurrency,
            max_calls=max_calls,
            tie_margin=tie_margin,
        )
        typer.echo(f"\nJudged {len(df)} pairs. Saved to: {output}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    scores: Optional[Path] = typer.Option(
        None,
        help="Single-score CSV from 'score --mode single'.",
    ),
    pairwise: Optional[str] = typer.Option(
        None,
        help="Comma-separated pairwise CSV paths from 'score --mode pairwise'.",
    ),
    output: Path = typer.Option(
        "outputs/analysis/llm_eval/",
        help="Output directory for analysis results.",
    ),
) -> None:
    """Analyze evaluation scores and produce summaries, plots, and a report."""
    from src.eval.analyze_responses import (
        analyze_single_scores,
        analyze_pairwise,
        generate_full_report,
    )

    output.mkdir(parents=True, exist_ok=True)

    if scores is not None and scores.exists():
        typer.echo(f"\n[1/3] Analyzing single scores: {scores}")
        analyze_single_scores(scores, output)
    elif scores is not None:
        typer.echo(f"WARNING: scores file not found: {scores}", err=True)

    pairwise_paths: List[Path] = []
    if pairwise:
        for p in pairwise.split(","):
            p = p.strip()
            if p:
                pw_path = Path(p)
                pairwise_paths.append(pw_path)
                if pw_path.exists():
                    typer.echo(f"\n[2/3] Analyzing pairwise: {pw_path}")
                    analyze_pairwise(pw_path, output)
                else:
                    typer.echo(f"WARNING: pairwise file not found: {pw_path}", err=True)

    typer.echo(f"\n[3/3] Generating combined report...")
    generate_full_report(
        scores_csv=scores,
        pairwise_csvs=pairwise_paths,
        output_dir=output,
    )

    typer.echo(f"\nAnalysis complete. Results in: {output}")


# ---------------------------------------------------------------------------
# Convenience: run all three steps in sequence
# ---------------------------------------------------------------------------

@app.command()
def run_all(
    predictions: Path = typer.Option(..., help="Early fusion predictions JSONL."),
    labels: Path = typer.Option("dataset_200_labeled.csv", help="Labels CSV."),
    ppg_root: Path = typer.Option("data/raw/custom/ppg", help="PPG CSV root dir."),
    responses_output: Path = typer.Option(
        "outputs/llm_responses/responses.jsonl",
        help="Path for generated responses JSONL.",
    ),
    scores_output: Path = typer.Option(
        "outputs/llm_scores/single_scores.csv",
        help="Path for single scores CSV.",
    ),
    analysis_output: Path = typer.Option(
        "outputs/analysis/llm_eval/",
        help="Directory for analysis outputs.",
    ),
    model: str = typer.Option("gpt-5.4", help="OpenAI model for generation."),
    judge_model: str = typer.Option("gpt-5.4", help="OpenAI model for judging."),
    strategies: str = typer.Option("baseline,soft,explicit", help="Prompt strategies."),
    concurrency: int = typer.Option(5, help="Max concurrent API calls."),
    max_calls: Optional[int] = typer.Option(None, help="Budget cap per step."),
    max_tokens: int = typer.Option(
        1200,
        help="Max completion tokens per generation call, including reasoning tokens.",
    ),
    reasoning_effort: str = typer.Option(
        "none",
        help="GPT-5 reasoning effort for generation calls.",
    ),
    fusion_method: str = typer.Option("early_fusion", help="Fusion method label."),
) -> None:
    """Run the full pipeline: generate → score (single + all pairwise combos) → analyze."""
    _check_api_key()

    strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
    typer.echo("=" * 60)
    typer.echo("RAVDESS AV-FUSION LLM EVALUATION — FULL PIPELINE")
    typer.echo("=" * 60)

    # Step 1: Generate responses
    typer.echo("\n[STEP 1/3] Generating responses...")
    from src.eval.llm_response import generate_responses
    generate_responses(
        predictions_jsonl=predictions,
        labels_csv=labels,
        ppg_root=ppg_root,
        output_jsonl=responses_output,
        model=model,
        prompt_strategies=strategy_list,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        concurrency=concurrency,
        max_calls=max_calls,
        fusion_method=fusion_method,
    )

    # Step 2: Single scoring
    typer.echo("\n[STEP 2/3] Scoring responses...")
    from src.eval.judge import score_responses, score_pairwise
    score_responses(
        responses_jsonl=responses_output,
        output_csv=scores_output,
        judge_model=judge_model,
        concurrency=concurrency,
        max_calls=max_calls,
    )

    # Pairwise for all consecutive strategy pairs
    pairwise_paths: List[Path] = []
    if len(strategy_list) >= 2:
        pairs = list(zip(strategy_list, strategy_list[1:]))
        pairs.append((strategy_list[0], strategy_list[-1]))  # first vs last
        for strat_a, strat_b in set(pairs):
            if strat_a == strat_b:
                continue
            pw_path = scores_output.parent / f"pairwise_{strat_a}_vs_{strat_b}.csv"
            pairwise_paths.append(pw_path)
            score_pairwise(
                responses_jsonl=responses_output,
                output_csv=pw_path,
                strategy_a=strat_a,
                strategy_b=strat_b,
                judge_model=judge_model,
                concurrency=concurrency,
                max_calls=max_calls,
            )

    # Step 3: Analyze
    typer.echo("\n[STEP 3/3] Analyzing results...")
    from src.eval.analyze_responses import (
        analyze_single_scores,
        analyze_pairwise,
        generate_full_report,
    )
    analysis_output.mkdir(parents=True, exist_ok=True)
    analyze_single_scores(scores_output, analysis_output)
    for pw_path in pairwise_paths:
        if pw_path.exists():
            analyze_pairwise(pw_path, analysis_output)
    generate_full_report(scores_output, pairwise_paths, analysis_output)

    typer.echo("\n" + "=" * 60)
    typer.echo("PIPELINE COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"Responses:  {responses_output}")
    typer.echo(f"Scores:     {scores_output}")
    typer.echo(f"Analysis:   {analysis_output}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "  Create a .env file with OPENAI_API_KEY=your_key_here\n"
            "  or export it in your shell.",
            err=True,
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
