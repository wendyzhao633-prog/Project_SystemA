from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DIMENSION_KEYS = ["D1", "D2", "D3", "D4", "D5", "D6"]
QUALITY_DIMS   = ["D1", "D2", "D3", "D4", "D5"]

# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------

RUBRICS_INLINE = """
## D1 Emotional Attunement & Calibration
Whether the assistant correctly captures the user's emotional state (or its absence)
and matches intensity appropriately without over- or under-reacting.
5=well calibrated; 4=slightly generic; 3=emotionally generic; 2=notable mismatch; 1=dismissive/escalating.

## D2 Emotion-to-Response Strategy Fit
Whether the assistant converts the emotional state into the RIGHT response strategy.
5=strategy clearly matches emotion+task; 4=mostly appropriate; 3=reasonable generic help;
2=poorly matched; 1=worsens the interaction.
Examples: anxious→steadier wording; angry→validate frustration; sad→gentler activation.

## D3 Task Usefulness & Correctness
Whether the response genuinely advances the user's actual goal with actionable, correct content.
5=directly moves task forward; 4=helpful with minor gaps; 3=somewhat generic;
2=vague/hard to execute; 1=off-topic or misleading.

## D4 Clarity & Cognitive Load Fit
Whether the response is easy to follow, well structured, and sized appropriately.
5=well structured & easy to act on; 4=clear but slightly wordy; 3=understandable but scattered;
2=dense or rambling; 1=confusing.

## D5 Tone Safety, Respect & Risk Calibration
Whether the response is respectful, non-judgmental, and appropriately cautious.
5=supportive, no blame/pressure, appropriate caution; 4=generally safe with minor awkwardness;
3=neutral and harmless; 2=some blame/pressure/dismissiveness; 1=shaming or clearly unsafe.

## D6 Emotion Signal Utilization (mechanism metric — report separately from D1-D5)
Whether the assistant shows behavioral evidence that it used the injected emotion context
(predicted label, confidence, PPG) to adapt its response vs. giving a generic answer.
5=clear evidence of signal use; 4=uses signal shallowly; 3=slight personalization;
2=mostly generic; 1=contradicts or misuses the signal.
"""


def _single_judge_messages(
    user_text: str,
    response: str,
    predicted_label: str,
    confidence: float,
    ppg_summary: Dict[str, float],
    prompt_strategy: str,
) -> List[Dict[str, str]]:
    system = (
        "You are a precise evaluator for AI assistant responses in emotional contexts.\n"
        "Return STRICT JSON only — no markdown, no extra keys.\n"
        "Do not reward verbosity alone. Do not reward explicit emotion words alone.\n"
        "Do not penalize concise but well-calibrated responses."
    )

    ppg_str = ", ".join(f"{k}={v}" for k, v in ppg_summary.items()) or "unavailable"

    schema = (
        '{"D1":1-5,"D2":1-5,"D3":1-5,"D4":1-5,"D5":1-5,"D6":1-5,'
        '"rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}}'
    )

    user = (
        "RUBRICS:\n"
        f"{RUBRICS_INLINE}\n\n"
        "IMPORTANT SCORING RULES:\n"
        "- Score D1-D5 as quality dimensions.\n"
        "- Score D6 as a separate mechanism dimension (signal utilization).\n"
        "- For 'baseline' prompt_strategy, D6 should reflect whether the response would "
        "have been different if emotion context were available; score 3 (neutral) as default "
        "since no signal was provided.\n"
        "- Be strict and calibration-aware.\n\n"
        f"CONTEXT:\n"
        f"prompt_strategy: {prompt_strategy}\n"
        f"predicted_emotion: {predicted_label} (confidence={confidence:.2f})\n"
        f"ppg_features: {ppg_str}\n\n"
        f"USER MESSAGE:\n{user_text}\n\n"
        f"RESPONSE TO SCORE:\n{response}\n\n"
        f"Output JSON schema:\n{schema}\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pairwise_judge_messages(
    user_text: str,
    response_a: str,
    response_b: str,
    strategy_a: str,
    strategy_b: str,
    predicted_label: str,
    confidence: float,
    ppg_summary: Dict[str, float],
) -> List[Dict[str, str]]:
    system = (
        "You are a blind pairwise evaluator for AI assistant responses.\n"
        "Return STRICT JSON only — no markdown, no extra keys.\n"
        "Do not reward verbosity alone. Do not reward explicit emotion words alone.\n"
        "Do not penalize concise but well-calibrated responses.\n"
        "Score A and B independently on each dimension."
    )

    ppg_str = ", ".join(f"{k}={v}" for k, v in ppg_summary.items()) or "unavailable"

    schema = (
        '{\n'
        '  "A": {"D1":1-5,"D2":1-5,"D3":1-5,"D4":1-5,"D5":1-5,"D6":1-5,\n'
        '        "rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}},\n'
        '  "B": {"D1":1-5,"D2":1-5,"D3":1-5,"D4":1-5,"D5":1-5,"D6":1-5,\n'
        '        "rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}}\n'
        '}'
    )

    user = (
        "RUBRICS:\n"
        f"{RUBRICS_INLINE}\n\n"
        "IMPORTANT SCORING RULES:\n"
        "- Score D1-D5 as quality dimensions. D6 is a separate mechanism dimension.\n"
        "- Do NOT compute winner_overall in JSON — only raw scores.\n"
        "- For 'baseline' strategy, D6 defaults to 3 (no signal provided).\n"
        "- Be strict and calibration-aware.\n\n"
        f"CONTEXT:\n"
        f"predicted_emotion: {predicted_label} (confidence={confidence:.2f})\n"
        f"ppg_features: {ppg_str}\n\n"
        f"USER MESSAGE:\n{user_text}\n\n"
        f"Strategy for A: {strategy_a}\n"
        f"RESPONSE A:\n{response_a}\n\n"
        f"Strategy for B: {strategy_b}\n"
        f"RESPONSE B:\n{response_b}\n\n"
        f"Output JSON schema:\n{schema}\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"Could not parse judge JSON. Raw:\n{text[:2000]}")


def _coerce_score(val: Any) -> int:
    try:
        return max(1, min(5, int(val)))
    except Exception:
        return 3  # neutral default on parse failure


def _coerce_dim_scores(obj: Dict[str, Any]) -> Dict[str, int]:
    return {d: _coerce_score(obj.get(d, 3)) for d in DIMENSION_KEYS}


def _score_to_100(score_1_to_5: int) -> float:
    return ((score_1_to_5 - 1) / 4.0) * 100.0


def _overall_score(scores: Dict[str, int]) -> float:
    """Compute overall from D1-D5 only (D6 is always separate)."""
    vals = [_score_to_100(scores[d]) for d in QUALITY_DIMS]
    return round(sum(vals) / len(vals), 2)


def _winner(overall_a: float, overall_b: float, tie_margin: float) -> str:
    diff = overall_b - overall_a
    if abs(diff) < tie_margin:
        return "tie"
    return "B" if diff > 0 else "A"


# ---------------------------------------------------------------------------
# Async API helper
# ---------------------------------------------------------------------------

async def _call_openai(
    client,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 5,
) -> str:
    delay = 1.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=900,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            s = str(e).lower()
            if ("rate" in s or "429" in s or "quota" in s or "timeout" in s or "50" in s) \
                    and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                logger.warning("Judge API error (attempt %d): %s — retry in %.1fs", attempt + 1, e, wait)
                await asyncio.sleep(wait)
                last_err = e
            else:
                raise
    raise RuntimeError(f"Max retries exceeded: {last_err}")


# ---------------------------------------------------------------------------
# Single scoring
# ---------------------------------------------------------------------------

def score_responses(
    responses_jsonl: Path,
    output_csv: Path,
    judge_model: str = "gpt-5.4",
    concurrency: int = 5,
    max_calls: Optional[int] = None,
) -> pd.DataFrame:
    """Score every response in the JSONL on D1-D6 independently."""
    return asyncio.run(
        _score_async(responses_jsonl, output_csv, judge_model, concurrency, max_calls)
    )


async def _score_async(
    responses_jsonl: Path,
    output_csv: Path,
    judge_model: str,
    concurrency: int,
    max_calls: Optional[int],
) -> pd.DataFrame:
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError("openai package not installed. Run: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = AsyncOpenAI(api_key=api_key)

    from src.eval.llm_response import load_responses
    rows = load_responses(responses_jsonl)

    # Resume support
    done_keys: set = set()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        existing = pd.read_csv(output_csv, encoding="utf-8-sig")
        for _, r in existing.iterrows():
            done_keys.add((r["av_key"], r["prompt_strategy"]))
        logger.info("Resuming: %d scores already done", len(done_keys))

    work = [r for r in rows if (r["av_key"], r["prompt_strategy"]) not in done_keys]
    if max_calls is not None:
        work = work[:max_calls]
    logger.info("Scoring %d responses", len(work))

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict] = []
    lock = asyncio.Lock()

    async def score_one(row: Dict) -> Optional[Dict]:
        async with sem:
            ppg = row.get("ppg_features") or {}
            messages = _single_judge_messages(
                user_text=row["user_text"],
                response=row["response"],
                predicted_label=row.get("predicted_label", "neutral"),
                confidence=row.get("prediction_confidence", 0.0),
                ppg_summary=ppg,
                prompt_strategy=row.get("prompt_strategy", "baseline"),
            )
            try:
                raw = await _call_openai(client, messages, judge_model)
            except Exception as e:
                logger.error("Judge error for %s/%s: %s", row["av_key"], row["prompt_strategy"], e)
                return None

            try:
                parsed = _parse_json(raw)
            except ValueError as e:
                logger.error("Parse error for %s/%s: %s", row["av_key"], row["prompt_strategy"], e)
                return None

            scores = _coerce_dim_scores(parsed)
            rationale = parsed.get("rationale", {})

            out = {
                "av_key":                row["av_key"],
                "participant_id":        row.get("participant_id"),
                "sample_no":             row.get("sample_no"),
                "true_label":            row.get("true_label"),
                "predicted_label":       row.get("predicted_label"),
                "prediction_confidence": row.get("prediction_confidence"),
                "prompt_strategy":       row.get("prompt_strategy"),
                "fusion_method":         row.get("fusion_method"),
                "llm_model":             row.get("llm_model"),
                "judge_model":           judge_model,
                **{f"D{i}": scores[f"D{i}"] for i in range(1, 7)},
                "overall_d1_d5":         _overall_score(scores),
                "D6_signal_use":         _score_to_100(scores["D6"]),
                **{f"rationale_D{i}": rationale.get(f"D{i}", "") for i in range(1, 7)},
                "judge_raw":             raw[:1500],
                "scored_at":             datetime.now(timezone.utc).isoformat(),
            }

            async with lock:
                results.append(out)
                # Append to CSV incrementally
                df_row = pd.DataFrame([out])
                write_header = not output_csv.exists()
                df_row.to_csv(output_csv, mode="a", header=write_header,
                              index=False, encoding="utf-8-sig")

            return out

    await asyncio.gather(*[score_one(r) for r in work])

    # Load and return full CSV
    if output_csv.exists():
        return pd.read_csv(output_csv, encoding="utf-8-sig")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Pairwise scoring
# ---------------------------------------------------------------------------

def score_pairwise(
    responses_jsonl: Path,
    output_csv: Path,
    comparison_type: str = "prompt",
    strategy_a: str = "baseline",
    strategy_b: str = "soft",
    judge_model: str = "gpt-5.4",
    concurrency: int = 5,
    max_calls: Optional[int] = None,
    tie_margin: float = 3.0,
) -> pd.DataFrame:
    """Compare two prompt strategies pairwise for every sample.

    Args:
        responses_jsonl: JSONL with generated responses (all strategies).
        output_csv:      Output CSV path.
        comparison_type: Label for the comparison (e.g. "prompt", "fusion_method").
        strategy_a:      First strategy to compare.
        strategy_b:      Second strategy to compare.
        judge_model:     OpenAI model for judging.
        concurrency:     Max concurrent API calls.
        max_calls:       Budget cap.
        tie_margin:      Score difference below which we declare a tie (0-100 scale).
    """
    return asyncio.run(
        _pairwise_async(
            responses_jsonl, output_csv, comparison_type,
            strategy_a, strategy_b,
            judge_model, concurrency, max_calls, tie_margin,
        )
    )


async def _pairwise_async(
    responses_jsonl: Path,
    output_csv: Path,
    comparison_type: str,
    strategy_a: str,
    strategy_b: str,
    judge_model: str,
    concurrency: int,
    max_calls: Optional[int],
    tie_margin: float,
) -> pd.DataFrame:
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError("openai package not installed. Run: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = AsyncOpenAI(api_key=api_key)

    from src.eval.llm_response import load_responses
    rows = load_responses(responses_jsonl)

    # Group by av_key → {strategy: row}
    by_key: Dict[str, Dict[str, Dict]] = {}
    for r in rows:
        key = r["av_key"]
        if key not in by_key:
            by_key[key] = {}
        by_key[key][r["prompt_strategy"]] = r

    # Build pairs
    pairs: List[Tuple[str, Dict, Dict]] = []
    for av_key, strats in by_key.items():
        if strategy_a in strats and strategy_b in strats:
            pairs.append((av_key, strats[strategy_a], strats[strategy_b]))
    logger.info(
        "Pairwise (%s vs %s): %d pairs available",
        strategy_a, strategy_b, len(pairs)
    )

    # Resume support
    done_keys: set = set()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        existing = pd.read_csv(output_csv, encoding="utf-8-sig")
        for _, r in existing.iterrows():
            done_keys.add(r["av_key"])
        logger.info("Resuming: %d pairs already judged", len(done_keys))

    work_pairs = [(k, a, b) for k, a, b in pairs if k not in done_keys]
    if max_calls is not None:
        work_pairs = work_pairs[:max_calls]
    logger.info("Judging %d pairs", len(work_pairs))

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def judge_one(av_key: str, row_a: Dict, row_b: Dict) -> Optional[Dict]:
        async with sem:
            ppg = row_a.get("ppg_features") or {}
            messages = _pairwise_judge_messages(
                user_text=row_a["user_text"],
                response_a=row_a["response"],
                response_b=row_b["response"],
                strategy_a=row_a["prompt_strategy"],
                strategy_b=row_b["prompt_strategy"],
                predicted_label=row_a.get("predicted_label", "neutral"),
                confidence=row_a.get("prediction_confidence", 0.0),
                ppg_summary=ppg,
            )
            try:
                raw = await _call_openai(client, messages, judge_model)
            except Exception as e:
                logger.error("Pairwise judge error for %s: %s", av_key, e)
                return None

            try:
                parsed = _parse_json(raw)
            except ValueError as e:
                logger.error("Parse error for %s: %s", av_key, e)
                return None

            a_scores = _coerce_dim_scores(parsed.get("A", {}))
            b_scores = _coerce_dim_scores(parsed.get("B", {}))

            overall_a = _overall_score(a_scores)
            overall_b = _overall_score(b_scores)
            winner = _winner(overall_a, overall_b, tie_margin)

            out = {
                "comparison_type":   comparison_type,
                "av_key":            av_key,
                "participant_id":    row_a.get("participant_id"),
                "sample_no":         row_a.get("sample_no"),
                "true_label":        row_a.get("true_label"),
                "predicted_label":   row_a.get("predicted_label"),
                "strategy_a":        strategy_a,
                "strategy_b":        strategy_b,
                "fusion_method":     row_a.get("fusion_method"),
                "judge_model":       judge_model,
                **{f"A_D{i}": a_scores[f"D{i}"] for i in range(1, 7)},
                **{f"B_D{i}": b_scores[f"D{i}"] for i in range(1, 7)},
                "overall_a":         overall_a,
                "overall_b":         overall_b,
                "delta_overall":     round(overall_b - overall_a, 2),
                "winner_overall":    winner,
                "winner":            winner,  # alias
                "A_D6_signal":       _score_to_100(a_scores["D6"]),
                "B_D6_signal":       _score_to_100(b_scores["D6"]),
                "judge_raw":         raw[:2000],
                "scored_at":         datetime.now(timezone.utc).isoformat(),
            }

            async with lock:
                df_row = pd.DataFrame([out])
                write_header = not output_csv.exists()
                df_row.to_csv(output_csv, mode="a", header=write_header,
                              index=False, encoding="utf-8-sig")

            return out

    await asyncio.gather(*[judge_one(k, a, b) for k, a, b in work_pairs])

    if output_csv.exists():
        return pd.read_csv(output_csv, encoding="utf-8-sig")
    return pd.DataFrame()
