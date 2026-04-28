"""Run bidirectional pairwise judging across the three response files.

Usage:
    python -m src.eval.run_pairwise --judge-model gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import inspect
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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

DIMENSION_KEYS = ["D1", "D2", "D3", "D4", "D5", "D6"]
QUALITY_DIMS = ["D1", "D2", "D3", "D4", "D5"]
OVERALL_DIMENSION = "OVERALL_D1_D5"

PAIR_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "early_vs_baseline",
        "file_a": "outputs/llm_responses/responses_early_noppg_fusion.jsonl",
        "file_b": "outputs/llm_responses/responses_baseline.jsonl",
        "label_a": "early fusion (no PPG in classifier) + PPG in prompt",
        "label_b": "text-only baseline",
        "signal_a": True,
        "signal_b": False,
    },
    {
        "name": "late_vs_baseline",
        "file_a": "outputs/llm_responses/responses_late_noppg_fusion.jsonl",
        "file_b": "outputs/llm_responses/responses_baseline.jsonl",
        "label_a": "late fusion (no PPG in classifier) + PPG in prompt",
        "label_b": "text-only baseline",
        "signal_a": True,
        "signal_b": False,
    },
    {
        "name": "early_vs_late",
        "file_a": "outputs/llm_responses/responses_early_noppg_fusion.jsonl",
        "file_b": "outputs/llm_responses/responses_late_noppg_fusion.jsonl",
        "label_a": "early fusion (no PPG in classifier) + PPG in prompt",
        "label_b": "late fusion (no PPG in classifier) + PPG in prompt",
        "signal_a": True,
        "signal_b": True,
    },
]

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "A": {
            "type": "object",
            "properties": {
                **{dimension: {"type": "integer", "minimum": 1, "maximum": 5} for dimension in DIMENSION_KEYS},
                "rationale": {
                    "type": "object",
                    "properties": {dimension: {"type": "string"} for dimension in DIMENSION_KEYS},
                    "required": DIMENSION_KEYS,
                },
            },
            "required": [*DIMENSION_KEYS, "rationale"],
        },
        "B": {
            "type": "object",
            "properties": {
                **{dimension: {"type": "integer", "minimum": 1, "maximum": 5} for dimension in DIMENSION_KEYS},
                "rationale": {
                    "type": "object",
                    "properties": {dimension: {"type": "string"} for dimension in DIMENSION_KEYS},
                    "required": DIMENSION_KEYS,
                },
            },
            "required": [*DIMENSION_KEYS, "rationale"],
        },
    },
    "required": ["A", "B"],
}


def _load_local_env(env_path: Path) -> None:
    """Populate os.environ from .env without overwriting existing variables."""
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        pass

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip("'").strip('"')

        if key and key not in os.environ:
            os.environ[key] = value


def _load_rubrics(rubrics_path: Path) -> str:
    if not rubrics_path.is_file():
        raise FileNotFoundError(f"Rubrics file not found: {rubrics_path}")
    return rubrics_path.read_text(encoding="utf-8").strip()


def _load_jsonl_by_key(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            av_key = str(row["av_key"])
            rows[av_key] = row
    return rows


def _stem_to_av_key(stem: str) -> Tuple[str, int, int]:
    match = re.fullmatch(r"(\d+)_(\d{1,2})", stem.strip())
    if not match:
        raise ValueError(f"Unsupported stem format: {stem!r}. Expected e.g. '4_01'.")
    participant_id = int(match.group(1))
    local_sample_no = int(match.group(2))
    global_sample_no = ((participant_id - 1) * 50) + local_sample_no
    return f"{participant_id}_{global_sample_no}", participant_id, local_sample_no


def _load_feature_csv_by_key(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stem", "text", "reply"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Feature CSV is missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            stem = str(row.get("stem") or "").strip()
            if not stem:
                logger.warning("Skipping empty stem at %s:%d", path, line_no)
                continue
            av_key, participant_id, local_sample_no = _stem_to_av_key(stem)
            rows[av_key] = {
                "av_key": av_key,
                "participant_id": participant_id,
                "sample_no": local_sample_no,
                "user_text": str(row.get("text") or ""),
                "response": str(row.get("reply") or ""),
                "prompt_strategy": "external_csv",
                "fusion_method": "external_csv",
                "source_stem": stem,
                "face_available": row.get("face_available"),
                "voice_available": row.get("voice_available"),
                "hr_available": row.get("hr_available"),
            }
    return rows


def _load_responses_by_key(path: Path) -> Dict[str, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_by_key(path)
    if suffix == ".csv":
        return _load_feature_csv_by_key(path)
    raise ValueError(f"Unsupported response file format for {path}. Expected .jsonl or .csv")


def _load_existing_pair_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        logger.warning("Existing results file is empty, treating as no prior progress: %s", path)
        return pd.DataFrame()


def _extract_ppg_summary(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Dict[str, float]:
    ppg = row_a.get("ppg_features") or row_b.get("ppg_features") or {}
    return {str(k): float(v) for k, v in ppg.items()}


def _parse_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError(f"Could not parse judge JSON:\n{stripped[:2000]}")


def _coerce_score(value: Any) -> int:
    try:
        return max(1, min(5, int(value)))
    except Exception:
        return 3


def _coerce_dimension_scores(payload: Dict[str, Any]) -> Dict[str, int]:
    return {dimension: _coerce_score(payload.get(dimension, 3)) for dimension in DIMENSION_KEYS}


def _overall_0_to_100(scores: Dict[str, float]) -> float:
    quality_scores = [((scores[dimension] - 1.0) / 4.0) * 100.0 for dimension in QUALITY_DIMS]
    return round(sum(quality_scores) / len(quality_scores), 2)


def _winner(score_a: float, score_b: float, tie_margin: float = 0.0) -> str:
    if abs(score_a - score_b) <= tie_margin:
        return "tie"
    return "A" if score_a > score_b else "B"


def _build_judge_messages(
    *,
    rubrics_text: str,
    user_text: str,
    response_a: str,
    response_b: str,
    label_a: str,
    label_b: str,
    signal_a: bool,
    signal_b: bool,
    predicted_label: str,
    confidence: float,
    ppg_summary: Dict[str, float],
) -> List[Dict[str, str]]:
    system = (
        "You are a strict pairwise evaluator for AI assistant responses.\n"
        "Return JSON only. Do not include markdown. Do not include extra keys.\n"
        "Do not reward longer responses just for being longer.\n"
        "Do not reward explicit emotion words by themselves.\n"
        "Do not punish concise responses if they are better calibrated.\n"
        "Score Response A and Response B independently on D1-D6.\n"
        "The output must be strict JSON that can be parsed by a standard JSON parser.\n"
        "Inside rationale strings, do not use unescaped double-quote characters. Prefer single quotes instead."
    )

    ppg_str = ", ".join(f"{key}={value}" for key, value in sorted(ppg_summary.items())) or "unavailable"
    schema = (
        '{'
        '"A":{"D1":1,"D2":1,"D3":1,"D4":1,"D5":1,"D6":1,'
        '"rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}},'
        '"B":{"D1":1,"D2":1,"D3":1,"D4":1,"D5":1,"D6":1,'
        '"rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}}'
        '}'
    )

    user = (
        "Use the following rubric text exactly as the scoring basis.\n\n"
        f"{rubrics_text}\n\n"
        "Additional judge-side rules:\n"
        "- Score D1-D5 as quality dimensions.\n"
        "- Score D6 as a separate mechanism dimension.\n"
        "- Do not output winners. Output raw scores only.\n"
        "- If a system did not receive emotion context, D6 should not exceed 3.\n"
        "- Use ties implicitly by giving the same score when the difference is small.\n\n"
        "Shared context:\n"
        f"- predicted_emotion: {predicted_label}\n"
        f"- confidence: {confidence:.4f}\n"
        f"- ppg_features: {ppg_str}\n\n"
        "System A metadata:\n"
        f"- description: {label_a}\n"
        f"- received_emotion_context: {str(signal_a).lower()}\n\n"
        "System B metadata:\n"
        f"- description: {label_b}\n"
        f"- received_emotion_context: {str(signal_b).lower()}\n\n"
        f"User message:\n{user_text}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Return JSON in this schema:\n{schema}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _call_openai(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 5,
) -> str:
    backoff_seconds = 1.0
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=900,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            error_text = str(exc).lower()
            retryable = any(
                token in error_text
                for token in ("rate", "429", "quota", "timeout", "500", "502", "503", "504")
            )
            if retryable and attempt < max_retries - 1:
                wait_seconds = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Judge call failed on attempt %d/%d: %s. Retrying in %.1fs.",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                continue
            raise
    raise RuntimeError("OpenAI call failed after retries.")


def _call_gemini_sync(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
) -> str:
    system_text = next((item["content"] for item in messages if item["role"] == "system"), "")
    conversation_parts: List[str] = []
    for item in messages:
        if item["role"] == "system":
            continue
        conversation_parts.append(f"[{item['role']}]\n{item['content']}")
    prompt = f"{system_text}\n\n" + "\n\n".join(conversation_parts)
    prompt = prompt.strip()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_json_schema": JUDGE_SCHEMA,
        },
    )
    return getattr(response, "text", "") or ""


async def _call_gemini(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 5,
) -> str:
    backoff_seconds = 1.0
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(_call_gemini_sync, client, messages, model)
        except Exception as exc:
            error_text = str(exc).lower()
            retryable = any(
                token in error_text
                for token in ("rate", "429", "quota", "timeout", "500", "502", "503", "504", "resource_exhausted")
            )
            if retryable and attempt < max_retries - 1:
                wait_seconds = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Gemini judge call failed on attempt %d/%d: %s. Retrying in %.1fs.",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                continue
            raise
    raise RuntimeError("Gemini call failed after retries.")


async def _call_anthropic(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 5,
) -> str:
    system_text = next((item["content"] for item in messages if item["role"] == "system"), "")
    anthropic_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in messages
        if item["role"] in {"user", "assistant"}
    ]
    backoff_seconds = 1.0

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                system=system_text,
                messages=anthropic_messages,
                temperature=0.0,
                max_tokens=1200,
            )
            parts: List[str] = []
            for block in getattr(response, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts).strip()
        except Exception as exc:
            error_text = str(exc).lower()
            retryable = any(
                token in error_text
                for token in ("rate", "429", "quota", "timeout", "500", "502", "503", "504", "529", "overloaded")
            )
            if retryable and attempt < max_retries - 1:
                wait_seconds = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Claude judge call failed on attempt %d/%d: %s. Retrying in %.1fs.",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                continue
            raise
    raise RuntimeError("Anthropic call failed after retries.")


async def _call_deepseek(
    client: Any,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 5,
) -> str:
    backoff_seconds = 1.0
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            error_text = str(exc).lower()
            retryable = any(
                token in error_text
                for token in ("rate", "429", "quota", "timeout", "500", "502", "503", "504")
            )
            if retryable and attempt < max_retries - 1:
                wait_seconds = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "DeepSeek judge call failed on attempt %d/%d: %s. Retrying in %.1fs.",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                continue
            raise
    raise RuntimeError("DeepSeek call failed after retries.")


async def _judge_one_direction(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    rubrics_text: str,
    judge_model: str,
    judge_provider: Literal["openai", "gemini", "anthropic", "deepseek"],
    user_text: str,
    response_a: str,
    response_b: str,
    label_a: str,
    label_b: str,
    signal_a: bool,
    signal_b: bool,
    predicted_label: str,
    confidence: float,
    ppg_summary: Dict[str, float],
) -> Tuple[Dict[str, int], Dict[str, int], str]:
    messages = _build_judge_messages(
        rubrics_text=rubrics_text,
        user_text=user_text,
        response_a=response_a,
        response_b=response_b,
        label_a=label_a,
        label_b=label_b,
        signal_a=signal_a,
        signal_b=signal_b,
        predicted_label=predicted_label,
        confidence=confidence,
        ppg_summary=ppg_summary,
    )
    current_messages = list(messages)
    last_raw = ""
    parse_attempts = 3

    for parse_attempt in range(parse_attempts):
        async with semaphore:
            if judge_provider == "openai":
                raw = await _call_openai(client, current_messages, judge_model)
            elif judge_provider == "gemini":
                raw = await _call_gemini(client, current_messages, judge_model)
            elif judge_provider == "anthropic":
                raw = await _call_anthropic(client, current_messages, judge_model)
            elif judge_provider == "deepseek":
                raw = await _call_deepseek(client, current_messages, judge_model)
            else:
                raise ValueError(f"Unsupported judge provider: {judge_provider}")

        last_raw = raw
        try:
            parsed = _parse_json(raw)
            scores_a = _coerce_dimension_scores(parsed.get("A", {}))
            scores_b = _coerce_dimension_scores(parsed.get("B", {}))
            return scores_a, scores_b, raw
        except Exception as exc:
            if parse_attempt >= parse_attempts - 1:
                raise
            logger.warning(
                "Judge returned invalid JSON on parse attempt %d/%d: %s. Retrying with stricter formatting instruction.",
                parse_attempt + 1,
                parse_attempts,
                exc,
            )
            current_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. Re-output the exact same judgment as strict valid JSON only. "
                        "Do not use markdown. Do not use code fences. Do not include any explanatory text before or after the JSON. "
                        "Do not place unescaped double quotes inside rationale strings; use single quotes instead."
                    ),
                },
            ]

    raise ValueError(f"Could not parse judge JSON after retries:\n{last_raw[:2000]}")


async def _judge_bidirectional(
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    rubrics_text: str,
    judge_model: str,
    judge_provider: Literal["openai", "gemini", "anthropic", "deepseek"],
    pair_config: Dict[str, Any],
    av_key: str,
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> Dict[str, Any]:
    user_text = str(row_a.get("user_text") or row_b.get("user_text") or "")
    predicted_label = str(row_a.get("predicted_label") or row_b.get("predicted_label") or "unknown")
    confidence = float(
        row_a.get("prediction_confidence")
        or row_b.get("prediction_confidence")
        or 0.0
    )
    ppg_summary = _extract_ppg_summary(row_a, row_b)

    dir1_a, dir1_b, raw_dir1 = await _judge_one_direction(
        client=client,
        semaphore=semaphore,
        rubrics_text=rubrics_text,
        judge_model=judge_model,
        judge_provider=judge_provider,
        user_text=user_text,
        response_a=str(row_a.get("response", "")),
        response_b=str(row_b.get("response", "")),
        label_a=str(pair_config["label_a"]),
        label_b=str(pair_config["label_b"]),
        signal_a=bool(pair_config["signal_a"]),
        signal_b=bool(pair_config["signal_b"]),
        predicted_label=predicted_label,
        confidence=confidence,
        ppg_summary=ppg_summary,
    )

    dir2_b, dir2_a, raw_dir2 = await _judge_one_direction(
        client=client,
        semaphore=semaphore,
        rubrics_text=rubrics_text,
        judge_model=judge_model,
        judge_provider=judge_provider,
        user_text=user_text,
        response_a=str(row_b.get("response", "")),
        response_b=str(row_a.get("response", "")),
        label_a=str(pair_config["label_b"]),
        label_b=str(pair_config["label_a"]),
        signal_a=bool(pair_config["signal_b"]),
        signal_b=bool(pair_config["signal_a"]),
        predicted_label=predicted_label,
        confidence=confidence,
        ppg_summary=ppg_summary,
    )

    avg_a = {
        dimension: round((dir1_a[dimension] + dir2_a[dimension]) / 2.0, 2)
        for dimension in DIMENSION_KEYS
    }
    avg_b = {
        dimension: round((dir1_b[dimension] + dir2_b[dimension]) / 2.0, 2)
        for dimension in DIMENSION_KEYS
    }
    overall_a = _overall_0_to_100(avg_a)
    overall_b = _overall_0_to_100(avg_b)

    result: Dict[str, Any] = {
        "pair": pair_config["name"],
        "av_key": av_key,
        "participant_id": row_a.get("participant_id") or row_b.get("participant_id"),
        "sample_no": row_a.get("sample_no") or row_b.get("sample_no"),
        "true_label": row_a.get("true_label") or row_b.get("true_label"),
        "predicted_label": predicted_label,
        "prediction_confidence": confidence,
        "label_a": pair_config["label_a"],
        "label_b": pair_config["label_b"],
        "file_a": pair_config["file_a"],
        "file_b": pair_config["file_b"],
        "judge_model": judge_model,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "overall_A": overall_a,
        "overall_B": overall_b,
        "winner_overall": _winner(overall_a, overall_b),
        "raw_dir1": raw_dir1[:2000],
        "raw_dir2": raw_dir2[:2000],
    }

    for dimension in DIMENSION_KEYS:
        result[f"avg_A_{dimension}"] = avg_a[dimension]
        result[f"avg_B_{dimension}"] = avg_b[dimension]
        result[f"winner_{dimension}"] = _winner(avg_a[dimension], avg_b[dimension])

    return result


async def _run_pair_async(
    *,
    pair_config: Dict[str, Any],
    rubrics_text: str,
    judge_model: str,
    judge_provider: Literal["openai", "gemini", "anthropic", "deepseek"],
    concurrency: int,
    output_dir: Path,
) -> pd.DataFrame:
    file_a = REPO_ROOT / pair_config["file_a"]
    file_b = REPO_ROOT / pair_config["file_b"]
    if not file_a.is_file():
        raise FileNotFoundError(f"Response file not found: {file_a}")
    if not file_b.is_file():
        raise FileNotFoundError(f"Response file not found: {file_b}")

    index_a = _load_responses_by_key(file_a)
    index_b = _load_responses_by_key(file_b)
    common_keys = sorted(set(index_a) & set(index_b))
    logger.info(
        "[%s] Matched %d responses by av_key (A=%d, B=%d).",
        pair_config["name"],
        len(common_keys),
        len(index_a),
        len(index_b),
    )

    if not common_keys:
        return pd.DataFrame()

    pair_dir = output_dir / pair_config["name"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    results_path = pair_dir / "results.csv"

    existing_df = _load_existing_pair_results(results_path)
    existing_keys = set()
    if not existing_df.empty and "av_key" in existing_df.columns:
        existing_df["av_key"] = existing_df["av_key"].astype(str)
        existing_keys = set(existing_df["av_key"])

    pending_keys = [av_key for av_key in common_keys if av_key not in existing_keys]
    logger.info(
        "[%s] Existing judged rows=%d, pending=%d.",
        pair_config["name"],
        len(existing_keys),
        len(pending_keys),
    )

    if not pending_keys:
        logger.info("[%s] Nothing to judge. Keeping existing %s", pair_config["name"], results_path)
        return existing_df

    client: Any
    if judge_provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package not installed. Run `pip install openai`.") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        client = AsyncOpenAI(api_key=api_key)
    elif judge_provider == "gemini":
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("google-genai package not installed. Run `pip install google-genai`.") from exc

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
    elif judge_provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError("anthropic package not installed. Run `pip install anthropic`.") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        client = AsyncAnthropic(api_key=api_key)
    elif judge_provider == "deepseek":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package not installed. Run `pip install openai`.") from exc

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY is not set.")
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported judge provider: {judge_provider}")

    semaphore = asyncio.Semaphore(concurrency)

    try:
        async def judge_key(av_key: str) -> Optional[Dict[str, Any]]:
            try:
                return await _judge_bidirectional(
                    client=client,
                    semaphore=semaphore,
                    rubrics_text=rubrics_text,
                    judge_model=judge_model,
                    judge_provider=judge_provider,
                    pair_config=pair_config,
                    av_key=av_key,
                    row_a=index_a[av_key],
                    row_b=index_b[av_key],
                )
            except Exception as exc:
                logger.error("[%s] Failed on av_key=%s: %s", pair_config["name"], av_key, exc)
                return None

        results = await asyncio.gather(*(judge_key(av_key) for av_key in pending_keys))
        rows = [row for row in results if row is not None]
        new_df = pd.DataFrame(rows)
        if existing_df.empty:
            df = new_df
        elif new_df.empty:
            df = existing_df
        else:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            if "av_key" in df.columns:
                df["av_key"] = df["av_key"].astype(str)
                df = df.drop_duplicates(subset=["av_key"], keep="last")
                df = df.sort_values("av_key").reset_index(drop=True)
        df.to_csv(results_path, index=False, encoding="utf-8-sig")
        logger.info("[%s] Wrote %d judged rows to %s", pair_config["name"], len(df), results_path)
        return df
    finally:
        await _close_async_client(client)


def run_pair(
    *,
    pair_config: Dict[str, Any],
    rubrics_text: str,
    judge_model: str,
    judge_provider: Literal["openai", "gemini", "anthropic", "deepseek"],
    concurrency: int,
    output_dir: Path,
) -> pd.DataFrame:
    return asyncio.run(
        _run_pair_async(
            pair_config=pair_config,
            rubrics_text=rubrics_text,
            judge_model=judge_model,
            judge_provider=judge_provider,
            concurrency=concurrency,
            output_dir=output_dir,
        )
    )


async def _close_async_client(client: Any) -> None:
    for method_name in ("aclose", "close"):
        closer = getattr(client, method_name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Failed to close async client cleanly: %s", exc)
        return


async def _run_all_pairs_async(
    *,
    active_pairs: List[Dict[str, Any]],
    rubrics_text: str,
    judge_model: str,
    judge_provider: Literal["openai", "gemini", "anthropic"],
    concurrency: int,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    all_results: Dict[str, pd.DataFrame] = {}
    for pair_config in active_pairs:
        logger.info("Starting pair: %s", pair_config["name"])
        df = await _run_pair_async(
            pair_config=pair_config,
            rubrics_text=rubrics_text,
            judge_model=judge_model,
            judge_provider=judge_provider,
            concurrency=concurrency,
            output_dir=output_dir,
        )
        all_results[pair_config["name"]] = df
    return all_results


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


def build_summary(all_results: Dict[str, pd.DataFrame], active_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for pair_config in active_pairs:
        df = all_results.get(pair_config["name"])
        if df is None or df.empty:
            continue

        for dimension in DIMENSION_KEYS:
            win_rates = _win_rates(df, f"winner_{dimension}")
            records.append(
                {
                    "pair": pair_config["name"],
                    "dimension": dimension,
                    "dimension_group": "quality" if dimension in QUALITY_DIMS else "mechanism",
                    "label_a": pair_config["label_a"],
                    "label_b": pair_config["label_b"],
                    "n": len(df),
                    "a_win_rate": win_rates["A"],
                    "b_win_rate": win_rates["B"],
                    "tie_rate": win_rates["tie"],
                    "avg_score_a": round(df[f"avg_A_{dimension}"].mean(), 3),
                    "avg_score_b": round(df[f"avg_B_{dimension}"].mean(), 3),
                }
            )

        overall_win_rates = _win_rates(df, "winner_overall")
        records.append(
            {
                "pair": pair_config["name"],
                "dimension": OVERALL_DIMENSION,
                "dimension_group": "overall",
                "label_a": pair_config["label_a"],
                "label_b": pair_config["label_b"],
                "n": len(df),
                "a_win_rate": overall_win_rates["A"],
                "b_win_rate": overall_win_rates["B"],
                "tie_rate": overall_win_rates["tie"],
                "avg_score_a": round(df["overall_A"].mean(), 3),
                "avg_score_b": round(df["overall_B"].mean(), 3),
            }
        )

    summary_df = pd.DataFrame(
        records,
        columns=[
            "pair",
            "dimension",
            "dimension_group",
            "label_a",
            "label_b",
            "n",
            "a_win_rate",
            "b_win_rate",
            "tie_rate",
            "avg_score_a",
            "avg_score_b",
        ],
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["pair", "dimension"], ignore_index=True)
    return summary_df


def write_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    dimension_names = {
        "D1": "Emotional Attunement & Calibration",
        "D2": "Emotion-to-Response Strategy Fit",
        "D3": "Task Usefulness & Correctness",
        "D4": "Clarity & Cognitive Load Fit",
        "D5": "Tone Safety, Respect & Risk Calibration",
        "D6": "Emotion Signal Utilization",
        OVERALL_DIMENSION: "Overall (D1-D5 only)",
    }

    lines: List[str] = []
    lines.append("PAIRWISE JUDGE SUMMARY")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    if summary_df.empty:
        lines.append("No judged rows were produced.")
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        logger.info("Wrote text report to %s", output_path)
        return

    for pair_name in summary_df["pair"].unique():
        pair_rows = summary_df[summary_df["pair"] == pair_name]
        if pair_rows.empty:
            continue

        first_row = pair_rows.iloc[0]
        lines.append(f"{pair_name}")
        lines.append(f"A: {first_row['label_a']}")
        lines.append(f"B: {first_row['label_b']}")
        lines.append(f"n: {int(first_row['n'])}")
        lines.append("")
        lines.append("dimension | a_win_rate | b_win_rate | tie_rate | avg_score_a | avg_score_b")

        display_order = QUALITY_DIMS + ["D6", OVERALL_DIMENSION]
        for dimension in display_order:
            row = pair_rows[pair_rows["dimension"] == dimension]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"{dimension_names[dimension]} | "
                f"{item['a_win_rate']:.1f}% | "
                f"{item['b_win_rate']:.1f}% | "
                f"{item['tie_rate']:.1f}% | "
                f"{item['avg_score_a']:.3f} | "
                f"{item['avg_score_b']:.3f}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Wrote text report to %s", output_path)


def _build_external_pair_configs(
    *,
    external_response_file: str,
    external_system_name: str,
    external_system_label: str,
    external_signal_context: bool,
    compare_early_file: str,
    compare_late_file: str,
    compare_early_label: str,
    compare_late_label: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "name": f"{external_system_name}_vs_early",
            "file_a": external_response_file,
            "file_b": compare_early_file,
            "label_a": external_system_label,
            "label_b": compare_early_label,
            "signal_a": external_signal_context,
            "signal_b": True,
        },
        {
            "name": f"{external_system_name}_vs_late",
            "file_a": external_response_file,
            "file_b": compare_late_file,
            "label_a": external_system_label,
            "label_b": compare_late_label,
            "signal_a": external_signal_context,
            "signal_b": True,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bidirectional pairwise LLM judging for the three response-file pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--judge-provider",
        choices=("openai", "gemini", "anthropic", "deepseek"),
        default="openai",
        help="LLM provider used for judging.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.4-mini",
        help="Judge model name, e.g. gpt-5.4-mini or gemini-2.5-pro.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent judge calls.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pairwise",
        help="Directory for pairwise results, summary CSV, and text report.",
    )
    parser.add_argument(
        "--rubrics-path",
        default="RUBRICS.md",
        help="Path to the rubric file, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional subset of pairs to run.",
    )
    parser.add_argument(
        "--external-response-file",
        default=None,
        help="Optional external system response file (.csv with stem/text/reply columns, or .jsonl with av_key/user_text/response).",
    )
    parser.add_argument(
        "--external-system-name",
        default="feature",
        help="Short slug used in pair names for the external system.",
    )
    parser.add_argument(
        "--external-system-label",
        default="feature-based external system",
        help="Readable label used in reports for the external system.",
    )
    parser.add_argument(
        "--external-signal-context",
        action="store_true",
        help="Mark the external system as having received emotion/physiology context for D6 judging.",
    )
    parser.add_argument(
        "--external-early-file",
        default="outputs/llm_responses/responses_early_noppg_fusion.jsonl",
        help="Response file used as the early-system comparison target for external comparisons.",
    )
    parser.add_argument(
        "--external-late-file",
        default="outputs/llm_responses/responses_late_noppg_fusion.jsonl",
        help="Response file used as the late-system comparison target for external comparisons.",
    )
    parser.add_argument(
        "--external-early-label",
        default="early fusion (no PPG in classifier) + PPG in prompt",
        help="Readable label for the early-system comparison target.",
    )
    parser.add_argument(
        "--external-late-label",
        default="late fusion (no PPG in classifier) + PPG in prompt",
        help="Readable label for the late-system comparison target.",
    )
    parser.add_argument(
        "--skip-default-pairs",
        action="store_true",
        help="If set, run only the external comparisons instead of the built-in three pairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_local_env(REPO_ROOT / ".env")

    rubrics_path = Path(args.rubrics_path)
    if not rubrics_path.is_absolute():
        rubrics_path = REPO_ROOT / rubrics_path
    rubrics_text = _load_rubrics(rubrics_path)

    available_pairs: List[Dict[str, Any]] = [] if args.skip_default_pairs else list(PAIR_CONFIGS)
    if args.external_response_file:
        external_path = Path(args.external_response_file)
        if not external_path.is_absolute():
            external_path = REPO_ROOT / external_path
        external_pairs = _build_external_pair_configs(
            external_response_file=str(external_path),
            external_system_name=args.external_system_name,
            external_system_label=args.external_system_label,
            external_signal_context=args.external_signal_context,
            compare_early_file=args.external_early_file,
            compare_late_file=args.external_late_file,
            compare_early_label=args.external_early_label,
            compare_late_label=args.external_late_label,
        )
        available_pairs.extend(external_pairs)

    if not available_pairs:
        raise ValueError("No pairs configured to run. Provide --external-response-file or keep default pairs enabled.")

    active_pairs = available_pairs
    if args.pairs:
        requested = set(args.pairs)
        active_pairs = [config for config in available_pairs if config["name"] in requested]
        unknown = requested - {config["name"] for config in available_pairs}
        if unknown:
            raise ValueError(f"Unknown pair name(s): {sorted(unknown)}")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running %d pair(s) with provider=%s judge_model=%s concurrency=%d",
        len(active_pairs),
        args.judge_provider,
        args.judge_model,
        args.concurrency,
    )

    started = time.perf_counter()
    all_results = asyncio.run(
        _run_all_pairs_async(
            active_pairs=active_pairs,
            rubrics_text=rubrics_text,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            concurrency=args.concurrency,
            output_dir=output_dir,
        )
    )

    summary_df = build_summary(all_results, active_pairs)
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.txt"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(summary_df, report_path)

    logger.info("Wrote summary CSV to %s", summary_path)
    logger.info("Finished in %.1fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
