from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_CLASSES: List[str] = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise"
]

PROB_KEYS: Dict[str, str] = {
    "neutral":  "prob_neutral",
    "calm":     "prob_calm",
    "happy":    "prob_happy",
    "sad":      "prob_sad",
    "angry":    "prob_angry",
    "fearful":  "prob_fearful",
    "disgust":  "prob_disgust",
    "surprise": "prob_surprise",
}

# ---------------------------------------------------------------------------
# PPG helpers
# ---------------------------------------------------------------------------

def _ppg_file_no(global_sample_no: int) -> int:
    """Convert global sample_no (1-200) to per-person file number (1-50)."""
    return ((global_sample_no - 1) % 50) + 1


def _ppg_path(ppg_root: Path, participant_id: int, file_no: int) -> Path:
    return ppg_root / f"ppg_{participant_id}_{file_no:02d}.csv"


def _ppg_cache_path(ppg_cache_dir: Optional[Path], participant_id: int, file_no: int) -> Optional[Path]:
    if ppg_cache_dir is None:
        return None
    return ppg_cache_dir / f"ppg_{participant_id}_{file_no:02d}.json"


def _compact_existing_output(output_jsonl: Path) -> List[Dict[str, Any]]:
    """Deduplicate existing JSONL rows by (av_key, prompt_strategy).

    Prefer non-empty responses over empty placeholders, and otherwise keep the
    most recent row encountered in the file.
    """
    if not output_jsonl.exists():
        return []

    ordered_keys: List[Tuple[str, str]] = []
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    original_rows = 0

    with open(output_jsonl, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            key = (str(row["av_key"]), str(row["prompt_strategy"]))
            original_rows += 1

            if key not in deduped:
                ordered_keys.append(key)
                deduped[key] = row
                continue

            existing = deduped[key]
            existing_has_text = bool(str(existing.get("response", "")).strip())
            new_has_text = bool(str(row.get("response", "")).strip())
            if new_has_text or not existing_has_text:
                deduped[key] = row

    compacted_rows = [deduped[key] for key in ordered_keys]
    if len(compacted_rows) != original_rows:
        with open(output_jsonl, "w", encoding="utf-8") as handle:
            for row in compacted_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "Compacted existing output %s from %d rows to %d unique rows.",
            output_jsonl,
            original_rows,
            len(compacted_rows),
        )

    return compacted_rows


def load_ppg_features(
    participant_id: int,
    file_no: int,
    ppg_root: Path,
    ppg_cache_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Load PPG features for a sample, using JSON cache when available."""
    cache_path = _ppg_cache_path(ppg_cache_dir, participant_id, file_no)

    # Try cache first
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Extract from raw CSV
    from src.preprocess.ppg_features import (
        PPG_FEATURE_NAMES_ALL,
        extract_ppg_features,
    )

    raw_path = _ppg_path(ppg_root, participant_id, file_no)
    if raw_path.exists():
        features = extract_ppg_features(raw_path)
    else:
        logger.warning("PPG file not found: %s — using zeros", raw_path)
        features = {name: 0.0 for name in PPG_FEATURE_NAMES_ALL}

    # Write cache
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(features, f)

    return features


# ---------------------------------------------------------------------------
# Label lookup
# ---------------------------------------------------------------------------

def load_labels(labels_csv: Path) -> Dict[Tuple[int, int], str]:
    """Return a dict mapping (participant_id, sample_no) → user_text."""
    import csv
    mapping: Dict[Tuple[int, int], str] = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["participant_id"])
            sno = int(row["sample_no"])
            mapping[(pid, sno)] = row["text"]
    return mapping


def get_user_text(
    av_key: str,
    labels: Dict[Tuple[int, int], str],
) -> Optional[str]:
    """Resolve av_key → user text using the labels dict.

    av_key encodes a global sample_no (e.g. "4_151" = participant 4, global sample 151).
    dataset_200_labeled.csv also uses global sample_no (participant 4 → 151-200),
    so the lookup key is (participant_id, global_sample_no) directly.
    No per-person conversion is needed here; that conversion only applies to PPG file paths.
    """
    parts = av_key.split("_", 1)
    if len(parts) != 2:
        return None
    participant_id = int(parts[0])
    global_sample_no = int(parts[1])
    return labels.get((participant_id, global_sample_no))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_text_only_baseline_prompt(
    user_text: str,
) -> List[Dict[str, str]]:
    """Text-only baseline: no emotion label, no PPG, no affect context block."""
    system = (
        "You are a helpful assistant. "
        "Reply to the user's message in a natural and helpful way. "
        "Write exactly one assistant reply."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_text},
    ]


def build_empathy_then_help_prompt(
    user_text: str,
    predicted_label: str,
    ppg: Dict[str, float],
    *,
    ppg_prompt_feature_keys: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Empathy-then-help: structured affect context block + static instruction footer."""
    system = (
        "You are a helpful, natural, and emotionally attuned assistant.\n"
        "Reply to the user's message in a way that first briefly acknowledges the user's "
        "likely emotional state, then provides practical and useful help.\n"
        "If affect context is provided, use it to guide how you acknowledge the feeling, "
        "how intense your wording should be, and what kind of help should come next.\n"
        "Keep the emotional acknowledgement short, natural, and proportionate to the "
        "user's likely state.\n"
        "After that, move clearly into helpful next steps, guidance, or an answer to "
        "the user's request.\n"
        "Do not over-dramatize, over-comfort, or sound scripted.\n"
        "Do not explicitly mention hidden metadata such as valence, arousal, or emotion labels.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    )

    prompt_keys = ppg_prompt_feature_keys or ["hr_mean_bpm", "rmssd_ms", "sdnn_ms"]
    label_map = {
        "hr_mean_bpm": ("HR mean", "bpm"),
        "rmssd_ms": ("RMSSD", "ms"),
        "sdnn_ms": ("SDNN", "ms"),
        "rr_mean_ms": ("RR mean", "ms"),
        "pnn50": ("pNN50", ""),
    }
    prompt_parts: List[str] = []
    for key in prompt_keys:
        label, unit = label_map.get(key, (key, ""))
        value = float(ppg.get(key, 0.0))
        if unit:
            prompt_parts.append(f"{label}={value:.1f}{unit}")
        else:
            prompt_parts.append(f"{label}={value:.3f}")
    physiological = ", ".join(prompt_parts) if prompt_parts else "unavailable"

    affect_block = (
        "[affect_context]\n"
        "mode: DISCRETE\n"
        "valence: N/A\n"
        "arousal: N/A\n"
        f"emotion_label: {predicted_label}\n"
        f"physiological: {physiological}"
    )

    instruction_block = (
        "[instruction]\n"
        "Write one natural assistant reply to the user.\n"
        "Be emotionally appropriate and helpful.\n"
        "Do not mention the metadata explicitly.\n"
        "Avoid sounding templated or overly dramatic."
    )

    user_content = (
        f"{affect_block}\n\n"
        f"[user_message]\n{user_text}\n\n"
        f"{instruction_block}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# OpenAI async caller
# ---------------------------------------------------------------------------

class EmptyCompletionError(RuntimeError):
    """Raised when the API returns no visible assistant text."""

    def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


def _usage_detail(response: Any, *path: str) -> Optional[int]:
    """Safely read nested usage details from SDK response objects."""
    value: Any = response
    for key in path:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


async def _call_openai_async(
    client,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str = "none",
    max_retries: int = 5,
) -> str:
    """Call OpenAI chat completion with retries and empty-output detection."""
    delay = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            }
            if reasoning_effort and model.startswith("gpt-5"):
                request_kwargs["reasoning_effort"] = reasoning_effort

            response = await client.chat.completions.create(
                **request_kwargs,
            )
            choice = response.choices[0]
            message = choice.message
            content = (message.content or "").strip()
            refusal = (getattr(message, "refusal", None) or "").strip()
            finish_reason = (choice.finish_reason or "").strip()

            if content:
                return content

            if refusal:
                raise EmptyCompletionError(
                    "Model returned a refusal instead of assistant text.",
                    metadata={
                        "finish_reason": finish_reason or None,
                        "refusal": refusal,
                    },
                )

            raise EmptyCompletionError(
                "Model returned empty assistant content.",
                metadata={
                    "finish_reason": finish_reason or None,
                    "completion_tokens": _usage_detail(response, "usage", "completion_tokens"),
                    "reasoning_tokens": _usage_detail(
                        response, "usage", "completion_tokens_details", "reasoning_tokens"
                    ),
                },
            )
        except EmptyCompletionError as e:
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                logger.warning(
                    "Empty completion (attempt %d/%d): %s metadata=%s - retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    e.metadata,
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = e
            else:
                raise
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate" in err_str or "429" in err_str or "quota" in err_str
            is_transient  = "timeout" in err_str or "500" in err_str or "503" in err_str

            if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                logger.warning(
                    "OpenAI API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries, e, wait,
                )
                await asyncio.sleep(wait)
                last_error = e
            else:
                raise

    raise RuntimeError(f"Max retries exceeded. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_responses(
    predictions_jsonl: Path,
    labels_csv: Path,
    ppg_root: Path,
    output_jsonl: Path,
    model: str = "gpt-5.4",
    prompt_strategies: List[str] = ("empathy_then_help",),
    temperature: float = 0.2,
    max_tokens: int = 1200,
    concurrency: int = 5,
    max_calls: Optional[int] = None,
    ppg_cache_dir: Optional[Path] = None,
    fusion_method: str = "early_fusion",
    reasoning_effort: str = "minimal",
    ppg_prompt_feature_keys: Optional[List[str]] = None,
    ppg_output_feature_keys: Optional[List[str]] = None,
) -> None:
    """Generate LLM responses for all samples × prompt strategies.

    Args:
        predictions_jsonl: Path to early fusion prediction JSONL.
        labels_csv:        Path to dataset_200_labeled.csv.
        ppg_root:          Directory containing ppg_P_NN.csv files.
        output_jsonl:      Output file path.
        model:             OpenAI model name.
        prompt_strategies: List of strategies to run.
        temperature:       Sampling temperature.
        max_tokens:        Max completion tokens per response, including reasoning tokens.
        concurrency:       Max concurrent API calls.
        max_calls:         Budget cap (None = unlimited).
        ppg_cache_dir:     Directory to cache extracted PPG features as JSON.
        fusion_method:     Label string written into the output rows.
        reasoning_effort:  GPT-5 reasoning effort level.
    """
    asyncio.run(
        _generate_async(
            predictions_jsonl=predictions_jsonl,
            labels_csv=labels_csv,
            ppg_root=ppg_root,
            output_jsonl=output_jsonl,
            model=model,
            prompt_strategies=list(prompt_strategies),
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency=concurrency,
            max_calls=max_calls,
            ppg_cache_dir=ppg_cache_dir,
            fusion_method=fusion_method,
            reasoning_effort=reasoning_effort,
            ppg_prompt_feature_keys=ppg_prompt_feature_keys,
            ppg_output_feature_keys=ppg_output_feature_keys,
        )
    )


async def _generate_async(
    predictions_jsonl: Path,
    labels_csv: Path,
    ppg_root: Path,
    output_jsonl: Path,
    model: str,
    prompt_strategies: List[str],
    temperature: float,
    max_tokens: int,
    concurrency: int,
    max_calls: Optional[int],
    ppg_cache_dir: Optional[Path],
    fusion_method: str,
    reasoning_effort: str,
    ppg_prompt_feature_keys: Optional[List[str]],
    ppg_output_feature_keys: Optional[List[str]],
) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError("openai package not installed. Run: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = AsyncOpenAI(api_key=api_key)

    # Load labels
    labels = load_labels(labels_csv)
    logger.info("Loaded %d label rows from %s", len(labels), labels_csv)

    # Load predictions
    predictions: List[Dict] = []
    with open(predictions_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    logger.info("Loaded %d prediction rows", len(predictions))

    # Resume: load already-written (av_key, prompt_strategy) pairs
    done_keys: Set[Tuple[str, str]] = set()
    empty_done_keys: Set[Tuple[str, str]] = set()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _compact_existing_output(output_jsonl)
    if existing_rows:
        for row in existing_rows:
            key = (row["av_key"], row["prompt_strategy"])
            if str(row.get("response", "")).strip():
                done_keys.add(key)
                empty_done_keys.discard(key)
            else:
                empty_done_keys.add(key)
        logger.info(
            "Resuming: %d completed entries already written, %d empty entries will be retried",
            len(done_keys),
            len(empty_done_keys - done_keys),
        )

    # Build work queue
    work_items = []
    for pred in predictions:
        av_key = pred["av_key"]
        user_text = get_user_text(av_key, labels)
        if user_text is None:
            logger.warning("No label found for av_key=%s — skipping", av_key)
            continue
        for strategy in prompt_strategies:
            if (av_key, strategy) not in done_keys:
                work_items.append((pred, user_text, strategy))

    total = len(work_items)
    if max_calls is not None:
        work_items = work_items[:max_calls]
        if len(work_items) < total:
            logger.info("Budget cap applied: running %d of %d items", len(work_items), total)

    logger.info("Generating %d responses (concurrency=%d)", len(work_items), concurrency)

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def process_one(pred: Dict, user_text: str, strategy: str) -> Optional[Dict]:
        nonlocal completed
        async with sem:
            av_key = pred["av_key"]

            # predicted_label: late fusion uses "pred_name"; early fusion also has it
            predicted_label = pred.get("pred_name", "neutral")

            # Probability distribution: late fusion stores a "probs" list [p0..p7];
            # early fusion stores individual "prob_<label>" fields.
            if isinstance(pred.get("probs"), list):
                prob_dist = dict(zip(EMOTION_CLASSES, pred["probs"]))
            else:
                prob_dist = {em: pred.get(PROB_KEYS[em], 0.0) for em in EMOTION_CLASSES}

            confidence = prob_dist.get(predicted_label, 0.0)

            # PPG features — convert global sample_no → per-person file number
            parts = av_key.split("_", 1)
            participant_id = int(parts[0])
            global_sample_no = int(parts[1])
            per_person_no = ((global_sample_no - 1) % 50) + 1  # e.g. 151 → 1

            ppg = load_ppg_features(participant_id, per_person_no, ppg_root, ppg_cache_dir)

            # Build prompt
            if strategy == "empathy_then_help":
                messages = build_empathy_then_help_prompt(
                    user_text,
                    predicted_label,
                    ppg,
                    ppg_prompt_feature_keys=ppg_prompt_feature_keys,
                )
            elif strategy == "text_only_baseline":
                messages = build_text_only_baseline_prompt(user_text)
            else:
                raise ValueError(f"Unknown prompt strategy: {strategy}")

            try:
                response_text = await _call_openai_async(
                    client, messages, model, temperature, max_tokens, reasoning_effort
                )
            except Exception as e:
                logger.error("API error for av_key=%s strategy=%s: %s", av_key, strategy, e)
                return None

            # Build output row
            output_keys = ppg_output_feature_keys or [
                "hr_mean_bpm",
                "hr_std_bpm",
                "rmssd_ms",
                "sdnn_ms",
                "rr_mean_ms",
                "pnn50",
                "lf_hf_ratio",
            ]
            ppg_summary = {
                k: round(v, 3) for k, v in ppg.items()
                if k in output_keys and v != 0.0
            }

            row = {
                "av_key":               av_key,
                "participant_id":       participant_id,
                "sample_no":            per_person_no,
                "user_text":            user_text,
                "true_label":           pred.get("label_name"),
                "predicted_label":      predicted_label,
                "prediction_confidence": round(confidence, 4),
                "prob_distribution":    {k: round(v, 4) for k, v in prob_dist.items()},
                "ppg_features":         ppg_summary,
                "prompt_strategy":      strategy,
                "fusion_method":        fusion_method,
                "llm_model":            model,
                "response":             response_text,
            }

            async with lock:
                with open(output_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 10 == 0:
                    logger.info("Progress: %d/%d", completed, len(work_items))

            return row

    tasks = [process_one(pred, text, strat) for pred, text, strat in work_items]
    await asyncio.gather(*tasks)
    logger.info("Done. Wrote %d responses to %s", completed, output_jsonl)


# ---------------------------------------------------------------------------
# Convenience: load a responses JSONL
# ---------------------------------------------------------------------------

def load_responses(jsonl_path: Path) -> List[Dict]:
    """Load all rows from a responses JSONL file."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
