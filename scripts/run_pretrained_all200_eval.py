from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from scripts.run_custom_av_inference import collect_custom_samples, run_ecapa_inference, run_video_inference
from src.common.ravdess import LABEL_ORDER
from src.datasets.early_fusion_dataset import EarlyFusionDataset, load_manifest_examples
from src.eval import analyze_pairwise_weighted as weighted_analysis
from src.eval import run_pairwise as pairwise_judge
from src.eval.llm_response import generate_responses
from src.train.run_early_fusion import build_model, collect_probabilities, load_run_config, write_prediction_jsonl

PPG_FEATURE_KEYS = ["hr_mean_bpm", "rmssd_ms", "sdnn_ms", "rr_mean_ms", "pnn50"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 200-sample inference-only evaluation using pretrained RAVDESS early/late models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels-csv", default="dataset_200_labeled.csv")
    parser.add_argument("--ppg-root", default="data/raw/custom/ppg")
    parser.add_argument("--rubrics-path", default="RUBRICS.md")
    parser.add_argument("--output-root", default="outputs/pretrained_all200_eval")
    parser.add_argument(
        "--early-config",
        default="configs/early_fusion/early_fusion_3cnn_gated.yaml",
        help="Pretrained RAVDESS early-fusion config to load.",
    )
    parser.add_argument(
        "--early-checkpoint",
        default="checkpoints/early_fusion/early_fusion_3cnn_gated/best_model.pt",
        help="Pretrained RAVDESS early-fusion checkpoint.",
    )
    parser.add_argument("--response-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--response-concurrency", type=int, default=2)
    parser.add_argument("--judge-concurrency", type=int, default=3)
    parser.add_argument("--response-max-tokens", type=int, default=3000)
    parser.add_argument("--response-reasoning-effort", default="none")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda:0.")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-responses", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    parser.add_argument("--skip-weighted", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        return
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _jsonl_to_csv(path_jsonl: Path, path_csv: Path) -> int:
    rows = _load_jsonl_rows(path_jsonl)
    _write_csv(path_csv, rows)
    return len(rows)


def _responses_jsonl_to_csv(path_jsonl: Path, path_csv: Path) -> int:
    rows = _load_jsonl_rows(path_jsonl)
    deduped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("av_key", "")), str(row.get("prompt_strategy", "")))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue
        if str(row.get("response", "")).strip():
            deduped[key] = row
    final_rows = sorted(deduped.values(), key=lambda item: (str(item.get("av_key", "")), str(item.get("prompt_strategy", ""))))
    _write_csv(path_csv, final_rows)
    return len(final_rows)


def _build_all200_samples(labels_csv: Path) -> List[Dict[str, Any]]:
    samples = collect_custom_samples(REPO_ROOT, labels_csv=labels_csv)
    for sample in samples:
        sample["split"] = "test"
        sample["emotion_code"] = sample["label_name"]
    return sorted(samples, key=lambda item: str(item["av_key"]))


def _write_all200_manifest(samples: List[Dict[str, Any]], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        participant = int(sample["participant"])
        local_n = int(sample["local_n"])
        rows.append(
            {
                "av_key": sample["av_key"],
                "split": "test",
                "actor_id": sample["actor_id"],
                "label_id": sample["label_id"],
                "label_name": sample["label_name"],
                "emotion_code": sample["label_name"],
                "audio_sample_id": sample["audio_stem"],
                "audio_path": sample["audio_path"].as_posix(),
                "video_sample_id": sample["video_stem"],
                "video_path": sample["video_path"].as_posix(),
                "ppg_csv_path": (REPO_ROOT / "data/raw/custom/ppg" / f"ppg_{participant}_{local_n:02d}.csv").as_posix(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "av_key",
                "split",
                "actor_id",
                "label_id",
                "label_name",
                "emotion_code",
                "audio_sample_id",
                "audio_path",
                "video_sample_id",
                "video_path",
                "ppg_csv_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_early_inference(
    *,
    labels_csv: Path,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device_override: str | None,
) -> Dict[str, Any]:
    predictions_jsonl = output_dir / "predictions" / "early_pretrained_all200.jsonl"
    predictions_csv = output_dir / "predictions" / "early_pretrained_all200.csv"
    stage_start = time.perf_counter()

    run_config = load_run_config(config_path, repo_root=REPO_ROOT)
    run_config = replace(
        run_config,
        data=replace(
            run_config.data,
            test_manifest=manifest_path,
            enable_cache=False,
            validate_ravdess_av_key=False,
        ),
        train=replace(
            run_config.train,
            num_workers=0,
        ),
    )

    examples = load_manifest_examples(
        manifest_path,
        repo_root=REPO_ROOT,
        validate_av_key=False,
    )
    dataset = EarlyFusionDataset(repo_root=REPO_ROOT, examples=examples, config=run_config.data)
    device = torch.device(device_override or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dataloader = DataLoader(
        dataset,
        batch_size=run_config.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(run_config.model_name, run_config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    probabilities, labels, metadata_rows, _ = collect_probabilities(model, dataloader, device=device)

    write_prediction_jsonl(
        predictions_jsonl,
        metadata_rows=metadata_rows,
        probabilities=probabilities,
        true_labels=labels,
    )
    row_count = _jsonl_to_csv(predictions_jsonl, predictions_csv)

    return {
        "system": "early",
        "classification_fusion_time_sec": time.perf_counter() - stage_start,
        "num_samples": row_count,
        "predictions_jsonl": predictions_jsonl.as_posix(),
        "predictions_csv": predictions_csv.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "config_path": config_path.as_posix(),
        "device": str(device),
    }


def _load_weighted_prob_avg_weights() -> tuple[float, float]:
    path = REPO_ROOT / "checkpoints/late_fusion/ecapa_video_reuse_baseline/weighted_prob_avg/fitted_weight.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["audio_weight"]), float(payload["video_weight"])


def _write_late_predictions_jsonl(
    path: Path,
    *,
    samples: List[Dict[str, Any]],
    ecapa_results: Dict[str, Any],
    video_results: Dict[str, Any],
    audio_weight: float,
    video_weight: float,
) -> int:
    audio_by_sample = {row["sample_id"]: row for row in ecapa_results["predictions"]["test"]}
    video_by_sample = video_results["predictions"]["test"]
    rows_written = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            audio_pred = audio_by_sample[sample["audio_stem"]]
            video_pred = video_by_sample[sample["video_stem"]]
            audio_probs = pd.Series(audio_pred["probs"], dtype="float64").to_numpy()
            video_probs = pd.Series(video_pred["probs"], dtype="float64").to_numpy()
            fused_probs = audio_weight * audio_probs + video_weight * video_probs
            pred_id = int(fused_probs.argmax())
            record = {
                "mode": "weighted_prob_avg",
                "av_key": sample["av_key"],
                "split": "test",
                "actor_id": sample["actor_id"],
                "label_id": sample["label_id"],
                "label_name": sample["label_name"],
                "audio_sample_id": sample["audio_stem"],
                "video_sample_id": sample["video_stem"],
                "audio_path": sample["audio_path"].as_posix(),
                "video_path": sample["video_path"].as_posix(),
                "pred_id": pred_id,
                "pred_name": LABEL_ORDER[pred_id],
                "audio_weight": audio_weight,
                "video_weight": video_weight,
            }
            for label_name, value in zip(LABEL_ORDER, fused_probs.tolist(), strict=True):
                record[f"prob_{label_name}"] = float(value)
            handle.write(json.dumps(record) + "\n")
            rows_written += 1
    return rows_written


def run_late_inference(
    *,
    labels_csv: Path,
    samples: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    predictions_jsonl = output_dir / "predictions" / "late_pretrained_all200.jsonl"
    predictions_csv = output_dir / "predictions" / "late_pretrained_all200.csv"
    stage_start = time.perf_counter()

    ecapa_results = run_ecapa_inference(
        samples,
        REPO_ROOT,
        metadata_root=output_dir / "late_artifacts" / "audio_metadata",
        predictions_root=output_dir / "late_artifacts" / "audio_predictions",
        embeddings_root=output_dir / "late_artifacts" / "audio_embeddings",
    )
    video_results = run_video_inference(
        samples,
        REPO_ROOT,
        sequence_root=output_dir / "late_artifacts" / "video_sequences",
        predictions_root=output_dir / "late_artifacts" / "video_predictions",
    )
    audio_weight, video_weight = _load_weighted_prob_avg_weights()
    row_count = _write_late_predictions_jsonl(
        predictions_jsonl,
        samples=samples,
        ecapa_results=ecapa_results,
        video_results=video_results,
        audio_weight=audio_weight,
        video_weight=video_weight,
    )
    _jsonl_to_csv(predictions_jsonl, predictions_csv)

    return {
        "system": "late",
        "classification_fusion_time_sec": time.perf_counter() - stage_start,
        "num_samples": row_count,
        "predictions_jsonl": predictions_jsonl.as_posix(),
        "predictions_csv": predictions_csv.as_posix(),
        "audio_weight": audio_weight,
        "video_weight": video_weight,
    }


def generate_response_set(
    *,
    predictions_path: Path,
    labels_csv: Path,
    ppg_root: Path,
    output_jsonl: Path,
    output_csv: Path,
    model: str,
    prompt_strategy: str,
    fusion_method: str,
    concurrency: int,
    max_tokens: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    llm_start = time.perf_counter()
    generate_responses(
        predictions_jsonl=predictions_path,
        labels_csv=labels_csv,
        ppg_root=ppg_root,
        output_jsonl=output_jsonl,
        model=model,
        prompt_strategies=[prompt_strategy],
        temperature=0.2,
        max_tokens=max_tokens,
        concurrency=concurrency,
        max_calls=None,
        ppg_cache_dir=None,
        fusion_method=fusion_method,
        reasoning_effort=reasoning_effort,
        ppg_prompt_feature_keys=PPG_FEATURE_KEYS,
        ppg_output_feature_keys=PPG_FEATURE_KEYS,
    )
    row_count = _responses_jsonl_to_csv(output_jsonl, output_csv)
    return {
        "rows": row_count,
        "llm_inference_time_sec": time.perf_counter() - llm_start,
        "responses_jsonl": output_jsonl.as_posix(),
        "responses_csv": output_csv.as_posix(),
    }


def run_pairwise_eval(
    *,
    output_root: Path,
    rubrics_path: Path,
    judge_model: str,
    concurrency: int,
    response_early: Path,
    response_late: Path,
    response_baseline: Path,
) -> Dict[str, Any]:
    pair_configs = [
        {
            "name": "early_vs_baseline",
            "file_a": str(response_early),
            "file_b": str(response_baseline),
            "label_a": "early fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
            "label_b": "text-only baseline",
            "signal_a": True,
            "signal_b": False,
        },
        {
            "name": "late_vs_baseline",
            "file_a": str(response_late),
            "file_b": str(response_baseline),
            "label_a": "late fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
            "label_b": "text-only baseline",
            "signal_a": True,
            "signal_b": False,
        },
        {
            "name": "early_vs_late",
            "file_a": str(response_early),
            "file_b": str(response_late),
            "label_a": "early fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
            "label_b": "late fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
            "signal_a": True,
            "signal_b": True,
        },
    ]
    pairwise_dir = output_root / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    rubrics_text = pairwise_judge._load_rubrics(rubrics_path)
    pairwise_judge._load_local_env(REPO_ROOT / ".env")
    started = time.perf_counter()
    all_results = asyncio.run(
        pairwise_judge._run_all_pairs_async(
            active_pairs=pair_configs,
            rubrics_text=rubrics_text,
            judge_model=judge_model,
            judge_provider="openai",
            concurrency=concurrency,
            output_dir=pairwise_dir,
        )
    )
    summary_df = pairwise_judge.build_summary(all_results, pair_configs)
    summary_path = pairwise_dir / "summary.csv"
    report_path = pairwise_dir / "report.txt"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pairwise_judge.write_report(summary_df, report_path)
    return {
        "pairwise_dir": pairwise_dir.as_posix(),
        "summary_csv": summary_path.as_posix(),
        "report_txt": report_path.as_posix(),
        "elapsed_sec": time.perf_counter() - started,
        "pair_configs": pair_configs,
    }


def run_weighted_eval(
    *,
    output_root: Path,
    pair_configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    input_dir = output_root / "pairwise"
    weighted_dir = output_root / "pairwise_weighted"
    weighted_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_results: Dict[str, pd.DataFrame] = {}
    for pair_config in pair_configs:
        all_results[pair_config["name"]] = weighted_analysis.annotate_pair_results(pair_config, input_dir, weighted_dir)
    summary_df = weighted_analysis.build_summary(all_results, pair_configs)
    summary_path = weighted_dir / "summary.csv"
    report_path = weighted_dir / "report.txt"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    weighted_analysis.write_report(summary_df, report_path)
    return {
        "weighted_dir": weighted_dir.as_posix(),
        "summary_csv": summary_path.as_posix(),
        "report_txt": report_path.as_posix(),
        "elapsed_sec": time.perf_counter() - started,
    }


def build_runtime_rows(early_result: Dict[str, Any], late_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in (early_result, late_result):
        total = float(result["classification_fusion_time_sec"]) + float(result["llm_inference_time_sec"])
        num_samples = int(result["num_samples"])
        rows.append(
            {
                "system": result["system"],
                "num_samples": num_samples,
                "classification_fusion_time_sec": round(float(result["classification_fusion_time_sec"]), 4),
                "llm_inference_time_sec": round(float(result["llm_inference_time_sec"]), 4),
                "total_end_to_end_time_sec": round(total, 4),
                "avg_time_per_sample_sec": round(total / max(num_samples, 1), 4),
            }
        )
    return rows


def _load_runtime_result(path: Path, system: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    for row in rows:
        if row.get("system") == system:
            return row
    raise FileNotFoundError(f"Missing runtime row for system={system} in {path}")


def main() -> int:
    args = parse_args()
    labels_csv = (REPO_ROOT / args.labels_csv).resolve()
    ppg_root = (REPO_ROOT / args.ppg_root).resolve()
    rubrics_path = (REPO_ROOT / args.rubrics_path).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    config_path = (REPO_ROOT / args.early_config).resolve()
    checkpoint_path = (REPO_ROOT / args.early_checkpoint).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = _build_all200_samples(labels_csv)
    manifest_path = output_root / "manifests" / "av_all200_testonly.csv"
    _write_all200_manifest(samples, manifest_path)
    runtime_csv = output_root / "runtime" / "runtime_summary.csv"
    runtime_json = output_root / "runtime" / "runtime_summary.json"

    if args.skip_inference:
        early_result = _load_runtime_result(runtime_json, "early")
        late_result = _load_runtime_result(runtime_json, "late")
        early_result.update(
            {
                "system": "early",
                "predictions_jsonl": (output_root / "predictions" / "early_pretrained_all200.jsonl").as_posix(),
                "predictions_csv": (output_root / "predictions" / "early_pretrained_all200.csv").as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "config_path": config_path.as_posix(),
            }
        )
        late_result.update(
            {
                "system": "late",
                "predictions_jsonl": (output_root / "predictions" / "late_pretrained_all200.jsonl").as_posix(),
                "predictions_csv": (output_root / "predictions" / "late_pretrained_all200.csv").as_posix(),
            }
        )
    else:
        early_result = run_early_inference(
            labels_csv=labels_csv,
            manifest_path=manifest_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_root,
            device_override=args.device,
        )
        late_result = run_late_inference(
            labels_csv=labels_csv,
            samples=samples,
            output_dir=output_root,
        )

    responses_dir = output_root / "llm_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_responses:
        early_result.update(
            {
                "responses_jsonl": (responses_dir / "responses_early_pretrained_all200.jsonl").as_posix(),
                "responses_csv": (responses_dir / "responses_early_pretrained_all200.csv").as_posix(),
            }
        )
        late_result.update(
            {
                "responses_jsonl": (responses_dir / "responses_late_pretrained_all200.jsonl").as_posix(),
                "responses_csv": (responses_dir / "responses_late_pretrained_all200.csv").as_posix(),
            }
        )
        baseline_result = {
            "responses_jsonl": (responses_dir / "responses_baseline_pretrained_all200.jsonl").as_posix(),
            "responses_csv": (responses_dir / "responses_baseline_pretrained_all200.csv").as_posix(),
        }
    else:
        early_response = generate_response_set(
            predictions_path=Path(early_result["predictions_jsonl"]),
            labels_csv=labels_csv,
            ppg_root=ppg_root,
            output_jsonl=responses_dir / "responses_early_pretrained_all200.jsonl",
            output_csv=responses_dir / "responses_early_pretrained_all200.csv",
            model=args.response_model,
            prompt_strategy="empathy_then_help",
            fusion_method="early_fusion_pretrained_all200",
            concurrency=args.response_concurrency,
            max_tokens=args.response_max_tokens,
            reasoning_effort=args.response_reasoning_effort,
        )
        early_result.update(early_response)

        late_response = generate_response_set(
            predictions_path=Path(late_result["predictions_jsonl"]),
            labels_csv=labels_csv,
            ppg_root=ppg_root,
            output_jsonl=responses_dir / "responses_late_pretrained_all200.jsonl",
            output_csv=responses_dir / "responses_late_pretrained_all200.csv",
            model=args.response_model,
            prompt_strategy="empathy_then_help",
            fusion_method="late_fusion_pretrained_all200",
            concurrency=args.response_concurrency,
            max_tokens=args.response_max_tokens,
            reasoning_effort=args.response_reasoning_effort,
        )
        late_result.update(late_response)

        baseline_result = generate_response_set(
            predictions_path=Path(early_result["predictions_jsonl"]),
            labels_csv=labels_csv,
            ppg_root=ppg_root,
            output_jsonl=responses_dir / "responses_baseline_pretrained_all200.jsonl",
            output_csv=responses_dir / "responses_baseline_pretrained_all200.csv",
            model=args.response_model,
            prompt_strategy="text_only_baseline",
            fusion_method="baseline_text_only_all200",
            concurrency=args.response_concurrency,
            max_tokens=args.response_max_tokens,
            reasoning_effort=args.response_reasoning_effort,
        )

        runtime_rows = build_runtime_rows(early_result, late_result)
        _write_csv(runtime_csv, runtime_rows)
        _write_json(runtime_json, {"generated_at": datetime.now(timezone.utc).isoformat(), "rows": runtime_rows})

    if args.skip_pairwise:
        pairwise_result = {
            "pairwise_dir": (output_root / "pairwise").as_posix(),
            "summary_csv": (output_root / "pairwise" / "summary.csv").as_posix(),
            "report_txt": (output_root / "pairwise" / "report.txt").as_posix(),
            "pair_configs": [
                {
                    "name": "early_vs_baseline",
                    "file_a": str(Path(early_result["responses_jsonl"])),
                    "file_b": str(Path(baseline_result["responses_jsonl"])),
                    "label_a": "early fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
                    "label_b": "text-only baseline",
                },
                {
                    "name": "late_vs_baseline",
                    "file_a": str(Path(late_result["responses_jsonl"])),
                    "file_b": str(Path(baseline_result["responses_jsonl"])),
                    "label_a": "late fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
                    "label_b": "text-only baseline",
                },
                {
                    "name": "early_vs_late",
                    "file_a": str(Path(early_result["responses_jsonl"])),
                    "file_b": str(Path(late_result["responses_jsonl"])),
                    "label_a": "early fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
                    "label_b": "late fusion (RAVDESS-pretrained, inference-only) + 5-feature PPG in prompt",
                },
            ],
        }
    else:
        pairwise_result = run_pairwise_eval(
            output_root=output_root,
            rubrics_path=rubrics_path,
            judge_model=args.judge_model,
            concurrency=args.judge_concurrency,
            response_early=Path(early_result["responses_jsonl"]),
            response_late=Path(late_result["responses_jsonl"]),
            response_baseline=Path(baseline_result["responses_jsonl"]),
        )

    if args.skip_weighted:
        weighted_result = {
            "weighted_dir": (output_root / "pairwise_weighted").as_posix(),
            "summary_csv": (output_root / "pairwise_weighted" / "summary.csv").as_posix(),
            "report_txt": (output_root / "pairwise_weighted" / "report.txt").as_posix(),
        }
    else:
        weighted_result = run_weighted_eval(
            output_root=output_root,
            pair_configs=pairwise_result["pair_configs"],
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels_csv": labels_csv.as_posix(),
        "ppg_root": ppg_root.as_posix(),
        "early": early_result,
        "late": late_result,
        "baseline": baseline_result,
        "runtime_csv": runtime_csv.as_posix(),
        "runtime_json": runtime_json.as_posix(),
        "pairwise": pairwise_result,
        "weighted": weighted_result,
    }
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    print(f"summary: {summary_path}")
    print(f"runtime_csv: {runtime_csv}")
    print(f"pairwise_report: {pairwise_result['report_txt']}")
    print(f"weighted_report: {weighted_result['report_txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
