from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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

from src.common.ravdess import LABEL_ORDER
from src.datasets.early_fusion_dataset import EarlyFusionDataset, load_manifest_examples
from src.eval.llm_response import generate_responses
from src.preprocess.audio_ecapa import build_ecapa_layout
from src.train.run_early_fusion import (
    build_model,
    collect_probabilities,
    load_run_config,
    write_prediction_jsonl,
)
from scripts.run_custom_av_inference import (
    collect_custom_samples,
    run_ecapa_inference,
    run_video_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure full pipeline wall-clock time from raw A/V(+PPG) to final LLM replies."
    )
    parser.add_argument(
        "--systems",
        default="early,late",
        help="Comma-separated systems to measure: early, late.",
    )
    parser.add_argument(
        "--labels-csv",
        default="dataset_200_labeled.csv",
        help="Path to dataset_200_labeled.csv.",
    )
    parser.add_argument(
        "--ppg-root",
        default="data/raw/custom/ppg",
        help="Directory containing ppg_P_NN.csv files.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/runtime_measurements",
        help="Directory for temporary predictions, responses, and timing summaries.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4",
        help="LLM used to generate final replies.",
    )
    parser.add_argument(
        "--prompt-strategy",
        default="empathy_then_help",
        choices=("empathy_then_help", "text_only_baseline"),
        help="Prompt strategy used for the final LLM reply stage.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--early-config",
        default="configs/early_fusion/early_fusion_3cnn_gated_custom.yaml",
        help="Config path for the early-fusion run.",
    )
    parser.add_argument(
        "--early-checkpoint",
        default="checkpoints/early_fusion/early_fusion_3cnn_gated_custom/best_model.pt",
        help="Checkpoint path for the early-fusion run.",
    )
    parser.add_argument(
        "--use-early-cache",
        action="store_true",
        help="If set, allow the early-fusion dataset to read/write cached features.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device string for early/video inference, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_print(message: str) -> None:
    text = str(message)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def _load_weighted_prob_avg_weights(repo_root: Path) -> tuple[float, float]:
    path = (
        repo_root
        / "checkpoints/late_fusion/ecapa_video_reuse_baseline/weighted_prob_avg/fitted_weight.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["audio_weight"]), float(payload["video_weight"])


def _clear_existing_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _system_output_dir(output_root: Path, run_id: str, system: str) -> Path:
    return output_root / run_id / system


def measure_early_pipeline(
    *,
    repo_root: Path,
    labels_csv: Path,
    ppg_root: Path,
    output_root: Path,
    run_id: str,
    llm_model: str,
    prompt_strategy: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    concurrency: int,
    config_path: Path,
    checkpoint_path: Path,
    use_cache: bool,
    device_override: str | None,
) -> dict[str, Any]:
    system_dir = _system_output_dir(output_root, run_id, "early")
    predictions_path = system_dir / "predictions_test.jsonl"
    responses_path = system_dir / "responses_test.jsonl"
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()
    if responses_path.exists():
        responses_path.unlink()

    stage_start = time.perf_counter()
    run_config = load_run_config(config_path, repo_root=repo_root)
    run_config = replace(run_config, data=replace(run_config.data, enable_cache=use_cache))
    examples = load_manifest_examples(
        repo_root / run_config.data.test_manifest,
        repo_root=repo_root,
        validate_av_key=run_config.data.validate_ravdess_av_key,
    )
    dataset = EarlyFusionDataset(repo_root=repo_root, examples=examples, config=run_config.data)
    device = torch.device(device_override or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dataloader = DataLoader(
        dataset,
        batch_size=run_config.train.batch_size,
        shuffle=False,
        num_workers=run_config.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(run_config.model_name, run_config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    probabilities, labels, metadata_rows, _ = collect_probabilities(model, dataloader, device=device)
    write_prediction_jsonl(
        predictions_path,
        metadata_rows=metadata_rows,
        probabilities=probabilities,
        true_labels=labels,
    )
    prediction_stage_sec = time.perf_counter() - stage_start

    llm_start = time.perf_counter()
    generate_responses(
        predictions_jsonl=predictions_path,
        labels_csv=labels_csv,
        ppg_root=ppg_root,
        output_jsonl=responses_path,
        model=llm_model,
        prompt_strategies=[prompt_strategy],
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
        max_calls=None,
        ppg_cache_dir=None,
        fusion_method="early_fusion",
        reasoning_effort=reasoning_effort,
    )
    llm_stage_sec = time.perf_counter() - llm_start

    num_rows = len([line for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return {
        "system": "early",
        "num_samples": num_rows,
        "prediction_stage_sec": prediction_stage_sec,
        "llm_stage_sec": llm_stage_sec,
        "total_end_to_end_sec": prediction_stage_sec + llm_stage_sec,
        "avg_sec_per_sample": (prediction_stage_sec + llm_stage_sec) / max(num_rows, 1),
        "predictions_path": predictions_path.as_posix(),
        "responses_path": responses_path.as_posix(),
        "cache_enabled": use_cache,
        "device": str(device),
    }


def _write_late_predictions_jsonl(
    path: Path,
    *,
    samples: list[dict[str, Any]],
    ecapa_results: dict[str, Any],
    video_results: dict[str, Any],
    audio_weight: float,
    video_weight: float,
) -> int:
    audio_by_sample = {row["sample_id"]: row for row in ecapa_results["predictions"]["test"]}
    video_by_sample = video_results["predictions"]["test"]
    rows_written = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in sorted(samples, key=lambda item: item["av_key"]):
            audio_pred = audio_by_sample[sample["audio_stem"]]
            video_pred = video_by_sample[sample["video_stem"]]
            audio_probs = np.asarray(audio_pred["probs"], dtype=np.float64)
            video_probs = np.asarray(video_pred["probs"], dtype=np.float64)
            fused_probs = audio_weight * audio_probs + video_weight * video_probs
            pred_id = int(np.argmax(fused_probs))
            record = {
                "mode": "weighted_prob_avg",
                "av_key": sample["av_key"],
                "split": sample["split"],
                "actor_id": sample["actor_id"],
                "label_id": sample["label_id"],
                "label_name": sample["label_name"],
                "audio_sample_id": sample["audio_stem"],
                "video_sample_id": sample["video_stem"],
                "pred_id": pred_id,
                "pred_name": LABEL_ORDER[pred_id],
                "probs": [float(value) for value in fused_probs.tolist()],
                "audio_weight": audio_weight,
                "video_weight": video_weight,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
    return rows_written


def measure_late_pipeline(
    *,
    repo_root: Path,
    labels_csv: Path,
    ppg_root: Path,
    output_root: Path,
    run_id: str,
    llm_model: str,
    prompt_strategy: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    concurrency: int,
) -> dict[str, Any]:
    system_dir = _system_output_dir(output_root, run_id, "late")
    predictions_path = system_dir / "predictions_test.jsonl"
    responses_path = system_dir / "responses_test.jsonl"
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()
    if responses_path.exists():
        responses_path.unlink()

    stage_start = time.perf_counter()
    samples = collect_custom_samples(repo_root, labels_csv=labels_csv)
    test_samples = [sample for sample in samples if sample["split"] == "test"]
    ecapa_layout = build_ecapa_layout(repo_root)
    sb_root = str(ecapa_layout.audio_branch_root)
    if sb_root not in sys.path:
        sys.path.insert(0, sb_root)
    _clear_existing_files(
        [
            repo_root / "data/processed/audio/ecapa_metadata_custom/train.json",
            repo_root / "data/processed/audio/ecapa_metadata_custom/valid.json",
            repo_root / "data/processed/audio/ecapa_metadata_custom/test.json",
        ]
    )
    ecapa_results = run_ecapa_inference(test_samples, repo_root)
    video_results = run_video_inference(test_samples, repo_root)
    audio_weight, video_weight = _load_weighted_prob_avg_weights(repo_root)
    num_rows = _write_late_predictions_jsonl(
        predictions_path,
        samples=test_samples,
        ecapa_results=ecapa_results,
        video_results=video_results,
        audio_weight=audio_weight,
        video_weight=video_weight,
    )
    prediction_stage_sec = time.perf_counter() - stage_start

    llm_start = time.perf_counter()
    generate_responses(
        predictions_jsonl=predictions_path,
        labels_csv=labels_csv,
        ppg_root=ppg_root,
        output_jsonl=responses_path,
        model=llm_model,
        prompt_strategies=[prompt_strategy],
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
        max_calls=None,
        ppg_cache_dir=None,
        fusion_method="late_fusion",
        reasoning_effort=reasoning_effort,
    )
    llm_stage_sec = time.perf_counter() - llm_start

    return {
        "system": "late",
        "num_samples": num_rows,
        "prediction_stage_sec": prediction_stage_sec,
        "llm_stage_sec": llm_stage_sec,
        "total_end_to_end_sec": prediction_stage_sec + llm_stage_sec,
        "avg_sec_per_sample": (prediction_stage_sec + llm_stage_sec) / max(num_rows, 1),
        "predictions_path": predictions_path.as_posix(),
        "responses_path": responses_path.as_posix(),
        "audio_weight": audio_weight,
        "video_weight": video_weight,
    }


def main() -> int:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    labels_csv = (REPO_ROOT / args.labels_csv).resolve()
    ppg_root = (REPO_ROOT / args.ppg_root).resolve()
    config_path = (REPO_ROOT / args.early_config).resolve()
    checkpoint_path = (REPO_ROOT / args.early_checkpoint).resolve()
    systems = [item.strip().lower() for item in args.systems.split(",") if item.strip()]
    run_id = _timestamp_slug()

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "settings": {
            "systems": systems,
            "labels_csv": labels_csv.as_posix(),
            "ppg_root": ppg_root.as_posix(),
            "llm_model": args.llm_model,
            "prompt_strategy": args.prompt_strategy,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "concurrency": args.concurrency,
            "early_config": config_path.as_posix(),
            "early_checkpoint": checkpoint_path.as_posix(),
            "early_cache_enabled": args.use_early_cache,
        },
        "systems": {},
    }

    for system in systems:
        if system == "early":
            result = measure_early_pipeline(
                repo_root=REPO_ROOT,
                labels_csv=labels_csv,
                ppg_root=ppg_root,
                output_root=output_root,
                run_id=run_id,
                llm_model=args.llm_model,
                prompt_strategy=args.prompt_strategy,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                concurrency=args.concurrency,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                use_cache=args.use_early_cache,
                device_override=args.device,
            )
        elif system == "late":
            result = measure_late_pipeline(
                repo_root=REPO_ROOT,
                labels_csv=labels_csv,
                ppg_root=ppg_root,
                output_root=output_root,
                run_id=run_id,
                llm_model=args.llm_model,
                prompt_strategy=args.prompt_strategy,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                concurrency=args.concurrency,
            )
        else:
            raise ValueError(f"Unsupported system '{system}'. Use early and/or late.")
        summary["systems"][system] = result

    summary_path = output_root / run_id / "summary.json"
    _write_json(summary_path, summary)
    _safe_print(f"summary: {summary_path}")
    for system, result in summary["systems"].items():
        _safe_print(
            f"{system}: prediction={result['prediction_stage_sec']:.3f}s "
            f"llm={result['llm_stage_sec']:.3f}s total={result['total_end_to_end_sec']:.3f}s "
            f"per_sample={result['avg_sec_per_sample']:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
