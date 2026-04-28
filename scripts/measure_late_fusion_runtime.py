from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.ravdess import LABEL_ORDER
from src.fusion.metrics import compute_classification_metrics, write_confusion_matrix_csv


AUDIO_PROB_FIELDS = [f"prob_{label_name}" for label_name in LABEL_ORDER]
VIDEO_PROB_FIELDS = [f"prob_{label_name}" for label_name in LABEL_ORDER]

AV_JOIN_FIELDS = [
    "av_key", "split", "actor_id", "label_id", "label_name",
    "emotion_code", "intensity_code", "statement_code", "repetition_code",
    "audio_sample_id", "audio_wav_path", "audio_pred_id", "audio_pred_name",
    "audio_embedding_path", "audio_embedding_dim",
    "video_sample_id", "video_path", "video_pred_id", "video_pred_name",
    "video_sequence_path", "video_sequence_num_frames", "video_sequence_feature_dim",
    "video_sequence_source", "video_prediction_source",
] + [f"audio_prob_{ln}" for ln in LABEL_ORDER] + [f"video_prob_{ln}" for ln in LABEL_ORDER]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _join_audio_video(
    audio_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    *,
    split: str,
) -> list[dict[str, str]]:
    audio_index = {row["av_key"]: row for row in audio_rows}
    video_index = {row["av_key"]: row for row in video_rows}

    if set(audio_index) != set(video_index):
        missing_in_video = sorted(set(audio_index) - set(video_index))
        missing_in_audio = sorted(set(video_index) - set(audio_index))
        raise ValueError(
            f"AV join mismatch for split '{split}'. Missing in video={missing_in_video[:5]}, "
            f"missing in audio={missing_in_audio[:5]}."
        )

    joined_rows: list[dict[str, str]] = []
    for av_key in sorted(audio_index):
        audio_row = audio_index[av_key]
        video_row = video_index[av_key]
        row = {
            "av_key": av_key,
            "split": split,
            "actor_id": audio_row["actor_id"],
            "label_id": audio_row["label_id"],
            "label_name": audio_row["label_name"],
            "emotion_code": audio_row["emotion_code"],
            "intensity_code": audio_row["intensity_code"],
            "statement_code": audio_row["statement_code"],
            "repetition_code": audio_row["repetition_code"],
            "audio_sample_id": audio_row["sample_id"],
            "audio_wav_path": audio_row["wav_path"],
            "audio_pred_id": audio_row["pred_id"],
            "audio_pred_name": audio_row["pred_name"],
            "audio_embedding_path": audio_row["embedding_path"],
            "audio_embedding_dim": audio_row["embedding_dim"],
            "video_sample_id": video_row["sample_id"],
            "video_path": video_row["video_path"],
            "video_pred_id": video_row["pred_id"],
            "video_pred_name": video_row["pred_name"],
            "video_sequence_path": video_row["sequence_path"],
            "video_sequence_num_frames": video_row["sequence_num_frames"],
            "video_sequence_feature_dim": video_row["sequence_feature_dim"],
            "video_sequence_source": video_row["sequence_video_id_source"],
            "video_prediction_source": video_row["prediction_video_id_source"],
        }
        for field in AUDIO_PROB_FIELDS:
            row[f"audio_{field}"] = audio_row[field]
        for field in VIDEO_PROB_FIELDS:
            row[f"video_{field}"] = video_row[field]
        joined_rows.append(row)
    return joined_rows


def _load_prob_matrix(rows: list[dict[str, str]], *, prefix: str) -> np.ndarray:
    fields = [f"{prefix}_prob_{label_name}" for label_name in LABEL_ORDER]
    return np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=np.float32)


def _write_fusion_predictions(
    path: Path,
    rows: list[dict[str, str]],
    probs: np.ndarray,
    pred_ids: list[int],
    *,
    audio_weight: float,
    video_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, prob_vector, pred_id in zip(rows, probs.tolist(), pred_ids, strict=True):
            record = {
                "mode": "weighted_prob_avg",
                "av_key": row["av_key"],
                "split": row["split"],
                "actor_id": int(row["actor_id"]),
                "label_id": int(row["label_id"]),
                "label_name": row["label_name"],
                "audio_sample_id": row["audio_sample_id"],
                "video_sample_id": row["video_sample_id"],
                "pred_id": int(pred_id),
                "pred_name": LABEL_ORDER[int(pred_id)],
                "probs": [float(value) for value in prob_vector],
                "audio_weight": audio_weight,
                "video_weight": video_weight,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def labels_are_placeholder(labels: np.ndarray) -> bool:
    return bool(np.all(labels == 0))


def main() -> int:
    repo_root = REPO_ROOT
    audio_csv_dir = repo_root / "data/processed/audio/fusion_ready"
    video_csv_dir = repo_root / "data/processed/video/fusion_ready"

    weight_path = (
        repo_root
        / "checkpoints/late_fusion/ecapa_video_reuse_baseline/weighted_prob_avg/fitted_weight.json"
    )
    if not weight_path.is_file():
        raise FileNotFoundError(f"Fitted weight file not found: {weight_path}")

    weight_cfg = json.loads(weight_path.read_text(encoding="utf-8"))
    audio_weight = float(weight_cfg["audio_weight"])
    video_weight = float(weight_cfg["video_weight"])

    metrics_dir = repo_root / "outputs/metrics/late_fusion_custom/weighted_prob_avg"
    preds_dir = repo_root / "outputs/predictions/late_fusion_custom/weighted_prob_avg"
    av_join_dir = repo_root / "data/processed/av/late_fusion_custom"
    log_dir = repo_root / "outputs/logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    av_join_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    split_timings: dict[str, float] = {}
    summary: dict[str, Any] = {}
    start = time.perf_counter()

    print(f"[Fusion] Using audio_weight={audio_weight}, video_weight={video_weight}")
    for split in ("train", "valid", "test"):
        split_start = time.perf_counter()
        audio_rows = _load_csv_rows(audio_csv_dir / f"audio_ecapa_custom_{split}.csv")
        video_rows = _load_csv_rows(video_csv_dir / f"video_reuse_custom_{split}.csv")

        av_rows = _join_audio_video(audio_rows, video_rows, split=split)
        av_path = av_join_dir / f"av_ecapa_video_reuse_custom_{split}.csv"
        _write_csv(av_path, av_rows, fieldnames=AV_JOIN_FIELDS)

        audio_probs = _load_prob_matrix(av_rows, prefix="audio")
        video_probs = _load_prob_matrix(av_rows, prefix="video")
        fused_probs = audio_weight * audio_probs + video_weight * video_probs

        labels = np.asarray([int(row["label_id"]) for row in av_rows], dtype=np.int64)
        metrics = compute_classification_metrics(labels, fused_probs)
        pred_ids = metrics["pred_ids"]

        pred_path = preds_dir / f"{split}_predictions.jsonl"
        _write_fusion_predictions(
            pred_path,
            av_rows,
            fused_probs,
            pred_ids,
            audio_weight=audio_weight,
            video_weight=video_weight,
        )

        cm_path = metrics_dir / f"{split}_confusion_matrix.csv"
        write_confusion_matrix_csv(cm_path, np.asarray(metrics["confusion_matrix"], dtype=np.int64))

        payload = {
            "split": split,
            "mode": "weighted_prob_avg",
            "audio_weight": audio_weight,
            "video_weight": video_weight,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "uar": metrics["uar"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "labels_are_placeholder": labels_are_placeholder(labels),
            "evaluation_time_sec": time.perf_counter() - split_start,
        }
        (metrics_dir / f"{split}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary[split] = payload
        split_timings[split] = payload["evaluation_time_sec"]
        print(
            f"[Fusion] {split}: acc={payload['accuracy']:.4f} "
            f"macro_f1={payload['macro_f1']:.4f} uar={payload['uar']:.4f} "
            f"time={payload['evaluation_time_sec']:.3f}s"
        )

    total_runtime_sec = time.perf_counter() - start
    runtime_payload = {
        "mode": "weighted_prob_avg",
        "audio_weight": audio_weight,
        "video_weight": video_weight,
        "split_runtime_sec": split_timings,
        "total_runtime_sec": total_runtime_sec,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    runtime_path = log_dir / "late_fusion_custom_runtime.json"
    runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    print(f"[Fusion] total_runtime_sec={total_runtime_sec:.3f}")
    print(f"[Fusion] runtime_json={runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
