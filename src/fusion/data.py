from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.ravdess import LABEL_ORDER, RAVDESS_SPLIT_ORDER

AUDIO_PROB_FIELDS = [f"prob_{label_name}" for label_name in LABEL_ORDER]
VIDEO_PROB_FIELDS = [f"prob_{label_name}" for label_name in LABEL_ORDER]

AV_JOIN_FIELDS = [
    "av_key",
    "split",
    "actor_id",
    "label_id",
    "label_name",
    "emotion_code",
    "intensity_code",
    "statement_code",
    "repetition_code",
    "audio_sample_id",
    "audio_wav_path",
    "audio_pred_id",
    "audio_pred_name",
    "audio_embedding_path",
    "audio_embedding_dim",
    "video_sample_id",
    "video_path",
    "video_pred_id",
    "video_pred_name",
    "video_sequence_path",
    "video_sequence_num_frames",
    "video_sequence_feature_dim",
    "video_sequence_source",
    "video_prediction_source",
]
AV_JOIN_FIELDS.extend([f"audio_{field}" for field in AUDIO_PROB_FIELDS])
AV_JOIN_FIELDS.extend([f"video_{field}" for field in VIDEO_PROB_FIELDS])


@dataclass(frozen=True)
class LateFusionLayout:
    repo_root: Path
    audio_fusion_ready_root: Path
    video_fusion_ready_root: Path
    av_join_root: Path
    checkpoints_root: Path
    metrics_root: Path
    predictions_root: Path
    logs_root: Path


def build_late_fusion_layout(run_name: str, repo_root: Path | None = None) -> LateFusionLayout:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    return LateFusionLayout(
        repo_root=resolved_repo_root,
        audio_fusion_ready_root=resolved_repo_root / "data/processed/audio/fusion_ready",
        video_fusion_ready_root=resolved_repo_root / "data/processed/video/fusion_ready",
        av_join_root=resolved_repo_root / "data/processed/av/late_fusion",
        checkpoints_root=resolved_repo_root / "checkpoints/late_fusion" / run_name,
        metrics_root=resolved_repo_root / "outputs/metrics/late_fusion" / run_name,
        predictions_root=resolved_repo_root / "outputs/predictions/late_fusion" / run_name,
        logs_root=resolved_repo_root / "outputs/logs/late_fusion" / run_name,
    )


def build_av_late_fusion_tables(layout: LateFusionLayout) -> dict[str, Path]:
    split_paths: dict[str, Path] = {}
    combined_rows: list[dict[str, Any]] = []

    for split in RAVDESS_SPLIT_ORDER:
        audio_rows = load_csv_rows(layout.audio_fusion_ready_root / f"audio_ecapa_{split}.csv")
        video_rows = load_csv_rows(layout.video_fusion_ready_root / f"video_reuse_{split}.csv")
        joined_rows = join_audio_video_rows(audio_rows, video_rows, split=split)
        split_path = layout.av_join_root / f"av_ecapa_video_reuse_{split}.csv"
        write_csv(split_path, joined_rows, fieldnames=AV_JOIN_FIELDS)
        split_paths[split] = split_path
        combined_rows.extend(joined_rows)

    all_path = layout.av_join_root / "av_ecapa_video_reuse_all.csv"
    write_csv(all_path, combined_rows, fieldnames=AV_JOIN_FIELDS)
    split_paths["all"] = all_path
    return split_paths


def load_join_rows(layout: LateFusionLayout, split: str) -> list[dict[str, str]]:
    return load_csv_rows(layout.av_join_root / f"av_ecapa_video_reuse_{split}.csv")


def join_audio_video_rows(
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
        validate_join_pair(audio_row, video_row, split=split)

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


def validate_join_pair(audio_row: dict[str, str], video_row: dict[str, str], *, split: str) -> None:
    for field in (
        "av_key",
        "actor_id",
        "split",
        "label_id",
        "label_name",
        "emotion_code",
        "intensity_code",
        "statement_code",
        "repetition_code",
    ):
        if audio_row[field] != video_row[field]:
            raise ValueError(
                f"Join mismatch for split '{split}', av_key '{audio_row['av_key']}', field '{field}': "
                f"audio={audio_row[field]} video={video_row[field]}"
            )

    if video_row["late_fusion_ready"] != "1":
        raise ValueError(f"Video row '{video_row['sample_id']}' is not marked late_fusion_ready.")
    if video_row["has_sequence"] != "1" or video_row["has_prediction"] != "1":
        raise ValueError(f"Video row '{video_row['sample_id']}' is missing sequence or prediction artifacts.")


def load_prob_matrix(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    fields = [f"{prefix}_{field}" for field in AUDIO_PROB_FIELDS]
    matrix = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=np.float32)
    if matrix.shape != (len(rows), len(LABEL_ORDER)):
        raise ValueError(f"Probability matrix shape mismatch for prefix '{prefix}': {matrix.shape}.")
    return matrix


def load_labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)


def load_audio_embeddings(rows: list[dict[str, str]]) -> np.ndarray:
    embeddings = [np.asarray(np.load(row["audio_embedding_path"]), dtype=np.float32) for row in rows]
    matrix = np.stack(embeddings, axis=0)
    if matrix.shape[1] != 96:
        raise ValueError(f"Expected audio embedding dim 96, got {matrix.shape}.")
    return matrix


def load_video_sequence_embeddings(rows: list[dict[str, str]]) -> np.ndarray:
    vectors = []
    for row in rows:
        sequence = np.asarray(np.load(row["video_sequence_path"]), dtype=np.float32)
        if sequence.shape != (5, 2048):
            raise ValueError(
                f"Expected video sequence shape (5, 2048) for '{row['video_sample_id']}', got {sequence.shape}."
            )
        vectors.append(sequence.mean(axis=0))

    matrix = np.stack(vectors, axis=0)
    if matrix.shape[1] != 2048:
        raise ValueError(f"Expected pooled video embedding dim 2048, got {matrix.shape}.")
    return matrix


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
