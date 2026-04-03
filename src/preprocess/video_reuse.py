from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.common.ravdess import LABEL_ORDER, RAVDESS_SPLIT_ORDER, build_av_key, parse_ravdess_stem

CANONICAL_SPLIT_COUNTS = {
    "train": 960,
    "valid": 240,
    "test": 240,
}

EXPECTED_SEQUENCE_SHAPE = (5, 2048)
EXPECTED_FRAME_COUNT = 5

PACKAGE_PREDICTION_FILES = {
    "train": "predictions_train.jsonl",
    "valid": "predictions_val.jsonl",
    "test": "predictions_test.jsonl",
}

FUSION_TABLE_FIELDS = [
    "av_key",
    "sample_id",
    "stem",
    "video_path",
    "actor_id",
    "split",
    "emotion_code",
    "label_id",
    "label_name",
    "intensity_code",
    "statement_code",
    "repetition_code",
    "sequence_video_id",
    "sequence_video_id_source",
    "sequence_path",
    "sequence_num_frames",
    "sequence_feature_dim",
    "prediction_video_id",
    "prediction_video_id_source",
    "prediction_frame_count",
    "pred_id",
    "pred_name",
    "prob_neutral",
    "prob_calm",
    "prob_happy",
    "prob_sad",
    "prob_angry",
    "prob_fearful",
    "prob_disgust",
    "prob_surprise",
    "has_sequence",
    "has_prediction",
    "late_fusion_ready",
    "artifact_status",
]

MISSING_ARTIFACT_FIELDS = [
    "split",
    "av_key",
    "sample_id",
    "actor_id",
    "label_id",
    "label_name",
    "missing_sequence",
    "missing_prediction",
]


@dataclass(frozen=True)
class VideoReuseLayout:
    repo_root: Path
    package_root: Path
    sequence_root: Path
    backfill_root: Path
    backfill_prediction_index_path: Path
    backfill_sequence_index_path: Path
    video_manifest_dir: Path
    processed_video_root: Path
    normalized_root: Path
    normalized_predictions_root: Path
    normalized_sequence_root: Path
    audit_root: Path
    audit_path: Path
    missing_artifacts_path: Path
    fusion_ready_root: Path


def build_video_reuse_layout(repo_root: Path | None = None) -> VideoReuseLayout:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    processed_video_root = resolved_repo_root / "data/processed/video"
    normalized_root = processed_video_root / "reusable_reference"
    audit_root = processed_video_root / "audit"

    return VideoReuseLayout(
        repo_root=resolved_repo_root,
        package_root=resolved_repo_root / "external/video_face_branch/early_fusion_facial_v2",
        sequence_root=resolved_repo_root / "external/video_face_branch/early_fusion_facial_v2/fusion_pack_5f/sequences",
        backfill_root=resolved_repo_root / "data/processed/video/backfill_actor15",
        backfill_prediction_index_path=resolved_repo_root
        / "data/processed/video/backfill_actor15/predictions_video_level/video_backfill_train.jsonl",
        backfill_sequence_index_path=resolved_repo_root
        / "data/processed/video/backfill_actor15/sequence_index/video_backfill_train.jsonl",
        video_manifest_dir=resolved_repo_root / "data/manifests/video",
        processed_video_root=processed_video_root,
        normalized_root=normalized_root,
        normalized_predictions_root=normalized_root / "predictions_video_level",
        normalized_sequence_root=normalized_root / "sequence_index",
        audit_root=audit_root,
        audit_path=audit_root / "video_reuse_audit.json",
        missing_artifacts_path=audit_root / "video_missing_reusable.csv",
        fusion_ready_root=processed_video_root / "fusion_ready",
    )


def audit_video_reuse(
    layout: VideoReuseLayout,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    analysis = analyze_video_reuse(layout, expected_counts=expected_counts)
    return analysis["audit"]


def normalize_video_artifacts(
    layout: VideoReuseLayout,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    analysis = analyze_video_reuse(layout, expected_counts=expected_counts)

    written_predictions: dict[str, Path] = {}
    written_sequences: dict[str, Path] = {}
    for split in RAVDESS_SPLIT_ORDER:
        predictions_path = layout.normalized_predictions_root / f"video_reuse_{split}.jsonl"
        sequences_path = layout.normalized_sequence_root / f"video_reuse_{split}.jsonl"
        write_jsonl(predictions_path, analysis["normalized_predictions"][split])
        write_jsonl(sequences_path, analysis["normalized_sequences"][split])
        written_predictions[split] = predictions_path
        written_sequences[split] = sequences_path

    layout.audit_root.mkdir(parents=True, exist_ok=True)
    layout.audit_path.write_text(json.dumps(analysis["audit"], indent=2), encoding="utf-8")
    write_csv(
        layout.missing_artifacts_path,
        analysis["missing_rows"],
        fieldnames=MISSING_ARTIFACT_FIELDS,
    )

    return {
        "predictions": written_predictions,
        "sequences": written_sequences,
        "audit_path": layout.audit_path,
        "missing_artifacts_path": layout.missing_artifacts_path,
        "audit": analysis["audit"],
    }


def build_fusion_ready_video_tables(
    layout: VideoReuseLayout,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Path]:
    manifest_rows = load_video_manifest_rows(layout.video_manifest_dir)
    resolved_expected_counts = expected_counts or expected_counts_from_manifests(manifest_rows)
    if expected_counts is None:
        validate_expected_counts(resolved_expected_counts)

    split_paths: dict[str, Path] = {}
    combined_rows: list[dict[str, Any]] = []
    layout.fusion_ready_root.mkdir(parents=True, exist_ok=True)

    for split in RAVDESS_SPLIT_ORDER:
        prediction_index = {
            row["sample_id"]: row
            for row in load_jsonl(layout.normalized_predictions_root / f"video_reuse_{split}.jsonl")
        }
        sequence_index = {
            row["sample_id"]: row
            for row in load_jsonl(layout.normalized_sequence_root / f"video_reuse_{split}.jsonl")
        }

        joined_rows: list[dict[str, Any]] = []
        for manifest_row in manifest_rows[split]:
            sample_id = manifest_row["sample_id"]
            prediction_row = prediction_index.get(sample_id)
            sequence_row = sequence_index.get(sample_id)

            row = {
                "av_key": manifest_row["av_key"],
                "sample_id": sample_id,
                "stem": manifest_row["stem"],
                "video_path": manifest_row["video_path"],
                "actor_id": manifest_row["actor_id"],
                "split": split,
                "emotion_code": manifest_row["emotion_code"],
                "label_id": manifest_row["label_id"],
                "label_name": manifest_row["label_name"],
                "intensity_code": manifest_row["intensity_code"],
                "statement_code": manifest_row["statement_code"],
                "repetition_code": manifest_row["repetition_code"],
                "sequence_video_id": "",
                "sequence_video_id_source": "",
                "sequence_path": "",
                "sequence_num_frames": "",
                "sequence_feature_dim": "",
                "prediction_video_id": "",
                "prediction_video_id_source": "",
                "prediction_frame_count": "",
                "pred_id": "",
                "pred_name": "",
                "has_sequence": "0",
                "has_prediction": "0",
                "late_fusion_ready": "0",
                "artifact_status": "",
            }
            row.update(empty_probability_columns())

            parsed = parse_ravdess_stem(sample_id)
            if parsed.av_key != manifest_row["av_key"]:
                raise ValueError(f"Manifest av_key mismatch for video sample '{sample_id}'.")

            if sequence_row is not None:
                row["sequence_video_id"] = sequence_row["sequence_video_id"]
                row["sequence_video_id_source"] = sequence_row["sequence_video_id_source"]
                row["sequence_path"] = sequence_row["sequence_path"]
                row["sequence_num_frames"] = sequence_row["sequence_num_frames"]
                row["sequence_feature_dim"] = sequence_row["sequence_feature_dim"]
                row["has_sequence"] = "1"

            if prediction_row is not None:
                row["prediction_video_id"] = prediction_row["prediction_video_id"]
                row["prediction_video_id_source"] = prediction_row["prediction_video_id_source"]
                row["prediction_frame_count"] = prediction_row["prediction_frame_count"]
                row["pred_id"] = prediction_row["pred_id"]
                row["pred_name"] = prediction_row["pred_name"]
                for label_name, prob in zip(LABEL_ORDER, prediction_row["probs"]):
                    row[f"prob_{label_name}"] = prob
                row["has_prediction"] = "1"

            row["late_fusion_ready"] = "1" if row["has_sequence"] == "1" and row["has_prediction"] == "1" else "0"
            row["artifact_status"] = compute_artifact_status(
                has_sequence=row["has_sequence"] == "1",
                has_prediction=row["has_prediction"] == "1",
            )
            joined_rows.append(row)

        split_path = layout.fusion_ready_root / f"video_reuse_{split}.csv"
        write_csv(split_path, joined_rows, fieldnames=FUSION_TABLE_FIELDS)
        split_paths[split] = split_path
        combined_rows.extend(joined_rows)

    all_path = layout.fusion_ready_root / "video_reuse_all.csv"
    write_csv(all_path, combined_rows, fieldnames=FUSION_TABLE_FIELDS)
    split_paths["all"] = all_path
    return split_paths


def analyze_video_reuse(
    layout: VideoReuseLayout,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    manifest_rows = load_video_manifest_rows(layout.video_manifest_dir)
    resolved_expected_counts = expected_counts or expected_counts_from_manifests(manifest_rows)
    if expected_counts is None:
        validate_expected_counts(resolved_expected_counts)

    sequence_candidates, sequence_audit = collect_sequence_candidates(layout)
    prediction_candidates, prediction_audit = collect_prediction_candidates(layout)

    normalized_predictions: dict[str, list[dict[str, Any]]] = {split: [] for split in RAVDESS_SPLIT_ORDER}
    normalized_sequences: dict[str, list[dict[str, Any]]] = {split: [] for split in RAVDESS_SPLIT_ORDER}
    missing_rows: list[dict[str, Any]] = []
    selected_summary: dict[str, dict[str, Any]] = {}

    for split in RAVDESS_SPLIT_ORDER:
        prediction_fallbacks = 0
        sequence_fallbacks = 0
        missing_prediction_keys: list[str] = []
        missing_sequence_keys: list[str] = []
        prediction_frame_counts: dict[int, int] = {}
        selected_sequence_shapes: dict[str, int] = {}

        for manifest_row in manifest_rows[split]:
            av_key = manifest_row["av_key"]

            prediction_record, prediction_source = select_preferred_record(
                av_key,
                prediction_candidates[split].get(av_key, {}),
            )
            sequence_record, sequence_source = select_preferred_record(
                av_key,
                sequence_candidates[split].get(av_key, {}),
            )

            if prediction_record is not None:
                if prediction_source == "fallback_01":
                    prediction_fallbacks += 1
                prediction_frame_counts[int(prediction_record["prediction_frame_count"])] = (
                    prediction_frame_counts.get(int(prediction_record["prediction_frame_count"]), 0) + 1
                )
                normalized_predictions[split].append(
                    {
                        "sample_id": manifest_row["sample_id"],
                        "av_key": av_key,
                        "split": split,
                        "actor_id": manifest_row["actor_id"],
                        "label_id": manifest_row["label_id"],
                        "label_name": manifest_row["label_name"],
                        "prediction_video_id": prediction_record["prediction_video_id"],
                        "prediction_video_id_source": prediction_record.get(
                            "prediction_video_id_source",
                            prediction_source,
                        ),
                        "prediction_frame_count": prediction_record["prediction_frame_count"],
                        "pred_id": prediction_record["pred_id"],
                        "pred_name": prediction_record["pred_name"],
                        "probs": prediction_record["probs"],
                    }
                )
            else:
                missing_prediction_keys.append(av_key)

            if sequence_record is not None:
                if sequence_source == "fallback_01":
                    sequence_fallbacks += 1
                shape_key = f"({sequence_record['sequence_num_frames']}, {sequence_record['sequence_feature_dim']})"
                selected_sequence_shapes[shape_key] = selected_sequence_shapes.get(shape_key, 0) + 1
                normalized_sequences[split].append(
                    {
                        "sample_id": manifest_row["sample_id"],
                        "av_key": av_key,
                        "split": split,
                        "actor_id": manifest_row["actor_id"],
                        "label_id": manifest_row["label_id"],
                        "label_name": manifest_row["label_name"],
                        "sequence_video_id": sequence_record["sequence_video_id"],
                        "sequence_video_id_source": sequence_record.get(
                            "sequence_video_id_source",
                            sequence_source,
                        ),
                        "sequence_path": sequence_record["sequence_path"],
                        "sequence_num_frames": sequence_record["sequence_num_frames"],
                        "sequence_feature_dim": sequence_record["sequence_feature_dim"],
                    }
                )
            else:
                missing_sequence_keys.append(av_key)

            if prediction_record is None or sequence_record is None:
                missing_rows.append(
                    {
                        "split": split,
                        "av_key": av_key,
                        "sample_id": manifest_row["sample_id"],
                        "actor_id": manifest_row["actor_id"],
                        "label_id": manifest_row["label_id"],
                        "label_name": manifest_row["label_name"],
                        "missing_sequence": "1" if sequence_record is None else "0",
                        "missing_prediction": "1" if prediction_record is None else "0",
                    }
                )

        selected_summary[split] = {
            "expected_manifest_rows": resolved_expected_counts[split],
            "normalized_prediction_rows": len(normalized_predictions[split]),
            "normalized_sequence_rows": len(normalized_sequences[split]),
            "prediction_fallback_to_01": prediction_fallbacks,
            "sequence_fallback_to_01": sequence_fallbacks,
            "missing_prediction_count": len(missing_prediction_keys),
            "missing_sequence_count": len(missing_sequence_keys),
            "prediction_frame_count_distribution": prediction_frame_counts,
            "selected_sequence_shape_counts": selected_sequence_shapes,
            "missing_prediction_av_keys": missing_prediction_keys,
            "missing_sequence_av_keys": missing_sequence_keys,
            "late_fusion_ready_rows": sum(
                1 for row in manifest_rows[split] if row["av_key"] not in missing_prediction_keys and row["av_key"] not in missing_sequence_keys
            ),
        }

    audit = {
        "package_root": layout.package_root.as_posix(),
        "video_manifest_dir": layout.video_manifest_dir.as_posix(),
        "canonical_expected_counts": resolved_expected_counts,
        "predictions": prediction_audit,
        "sequences": sequence_audit,
        "selected_against_canonical_manifest": selected_summary,
        "late_fusion_ready": all(
            selected_summary[split]["late_fusion_ready_rows"] == resolved_expected_counts[split]
            for split in RAVDESS_SPLIT_ORDER
        ),
    }

    return {
        "normalized_predictions": normalized_predictions,
        "normalized_sequences": normalized_sequences,
        "missing_rows": missing_rows,
        "audit": audit,
    }


def collect_prediction_candidates(layout: VideoReuseLayout) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in RAVDESS_SPLIT_ORDER}
    audit_summary: dict[str, Any] = {
        "source_files": {},
        "frame_level": True,
        "selected_av_key_counts": {},
        "backfill_rows": {},
    }

    for split, filename in PACKAGE_PREDICTION_FILES.items():
        source_path = layout.package_root / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Video prediction file not found: '{source_path}'.")

        rows = load_jsonl(source_path)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            sample_id = str(row["sample_id"])
            video_id = sample_id.split("_f", 1)[0]
            grouped.setdefault(video_id, []).append(row)

        unique_video_ids = 0
        unique_speech_video_ids = 0
        unique_speech_av_keys = set()
        frame_counts: dict[int, int] = {}

        for package_video_id, frame_rows in grouped.items():
            unique_video_ids += 1
            parsed = parse_ravdess_stem(package_video_id)
            if parsed.vocal_channel != "01":
                continue
            if parsed.split != split:
                raise ValueError(
                    f"Prediction split mismatch for package video '{package_video_id}': expected {split}, got {parsed.split}."
                )

            unique_speech_video_ids += 1
            unique_speech_av_keys.add(parsed.av_key)
            aggregated = aggregate_frame_predictions(frame_rows, package_video_id=package_video_id)
            frame_count = int(aggregated["prediction_frame_count"])
            frame_counts[frame_count] = frame_counts.get(frame_count, 0) + 1
            candidates[split].setdefault(parsed.av_key, {})[package_video_id] = aggregated

        audit_summary["source_files"][split] = {
            "path": source_path.as_posix(),
            "frame_rows": len(rows),
            "unique_video_ids": unique_video_ids,
            "unique_speech_video_ids": unique_speech_video_ids,
            "unique_speech_av_keys": len(unique_speech_av_keys),
            "frame_count_distribution": frame_counts,
        }
        audit_summary["selected_av_key_counts"][split] = len(candidates[split])

    backfill_rows = load_optional_backfill_jsonl(layout.backfill_prediction_index_path)
    audit_summary["backfill_rows"]["train"] = len(backfill_rows)
    for row in backfill_rows:
        split = str(row["split"])
        av_key = str(row["av_key"])
        package_video_id = str(row["prediction_video_id"])
        candidates[split].setdefault(av_key, {}).setdefault(package_video_id, dict(row))
    for split in RAVDESS_SPLIT_ORDER:
        audit_summary["selected_av_key_counts"][split] = len(candidates[split])

    return candidates, audit_summary


def collect_sequence_candidates(layout: VideoReuseLayout) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in RAVDESS_SPLIT_ORDER}
    total_shape_counts: dict[str, int] = {}
    speech_shape_counts: dict[str, int] = {}
    total_files = 0
    speech_files = 0
    non_speech_files = 0

    if not layout.sequence_root.is_dir():
        raise FileNotFoundError(f"Video sequence root not found: '{layout.sequence_root}'.")

    for path in sorted(layout.sequence_root.glob("*.npy")):
        total_files += 1
        parsed = parse_ravdess_stem(path.stem)
        shape = validate_sequence_file(path)
        shape_key = str(shape)
        total_shape_counts[shape_key] = total_shape_counts.get(shape_key, 0) + 1

        if parsed.vocal_channel != "01":
            non_speech_files += 1
            continue

        speech_files += 1
        speech_shape_counts[shape_key] = speech_shape_counts.get(shape_key, 0) + 1
        candidates[parsed.split].setdefault(parsed.av_key, {})[parsed.stem] = {
            "sequence_video_id": parsed.stem,
            "sequence_path": path.resolve().as_posix(),
            "sequence_num_frames": shape[0],
            "sequence_feature_dim": shape[1],
        }

    audit_summary = {
        "sequence_root": layout.sequence_root.as_posix(),
        "total_files": total_files,
        "speech_files": speech_files,
        "non_speech_files": non_speech_files,
        "all_shape_counts": total_shape_counts,
        "speech_shape_counts": speech_shape_counts,
        "selected_av_key_counts": {split: len(candidates[split]) for split in RAVDESS_SPLIT_ORDER},
        "backfill_rows": {"train": 0},
    }

    backfill_rows = load_optional_backfill_jsonl(layout.backfill_sequence_index_path)
    audit_summary["backfill_rows"]["train"] = len(backfill_rows)
    for row in backfill_rows:
        sequence_path = Path(str(row["sequence_path"]))
        shape = validate_sequence_file(sequence_path)
        if shape != EXPECTED_SEQUENCE_SHAPE:
            raise ValueError(
                f"Backfill sequence at '{sequence_path}' must have shape {EXPECTED_SEQUENCE_SHAPE}, got {shape}."
            )
        split = str(row["split"])
        av_key = str(row["av_key"])
        video_id = str(row["sequence_video_id"])
        candidates[split].setdefault(av_key, {}).setdefault(video_id, dict(row))

    audit_summary["selected_av_key_counts"] = {split: len(candidates[split]) for split in RAVDESS_SPLIT_ORDER}
    return candidates, audit_summary


def aggregate_frame_predictions(frame_rows: list[dict[str, Any]], *, package_video_id: str) -> dict[str, Any]:
    if not frame_rows:
        raise ValueError(f"No frame rows provided for package video '{package_video_id}'.")
    if len(frame_rows) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"Package video '{package_video_id}' must have {EXPECTED_FRAME_COUNT} frame predictions, got {len(frame_rows)}."
        )

    parsed = parse_ravdess_stem(package_video_id)
    if parsed.vocal_channel != "01":
        raise ValueError(f"Package video '{package_video_id}' is not speech-only.")

    stacked_probs: list[np.ndarray] = []
    for row in frame_rows:
        sample_id = str(row["sample_id"])
        if sample_id.split("_f", 1)[0] != package_video_id:
            raise ValueError(
                f"Frame sample '{sample_id}' does not belong to package video '{package_video_id}'."
            )
        probs = row.get("probs")
        validate_probability_vector(probs, sample_id=sample_id)
        stacked_probs.append(np.asarray(probs, dtype=np.float64))

        if int(row["label_id"]) != parsed.label_id:
            raise ValueError(f"Label id mismatch for frame sample '{sample_id}'.")
        if str(row["label_name"]) != parsed.label_name:
            raise ValueError(f"Label name mismatch for frame sample '{sample_id}'.")

    mean_probs = np.stack(stacked_probs, axis=0).mean(axis=0)
    pred_id = int(np.argmax(mean_probs))

    return {
        "prediction_video_id": package_video_id,
        "av_key": build_av_key(package_video_id),
        "split": parsed.split,
        "actor_id": parsed.actor_id,
        "label_id": parsed.label_id,
        "label_name": parsed.label_name,
        "prediction_frame_count": len(frame_rows),
        "pred_id": pred_id,
        "pred_name": LABEL_ORDER[pred_id],
        "probs": [float(value) for value in mean_probs.tolist()],
    }


def select_preferred_record(
    av_key: str,
    candidates_by_video_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if not candidates_by_video_id:
        return None, ""

    preferred_video_id = f"02-{av_key}"
    fallback_video_id = f"01-{av_key}"
    if preferred_video_id in candidates_by_video_id:
        return candidates_by_video_id[preferred_video_id], "canonical_02"
    if fallback_video_id in candidates_by_video_id:
        return candidates_by_video_id[fallback_video_id], "fallback_01"

    first_video_id = sorted(candidates_by_video_id)[0]
    return candidates_by_video_id[first_video_id], "fallback_other"


def validate_sequence_file(path: Path) -> tuple[int, int]:
    array = np.load(path, mmap_mode="r")
    if array.shape != EXPECTED_SEQUENCE_SHAPE:
        raise ValueError(
            f"Sequence at '{path}' must have shape {EXPECTED_SEQUENCE_SHAPE}, got {array.shape}."
        )
    return int(array.shape[0]), int(array.shape[1])


def validate_probability_vector(probs: Any, *, sample_id: str) -> None:
    if not isinstance(probs, list) or len(probs) != len(LABEL_ORDER):
        raise ValueError(f"Probability vector for '{sample_id}' must have length 8.")


def load_video_manifest_rows(video_manifest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in RAVDESS_SPLIT_ORDER:
        csv_path = Path(video_manifest_dir) / f"video_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Video manifest not found: '{csv_path}'.")

        rows: list[dict[str, Any]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_row = dict(row)
                normalized_row["actor_id"] = int(row["actor_id"])
                normalized_row["label_id"] = int(row["label_id"])
                rows.append(normalized_row)
        rows_by_split[split] = rows
    return rows_by_split


def expected_counts_from_manifests(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {split: len(rows) for split, rows in rows_by_split.items()}


def validate_expected_counts(actual_counts: dict[str, int]) -> None:
    for split, expected_count in CANONICAL_SPLIT_COUNTS.items():
        actual_count = actual_counts.get(split)
        if actual_count != expected_count:
            raise ValueError(
                f"Count mismatch for split '{split}': expected {expected_count}, got {actual_count}."
            )


def compute_artifact_status(*, has_sequence: bool, has_prediction: bool) -> str:
    if has_sequence and has_prediction:
        return "complete"
    if has_sequence:
        return "missing_prediction"
    if has_prediction:
        return "missing_sequence"
    return "missing_both"


def empty_probability_columns() -> dict[str, str]:
    return {f"prob_{label_name}": "" for label_name in LABEL_ORDER}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_optional_backfill_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if Path(path).is_file() else []


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
