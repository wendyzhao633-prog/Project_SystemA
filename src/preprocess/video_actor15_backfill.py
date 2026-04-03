from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.common.ravdess import LABEL_ORDER, build_av_key, parse_ravdess_stem
from src.models.video_reference_resnet import (
    load_reference_backbone,
    load_reference_expression_model,
)

EXPECTED_MISSING_ROWS = 60
EXPECTED_SEQUENCE_SHAPE = (5, 2048)
FRAMES_PER_VIDEO = 5
IMAGE_SIZE = 256


@dataclass(frozen=True)
class Actor15BackfillLayout:
    repo_root: Path
    package_root: Path
    missing_rows_path: Path
    video_manifest_path: Path
    best_backbone_path: Path
    best_uar_path: Path
    backfill_root: Path
    sequences_root: Path
    predictions_root: Path
    sequence_index_path: Path
    prediction_index_path: Path
    summary_path: Path


def build_actor15_backfill_layout(repo_root: Path | None = None) -> Actor15BackfillLayout:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    package_root = resolved_repo_root / "external/video_face_branch/early_fusion_facial_v2"
    backfill_root = resolved_repo_root / "data/processed/video/backfill_actor15"
    return Actor15BackfillLayout(
        repo_root=resolved_repo_root,
        package_root=package_root,
        missing_rows_path=resolved_repo_root / "data/processed/video/audit/video_missing_reusable.csv",
        video_manifest_path=resolved_repo_root / "data/manifests/video/video_train.csv",
        best_backbone_path=package_root / "best_backbone.pth",
        best_uar_path=package_root / "best_uar.pth",
        backfill_root=backfill_root,
        sequences_root=backfill_root / "sequence_embeddings_5f/train",
        predictions_root=backfill_root / "predictions_video_level",
        sequence_index_path=backfill_root / "sequence_index/video_backfill_train.jsonl",
        prediction_index_path=backfill_root / "predictions_video_level/video_backfill_train.jsonl",
        summary_path=backfill_root / "backfill_summary.json",
    )


def run_actor15_backfill(layout: Actor15BackfillLayout) -> dict[str, Any]:
    missing_rows = load_missing_actor15_rows(layout.missing_rows_path, allow_empty=True)
    if not missing_rows:
        return summarize_existing_backfill(layout)

    manifest_index = load_video_manifest_index(layout.video_manifest_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    face_detector = build_face_detector()

    backbone_model = load_reference_backbone(layout.best_backbone_path, device=device)
    expression_model = load_reference_expression_model(layout.best_uar_path, device=device)

    layout.sequences_root.mkdir(parents=True, exist_ok=True)
    layout.predictions_root.mkdir(parents=True, exist_ok=True)
    layout.sequence_index_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []

    for missing_row in missing_rows:
        sample_id = str(missing_row["sample_id"])
        manifest_row = manifest_index[sample_id]
        parsed = parse_ravdess_stem(sample_id)
        video_path = (layout.repo_root / manifest_row["video_path"]).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing raw video for backfill: '{video_path}'.")

        frames, frame_indices, face_hits = extract_video_frames(
            video_path,
            frames_per_video=FRAMES_PER_VIDEO,
            image_size=IMAGE_SIZE,
            face_detector=face_detector,
        )
        if frames.shape != (FRAMES_PER_VIDEO, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Unexpected frame tensor shape for '{sample_id}': {tuple(frames.shape)}."
            )

        inputs = frames.to(device)
        with torch.no_grad():
            sequence_tensor = backbone_model(inputs).cpu().numpy().astype(np.float32)
            _, logits = expression_model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

        if sequence_tensor.shape != EXPECTED_SEQUENCE_SHAPE:
            raise ValueError(
                f"Backfilled sequence for '{sample_id}' must have shape {EXPECTED_SEQUENCE_SHAPE}, "
                f"got {sequence_tensor.shape}."
            )
        if probs.shape != (FRAMES_PER_VIDEO, len(LABEL_ORDER)):
            raise ValueError(
                f"Backfilled frame probabilities for '{sample_id}' must have shape "
                f"({FRAMES_PER_VIDEO}, {len(LABEL_ORDER)}), got {probs.shape}."
            )

        mean_probs = probs.mean(axis=0)
        pred_id = int(np.argmax(mean_probs))

        sequence_path = (layout.sequences_root / f"{sample_id}.npy").resolve()
        np.save(sequence_path, sequence_tensor)

        prediction_rows.append(
            {
                "sample_id": sample_id,
                "av_key": build_av_key(sample_id),
                "split": "train",
                "actor_id": parsed.actor_id,
                "label_id": parsed.label_id,
                "label_name": parsed.label_name,
                "prediction_video_id": sample_id,
                "prediction_video_id_source": "backfill_actor15",
                "prediction_frame_count": FRAMES_PER_VIDEO,
                "pred_id": pred_id,
                "pred_name": LABEL_ORDER[pred_id],
                "probs": [float(value) for value in mean_probs.tolist()],
            }
        )

        sequence_rows.append(
            {
                "sample_id": sample_id,
                "av_key": build_av_key(sample_id),
                "split": "train",
                "actor_id": parsed.actor_id,
                "label_id": parsed.label_id,
                "label_name": parsed.label_name,
                "sequence_video_id": sample_id,
                "sequence_video_id_source": "backfill_actor15",
                "sequence_path": sequence_path.as_posix(),
                "sequence_num_frames": FRAMES_PER_VIDEO,
                "sequence_feature_dim": EXPECTED_SEQUENCE_SHAPE[1],
            }
        )

        frame_summaries.append(
            {
                "sample_id": sample_id,
                "video_path": video_path.as_posix(),
                "frame_indices": frame_indices,
                "face_crop_hits": face_hits,
            }
        )

    write_jsonl(layout.prediction_index_path, prediction_rows)
    write_jsonl(layout.sequence_index_path, sequence_rows)

    summary = {
        "status": "generated",
        "device": str(device),
        "missing_rows_requested": len(missing_rows),
        "predictions_written": len(prediction_rows),
        "sequences_written": len(sequence_rows),
        "sequence_root": layout.sequences_root.as_posix(),
        "prediction_index_path": layout.prediction_index_path.as_posix(),
        "sequence_index_path": layout.sequence_index_path.as_posix(),
        "frame_summaries": frame_summaries,
    }
    layout.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_missing_actor15_rows(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reusable-row audit not found: '{path}'.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    filtered = [
        row
        for row in rows
        if row["split"] == "train"
        and int(row["actor_id"]) == 15
        and row["missing_sequence"] == "1"
        and row["missing_prediction"] == "1"
    ]

    if allow_empty and len(filtered) == 0:
        return []

    if len(filtered) != EXPECTED_MISSING_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_MISSING_ROWS} actor-15 train rows to backfill, got {len(filtered)}."
        )

    sample_ids = {row["sample_id"] for row in filtered}
    if len(sample_ids) != len(filtered):
        raise ValueError("Duplicate sample_id entries found in actor-15 backfill list.")

    return sorted(filtered, key=lambda row: row["sample_id"])


def summarize_existing_backfill(layout: Actor15BackfillLayout) -> dict[str, Any]:
    prediction_rows = load_jsonl(layout.prediction_index_path)
    sequence_rows = load_jsonl(layout.sequence_index_path)
    sequence_files = sorted(layout.sequences_root.glob("*.npy"))

    if len(prediction_rows) != EXPECTED_MISSING_ROWS:
        raise ValueError(
            "Actor-15 backfill is already marked complete in the audit, but the existing prediction index "
            f"has {len(prediction_rows)} rows instead of {EXPECTED_MISSING_ROWS}: '{layout.prediction_index_path}'."
        )
    if len(sequence_rows) != EXPECTED_MISSING_ROWS:
        raise ValueError(
            "Actor-15 backfill is already marked complete in the audit, but the existing sequence index "
            f"has {len(sequence_rows)} rows instead of {EXPECTED_MISSING_ROWS}: '{layout.sequence_index_path}'."
        )
    if len(sequence_files) != EXPECTED_MISSING_ROWS:
        raise ValueError(
            "Actor-15 backfill is already marked complete in the audit, but the existing sequence directory "
            f"has {len(sequence_files)} .npy files instead of {EXPECTED_MISSING_ROWS}: '{layout.sequences_root}'."
        )

    existing_summary = {}
    if layout.summary_path.is_file():
        existing_summary = json.loads(layout.summary_path.read_text(encoding="utf-8"))

    return {
        "status": "already_complete",
        "device": existing_summary.get("device", "existing"),
        "missing_rows_requested": 0,
        "predictions_written": len(prediction_rows),
        "sequences_written": len(sequence_rows),
        "sequence_root": layout.sequences_root.as_posix(),
        "prediction_index_path": layout.prediction_index_path.as_posix(),
        "sequence_index_path": layout.sequence_index_path.as_posix(),
        "frame_summaries": existing_summary.get("frame_summaries", []),
    }


def load_video_manifest_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: row for row in rows}


def extract_video_frames(
    video_path: Path,
    *,
    frames_per_video: int,
    image_size: int,
    face_detector: cv2.CascadeClassifier | None,
) -> tuple[torch.Tensor, list[int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video '{video_path}'.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    frame_indices = uniform_indices(total_frames, frames_per_video)
    frames, face_hits, observed_frames = read_selected_frames_sequential(
        video_path,
        frame_indices=frame_indices,
        image_size=image_size,
        face_detector=face_detector,
    )

    if len(frames) != frames_per_video and observed_frames > 0 and observed_frames != total_frames:
        frame_indices = uniform_indices(observed_frames, frames_per_video)
        frames, face_hits, observed_frames = read_selected_frames_sequential(
            video_path,
            frame_indices=frame_indices,
            image_size=image_size,
            face_detector=face_detector,
        )

    if len(frames) != frames_per_video:
        raise ValueError(
            f"Failed to extract {frames_per_video} frames from '{video_path}'. "
            f"Observed {observed_frames} readable frames and collected {len(frames)} target frames."
        )

    return torch.stack(frames, dim=0), frame_indices, face_hits


def uniform_indices(num_frames: int, count: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("Video has no frames.")
    if count <= 1:
        return [max(0, min(num_frames - 1, num_frames // 2))]

    positions = np.linspace(0, num_frames - 1, num=count, dtype=float)
    indices = sorted({max(0, min(num_frames - 1, int(round(value)))) for value in positions})
    if len(indices) != count:
        raise ValueError(f"Could not derive exactly {count} unique frame indices from {num_frames} frames.")
    return indices


def read_selected_frames_sequential(
    video_path: Path,
    *,
    frame_indices: list[int],
    image_size: int,
    face_detector: cv2.CascadeClassifier | None,
) -> tuple[list[torch.Tensor], int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video '{video_path}'.")

    targets = set(frame_indices)
    frames_by_index: dict[int, torch.Tensor] = {}
    face_hits = 0
    observed_frames = 0
    max_target = max(frame_indices)

    try:
        while observed_frames <= max_target:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            if observed_frames in targets:
                crop, face_used = crop_frame(frame_bgr, face_detector=face_detector)
                if face_used:
                    face_hits += 1
                resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames_by_index[observed_frames] = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

            observed_frames += 1
            if len(frames_by_index) == len(targets):
                break
    finally:
        cap.release()

    ordered_frames = [frames_by_index[index] for index in frame_indices if index in frames_by_index]
    return ordered_frames, face_hits, observed_frames


def crop_frame(
    frame_bgr: np.ndarray,
    *,
    face_detector: cv2.CascadeClassifier | None,
) -> tuple[np.ndarray, bool]:
    if face_detector is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
            x0, y0 = max(0, x), max(0, y)
            x1 = min(frame_bgr.shape[1], x + w)
            y1 = min(frame_bgr.shape[0], y + h)
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size > 0:
                return crop, True

    return center_crop_square(frame_bgr), False


def center_crop_square(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    y0 = max(0, (height - side) // 2)
    x0 = max(0, (width - side) // 2)
    return frame_bgr[y0 : y0 + side, x0 : x0 + side]


def build_face_detector() -> cv2.CascadeClassifier | None:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return None if detector.empty() else detector


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected JSONL artifact not found: '{path}'.")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
