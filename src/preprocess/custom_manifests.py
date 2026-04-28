"""Custom 200-sample dataset manifest builder.

This module is intentionally dataset-agnostic with respect to RAVDESS.
It does NOT import anything from src.common.ravdess.

Naming conventions for the 4-participant, 200-sample dataset:
  audio: audio_{participant_id}_{sample_no:02d}.wav
  video: video_{participant_id}_{sample_no:02d}.mp4
  PPG:   ppg_{participant_id}_{sample_no:02d}.csv
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Same 8-emotion label order as RAVDESS, defined locally to avoid RAVDESS imports.
CUSTOM_LABEL_ORDER: list[str] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprise",
    "calm",
]
CUSTOM_LABEL_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CUSTOM_LABEL_ORDER)}

MANIFEST_FIELDNAMES: list[str] = [
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
]


@dataclass
class CustomSample:
    participant_id: int
    sample_no: int
    label_id: int
    label_name: str
    split: str
    audio_path: Path
    video_path: Path
    ppg_csv_path: Path

    @property
    def av_key(self) -> str:
        return _build_av_key(self.participant_id, self.sample_no)

    @property
    def file_no(self) -> int:
        """Per-person file number (1-50), derived from the global sample_no."""
        return _per_person_no(self.sample_no)

    @property
    def audio_sample_id(self) -> str:
        return f"audio_{self.participant_id}_{self.file_no:02d}"

    @property
    def video_sample_id(self) -> str:
        return f"video_{self.participant_id}_{self.file_no:02d}"


def _build_av_key(participant_id: int, sample_no: int) -> str:
    return f"{participant_id}_{sample_no:02d}"


def _per_person_no(sample_no: int) -> int:
    """Convert global sample_no (1-200) to per-person file number (1-50)."""
    return ((sample_no - 1) % 50) + 1


def _assign_split(
    participant_id: int,
    *,
    test_participant: int,
    valid_participant: int,
) -> str:
    if participant_id == test_participant:
        return "test"
    if participant_id == valid_participant:
        return "valid"
    return "train"


def load_labels_csv(labels_csv: Path) -> list[dict[str, str]]:
    """Read dataset_200_labeled.csv and return raw rows as dicts."""
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_sample_list(
    labels_csv: Path,
    *,
    audio_root: Path,
    video_root: Path,
    ppg_root: Path,
    test_participant: int,
    valid_participant: int,
) -> tuple[list[CustomSample], list[str]]:
    """Build the full sample list from the labels CSV.

    Returns:
        samples: All samples (regardless of file existence).
        warnings: Human-readable warning messages for missing files.
    """
    rows = load_labels_csv(labels_csv)
    samples: list[CustomSample] = []
    warnings: list[str] = []

    for row in rows:
        pid = int(row["participant_id"])
        sno = int(row["sample_no"])
        label_id = int(row["label_id"])
        label_name = row["label_name"].strip()

        if label_name not in CUSTOM_LABEL_NAME_TO_ID:
            warnings.append(
                f"  [SKIP] Unknown label_name '{label_name}' for participant={pid} sample={sno}."
            )
            continue
        if label_id != CUSTOM_LABEL_NAME_TO_ID[label_name]:
            warnings.append(
                f"  [WARN] label_id/label_name mismatch for participant={pid} sample={sno}: "
                f"id={label_id} name={label_name}."
            )

        split = _assign_split(pid, test_participant=test_participant, valid_participant=valid_participant)

        fno = _per_person_no(sno)
        audio_file = audio_root / f"audio_{pid}_{fno:02d}.wav"
        video_file = video_root / f"video_{pid}_{fno:02d}.mp4"
        ppg_file = ppg_root / f"ppg_{pid}_{fno:02d}.csv"

        if not audio_file.is_file():
            warnings.append(f"  [MISSING] Audio: {audio_file}")
        if not video_file.is_file():
            warnings.append(f"  [MISSING] Video: {video_file}")
        if not ppg_file.is_file():
            warnings.append(f"  [MISSING] PPG:   {ppg_file}")

        samples.append(
            CustomSample(
                participant_id=pid,
                sample_no=sno,
                label_id=label_id,
                label_name=label_name,
                split=split,
                audio_path=audio_file,
                video_path=video_file,
                ppg_csv_path=ppg_file,
            )
        )

    return samples, warnings


def verify_files(
    labels_csv: Path,
    *,
    audio_root: Path,
    video_root: Path,
    ppg_root: Path,
) -> dict[str, Any]:
    """Check which files exist and which are missing. Does not write any output."""
    rows = load_labels_csv(labels_csv)
    total = len(rows)
    missing_audio: list[str] = []
    missing_video: list[str] = []
    missing_ppg: list[str] = []

    for row in rows:
        pid = int(row["participant_id"])
        sno = int(row["sample_no"])
        fno = _per_person_no(sno)
        if not (audio_root / f"audio_{pid}_{fno:02d}.wav").is_file():
            missing_audio.append(f"audio_{pid}_{fno:02d}.wav")
        if not (video_root / f"video_{pid}_{fno:02d}.mp4").is_file():
            missing_video.append(f"video_{pid}_{fno:02d}.mp4")
        if not (ppg_root / f"ppg_{pid}_{fno:02d}.csv").is_file():
            missing_ppg.append(f"ppg_{pid}_{fno:02d}.csv")

    return {
        "total_samples": total,
        "missing_audio": missing_audio,
        "missing_video": missing_video,
        "missing_ppg": missing_ppg,
        "audio_present": total - len(missing_audio),
        "video_present": total - len(missing_video),
        "ppg_present": total - len(missing_ppg),
    }


def write_manifests(
    samples: list[CustomSample],
    *,
    output_root: Path,
    audio_root: Path,
    video_root: Path,
    ppg_root: Path,
    repo_root: Path,
    skip_missing_av: bool = True,
) -> dict[str, int]:
    """Write train/valid/test manifest CSVs to output_root.

    Args:
        samples: Full sample list from build_sample_list().
        output_root: Directory where av_train.csv, av_valid.csv, av_test.csv are written.
        audio_root: Root for audio .wav files (used to compute relative paths).
        video_root: Root for video .mp4 files.
        ppg_root: Root for PPG .csv files.
        repo_root: Project root; paths in the manifest are made relative to this.
        skip_missing_av: If True, rows with missing audio OR video are omitted from
            the manifest. PPG absence is never a reason to skip (it loads as zeros).

    Returns:
        Dict of {split: row_count} for each written manifest.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    skipped = 0

    for sample in samples:
        audio_missing = not sample.audio_path.is_file()
        video_missing = not sample.video_path.is_file()

        if skip_missing_av and (audio_missing or video_missing):
            skipped += 1
            continue

        audio_rel = _make_relative(sample.audio_path, repo_root)
        video_rel = _make_relative(sample.video_path, repo_root)
        ppg_rel = _make_relative(sample.ppg_csv_path, repo_root)

        row: dict[str, str] = {
            "av_key": sample.av_key,
            "split": sample.split,
            "actor_id": str(sample.participant_id),
            "label_id": str(sample.label_id),
            "label_name": sample.label_name,
            "emotion_code": sample.label_name,
            "audio_sample_id": sample.audio_sample_id,
            "audio_path": audio_rel,
            "video_sample_id": sample.video_sample_id,
            "video_path": video_rel,
            "ppg_csv_path": ppg_rel,
        }
        split_rows[sample.split].append(row)

    counts: dict[str, int] = {}
    for split, rows in split_rows.items():
        rows_sorted = sorted(rows, key=lambda r: r["av_key"])
        out_path = output_root / f"av_{split}.csv"
        _write_csv(out_path, rows_sorted, MANIFEST_FIELDNAMES)
        counts[split] = len(rows_sorted)

    if skipped:
        counts["skipped_missing_av"] = skipped
    return counts


def summarize_split(
    samples: list[CustomSample],
    split: str,
) -> dict[str, Any]:
    """Return label distribution for a given split."""
    split_samples = [s for s in samples if s.split == split]
    distribution: dict[str, int] = {name: 0 for name in CUSTOM_LABEL_ORDER}
    for sample in split_samples:
        distribution[sample.label_name] += 1
    return {
        "split": split,
        "total": len(split_samples),
        "label_distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
