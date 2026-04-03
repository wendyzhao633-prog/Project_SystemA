from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable

from src.common.ravdess import (
    RAVDESS_SPLIT_ORDER,
    actor_id_from_dirname,
    parse_ravdess_stem,
)

AUDIO_MANIFEST_FIELDS = [
    "sample_id",
    "stem",
    "av_key",
    "wav_path",
    "actor_id",
    "split",
    "emotion_code",
    "label_id",
    "label_name",
    "intensity_code",
    "statement_code",
    "repetition_code",
]

VIDEO_MANIFEST_FIELDS = [
    "sample_id",
    "stem",
    "av_key",
    "video_path",
    "actor_id",
    "split",
    "emotion_code",
    "label_id",
    "label_name",
    "intensity_code",
    "statement_code",
    "repetition_code",
]

AV_MANIFEST_FIELDS = [
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
]

_VIDEO_WRAPPER_RE = re.compile(r"^Video_Speech_Actor_(\d{2})$")


def discover_audio_actor_dirs(audio_root: Path) -> dict[int, Path]:
    root = Path(audio_root)
    direct_actor_dirs = _collect_actor_dirs(root.iterdir())
    if direct_actor_dirs:
        return direct_actor_dirs

    wrapper = root / "Audio_Speech_Actors_01-24"
    if wrapper.is_dir():
        wrapped_actor_dirs = _collect_actor_dirs(wrapper.iterdir())
        if wrapped_actor_dirs:
            return wrapped_actor_dirs

    raise ValueError(
        "Could not discover audio actor directories under "
        f"'{root}'. Expected direct Actor_XX folders or Audio_Speech_Actors_01-24/Actor_XX."
    )


def discover_video_actor_dirs(video_root: Path) -> dict[int, Path]:
    root = Path(video_root)
    direct_actor_dirs = _collect_actor_dirs(root.iterdir())
    if direct_actor_dirs:
        return direct_actor_dirs

    wrapped_actor_dirs: dict[int, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        wrapper_match = _VIDEO_WRAPPER_RE.match(child.name)
        if not wrapper_match:
            continue

        wrapper_actor_id = int(wrapper_match.group(1))
        inner_actor_dir = child / f"Actor_{wrapper_actor_id:02d}"
        if not inner_actor_dir.is_dir():
            raise ValueError(
                f"Expected wrapped video directory '{child}' to contain '{inner_actor_dir.name}'."
            )
        wrapped_actor_dirs[wrapper_actor_id] = inner_actor_dir

    if wrapped_actor_dirs:
        return dict(sorted(wrapped_actor_dirs.items()))

    raise ValueError(
        "Could not discover video actor directories under "
        f"'{root}'. Expected direct Actor_XX folders or Video_Speech_Actor_XX/Actor_XX."
    )


def build_audio_manifest_rows(audio_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actor_id, actor_dir in discover_audio_actor_dirs(audio_root).items():
        for audio_file in _iter_media_files(actor_dir, ".wav"):
            parsed = parse_ravdess_stem(audio_file.stem)
            _validate_sample(parsed, actor_id=actor_id, expected_modality="03", path=audio_file)
            rows.append(
                {
                    "sample_id": parsed.stem,
                    "stem": parsed.stem,
                    "av_key": parsed.av_key,
                    "wav_path": audio_file.as_posix(),
                    "actor_id": parsed.actor_id,
                    "split": parsed.split,
                    "emotion_code": parsed.emotion_code,
                    "label_id": parsed.label_id,
                    "label_name": parsed.label_name,
                    "intensity_code": parsed.intensity_code,
                    "statement_code": parsed.statement_code,
                    "repetition_code": parsed.repetition_code,
                }
            )
    return sorted(rows, key=lambda row: (row["actor_id"], row["stem"]))


def build_video_manifest_rows(video_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actor_id, actor_dir in discover_video_actor_dirs(video_root).items():
        for video_file in _iter_media_files(actor_dir, ".mp4"):
            parsed = parse_ravdess_stem(video_file.stem)
            # Some local RAVDESS video folders contain both full-AV (01) and
            # video-only (02) MP4s. Keep the canonical video-only files only.
            if parsed.modality == "01":
                continue
            _validate_sample(parsed, actor_id=actor_id, expected_modality="02", path=video_file)
            rows.append(
                {
                    "sample_id": parsed.stem,
                    "stem": parsed.stem,
                    "av_key": parsed.av_key,
                    "video_path": video_file.as_posix(),
                    "actor_id": parsed.actor_id,
                    "split": parsed.split,
                    "emotion_code": parsed.emotion_code,
                    "label_id": parsed.label_id,
                    "label_name": parsed.label_name,
                    "intensity_code": parsed.intensity_code,
                    "statement_code": parsed.statement_code,
                    "repetition_code": parsed.repetition_code,
                }
            )
    return sorted(rows, key=lambda row: (row["actor_id"], row["stem"]))


def build_av_manifest_rows(
    audio_rows: Iterable[dict[str, object]],
    video_rows: Iterable[dict[str, object]],
    *,
    strict: bool = True,
) -> list[dict[str, object]]:
    audio_by_key = _index_unique(audio_rows, dataset_name="audio")
    video_by_key = _index_unique(video_rows, dataset_name="video")

    audio_keys = set(audio_by_key)
    video_keys = set(video_by_key)
    missing_from_video = sorted(audio_keys - video_keys)
    missing_from_audio = sorted(video_keys - audio_keys)

    if strict and (missing_from_video or missing_from_audio):
        raise ValueError(
            "AV pairing mismatch by av_key. "
            f"audio_only={missing_from_video[:5]} video_only={missing_from_audio[:5]}"
        )

    paired_rows: list[dict[str, object]] = []
    for av_key in sorted(audio_keys & video_keys):
        audio_row = audio_by_key[av_key]
        video_row = video_by_key[av_key]
        _validate_pair_consistency(av_key, audio_row=audio_row, video_row=video_row)
        paired_rows.append(
            {
                "av_key": av_key,
                "split": audio_row["split"],
                "actor_id": audio_row["actor_id"],
                "label_id": audio_row["label_id"],
                "label_name": audio_row["label_name"],
                "emotion_code": audio_row["emotion_code"],
                "audio_sample_id": audio_row["sample_id"],
                "audio_path": audio_row["wav_path"],
                "video_sample_id": video_row["sample_id"],
                "video_path": video_row["video_path"],
            }
        )

    return sorted(paired_rows, key=lambda row: (row["actor_id"], row["av_key"]))


def write_split_manifest(
    rows: Iterable[dict[str, object]],
    *,
    output_dir: Path,
    prefix: str,
    fieldnames: list[str],
) -> dict[str, Path]:
    grouped_rows = {split: [] for split in RAVDESS_SPLIT_ORDER}
    for row in rows:
        split = str(row["split"])
        if split not in grouped_rows:
            raise ValueError(f"Unknown split '{split}' in row {row}.")
        grouped_rows[split].append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, Path] = {}
    for split in RAVDESS_SPLIT_ORDER:
        output_path = output_dir / f"{prefix}_{split}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grouped_rows[split])
        written_paths[split] = output_path

    return written_paths


def build_and_write_manifests(
    *,
    audio_root: Path,
    video_root: Path,
    output_root: Path,
    strict_av: bool = True,
    mode: str = "all",
) -> dict[str, dict[str, Path]]:
    mode = mode.lower()
    valid_modes = {"all", "audio", "video", "av"}
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {sorted(valid_modes)}, got '{mode}'.")

    outputs: dict[str, dict[str, Path]] = {}
    audio_rows: list[dict[str, object]] | None = None
    video_rows: list[dict[str, object]] | None = None

    if mode in {"all", "audio", "av"}:
        audio_rows = build_audio_manifest_rows(audio_root)
    if mode in {"all", "video", "av"}:
        video_rows = build_video_manifest_rows(video_root)

    if mode in {"all", "audio"} and audio_rows is not None:
        outputs["audio"] = write_split_manifest(
            audio_rows,
            output_dir=Path(output_root) / "audio",
            prefix="audio",
            fieldnames=AUDIO_MANIFEST_FIELDS,
        )

    if mode in {"all", "video"} and video_rows is not None:
        outputs["video"] = write_split_manifest(
            video_rows,
            output_dir=Path(output_root) / "video",
            prefix="video",
            fieldnames=VIDEO_MANIFEST_FIELDS,
        )

    if mode in {"all", "av"}:
        if audio_rows is None:
            audio_rows = build_audio_manifest_rows(audio_root)
        if video_rows is None:
            video_rows = build_video_manifest_rows(video_root)
        av_rows = build_av_manifest_rows(audio_rows, video_rows, strict=strict_av)
        outputs["av"] = write_split_manifest(
            av_rows,
            output_dir=Path(output_root) / "av",
            prefix="av",
            fieldnames=AV_MANIFEST_FIELDS,
        )

    return outputs


def _collect_actor_dirs(children: Iterable[Path]) -> dict[int, Path]:
    actor_dirs: dict[int, Path] = {}
    for child in sorted(children):
        if not child.is_dir():
            continue
        try:
            actor_id = actor_id_from_dirname(child.name)
        except ValueError:
            continue
        actor_dirs[actor_id] = child
    return dict(sorted(actor_dirs.items()))


def _iter_media_files(actor_dir: Path, suffix: str) -> list[Path]:
    return sorted(
        path
        for path in actor_dir.iterdir()
        if path.is_file() and path.suffix.lower() == suffix
    )


def _validate_sample(parsed, *, actor_id: int, expected_modality: str, path: Path) -> None:
    if parsed.modality != expected_modality:
        raise ValueError(
            f"Unexpected modality '{parsed.modality}' for '{path}'. Expected '{expected_modality}'."
        )
    if parsed.vocal_channel != "01":
        raise ValueError(f"Found non-speech sample '{path}' with vocal channel '{parsed.vocal_channel}'.")
    if parsed.actor_id != actor_id:
        raise ValueError(
            f"Filename actor '{parsed.actor_id}' does not match directory actor '{actor_id}' for '{path}'."
        )


def _index_unique(rows: Iterable[dict[str, object]], *, dataset_name: str) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        av_key = str(row["av_key"])
        if av_key in indexed:
            raise ValueError(f"Duplicate av_key '{av_key}' found in {dataset_name} rows.")
        indexed[av_key] = dict(row)
    return indexed


def _validate_pair_consistency(
    av_key: str,
    *,
    audio_row: dict[str, object],
    video_row: dict[str, object],
) -> None:
    comparable_fields = ("actor_id", "split", "emotion_code", "label_id", "label_name")
    for field in comparable_fields:
        if audio_row[field] != video_row[field]:
            raise ValueError(
                f"Mismatched {field} for av_key '{av_key}': "
                f"audio={audio_row[field]} video={video_row[field]}"
            )
