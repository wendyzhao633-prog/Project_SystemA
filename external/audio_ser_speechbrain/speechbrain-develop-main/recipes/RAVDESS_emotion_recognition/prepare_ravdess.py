#!/usr/bin/env python3
"""
Data preparation script for RAVDESS Speech Emotion Recognition (SER).

Scans the RAVDESS Audio Speech dataset, filters for speech-only audio clips
(modality=03, vocal_channel=01), applies the fixed actor-independent split
defined in metadata/split_by_actor.json, and produces SpeechBrain-compatible
JSON manifests in the metadata/ directory.

RAVDESS filename format:  MM-VV-EE-II-SS-RR-AA.wav
  MM  modality        03 = audio-only
  VV  vocal channel   01 = speech
  EE  emotion         01-08
  II  intensity       01 = normal, 02 = strong
  SS  statement       01 or 02
  RR  repetition      01 or 02
  AA  actor           01-24

Fixed emotion code → label_id mapping (0-indexed, never reorder):
  01 neutral  → 0
  02 calm     → 1
  03 happy    → 2
  04 sad      → 3
  05 angry    → 4
  06 fearful  → 5
  07 disgust  → 6
  08 surprise → 7

Usage
-----
  python recipes/RAVDESS_emotion_recognition/prepare_ravdess.py ^
      --data_root "C:/Users/NannanLi/Desktop/Ravdess/Audio_Speech_Actors_01-24" ^
      --metadata_dir "C:/Users/NannanLi/Desktop/speechbrain-develop/metadata"
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import soundfile as sf

# ---------------------------------------------------------------------------
# Fixed label tables  (source of truth — also written to label_map.json)
# ---------------------------------------------------------------------------
EMOTION_CODE_TO_NAME = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprise",
}

# Fixed 0-indexed order — DO NOT change, it must match label_map.json and
# the out_n_neurons ordering used during training.
LABEL_NAME_TO_ID = {
    "neutral":  0,
    "calm":     1,
    "happy":    2,
    "sad":      3,
    "angry":    4,
    "fearful":  5,
    "disgust":  6,
    "surprise": 7,
}

ID_TO_LABEL_NAME = {v: k for k, v in LABEL_NAME_TO_ID.items()}
NUM_CLASSES = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ravdess_filename(stem: str):
    """Parse a RAVDESS filename stem (without .wav).

    Returns a dict with modality, vocal_channel, emotion_code, intensity_code,
    statement_code, repetition_code, actor (int), or None if format is wrong.
    """
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    modality, vocal_ch, emotion, intensity, statement, repetition, actor_str = parts
    try:
        actor = int(actor_str)
    except ValueError:
        return None
    return {
        "modality":       modality,
        "vocal_channel":  vocal_ch,
        "emotion_code":   emotion,
        "intensity_code": intensity,
        "statement_code": statement,
        "repetition_code": repetition,
        "actor":          actor,
    }


def get_audio_duration(wav_path: str) -> float:
    """Return duration in seconds using soundfile.info (no full decoding needed)."""
    info = sf.info(wav_path)
    return info.duration


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_ravdess(data_root: str, metadata_dir: str, split_file: str = None):
    """Scan RAVDESS data and write train/valid/test JSON manifests.

    Parameters
    ----------
    data_root : str
        Path to the RAVDESS Audio_Speech_Actors_01-24 directory.
    metadata_dir : str
        Directory where manifest files will be saved.
    split_file : str, optional
        Path to split_by_actor.json.
        Defaults to ``{metadata_dir}/split_by_actor.json``.
    """
    data_root = Path(data_root)
    metadata_dir = Path(metadata_dir)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}\n"
            "Please verify --data_root points to the Actor_01 … Actor_24 parent directory."
        )

    os.makedirs(metadata_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load actor split
    # ------------------------------------------------------------------
    if split_file is None:
        split_file = metadata_dir / "split_by_actor.json"
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            "Run with --metadata_dir pointing to a directory that contains split_by_actor.json,\n"
            "or pass --split_file explicitly."
        )

    with open(split_file, encoding="utf-8") as f:
        split_config = json.load(f)

    train_actors = set(split_config["train"])
    val_actors   = set(split_config["val"])
    test_actors  = set(split_config["test"])

    print(f"Split config loaded from {split_file}")
    print(f"  train actors: {sorted(train_actors)}")
    print(f"  val   actors: {sorted(val_actors)}")
    print(f"  test  actors: {sorted(test_actors)}")

    # ------------------------------------------------------------------
    # Scan wav files
    # ------------------------------------------------------------------
    all_samples = []
    skipped = 0

    actor_dirs = sorted(d for d in data_root.iterdir()
                        if d.is_dir() and d.name.startswith("Actor_"))
    if not actor_dirs:
        raise ValueError(
            f"No 'Actor_XX' subdirectories found in {data_root}.\n"
            "Please check --data_root."
        )

    print(f"\nScanning {len(actor_dirs)} Actor directories in {data_root} ...")

    for actor_dir in actor_dirs:
        wav_files = sorted(actor_dir.glob("*.wav"))
        for wav_file in wav_files:
            stem = wav_file.stem
            parsed = parse_ravdess_filename(stem)

            if parsed is None:
                print(f"  SKIP (bad filename): {wav_file.name}", file=sys.stderr)
                skipped += 1
                continue

            # Keep only audio-only speech files
            if parsed["modality"] != "03" or parsed["vocal_channel"] != "01":
                skipped += 1
                continue

            emotion_code = parsed["emotion_code"]
            if emotion_code not in EMOTION_CODE_TO_NAME:
                print(
                    f"  SKIP (unknown emotion code {emotion_code}): {wav_file.name}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            label_name = EMOTION_CODE_TO_NAME[emotion_code]
            label_id   = LABEL_NAME_TO_ID[label_name]

            try:
                duration = get_audio_duration(str(wav_file))
            except Exception as exc:
                print(
                    f"  SKIP (cannot read audio info): {wav_file.name} — {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            actor_id = parsed["actor"]

            # Assign split
            if actor_id in train_actors:
                split = "train"
            elif actor_id in val_actors:
                split = "valid"
            elif actor_id in test_actors:
                split = "test"
            else:
                print(
                    f"  SKIP (actor {actor_id} not in any split): {wav_file.name}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            # Use forward slashes for cross-platform compatibility
            wav_path = str(wav_file).replace("\\", "/")

            all_samples.append({
                "sample_id":      stem,
                "wav":            wav_path,
                "duration":       round(duration, 4),
                "split":          split,
                "actor":          actor_id,
                "emotion_code":   emotion_code,
                "intensity_code": parsed["intensity_code"],
                "statement_code": parsed["statement_code"],
                "repetition_code": parsed["repetition_code"],
                "label_name":     label_name,
                "label_id":       label_id,
            })

    if not all_samples:
        raise ValueError(
            f"No valid RAVDESS speech-only audio files found in {data_root}.\n"
            "Expected files like 03-01-XX-XX-XX-XX-XX.wav inside Actor_XX subdirs."
        )

    # ------------------------------------------------------------------
    # Write per-split JSON manifests (SpeechBrain DynamicItemDataset format)
    # The key is sample_id; each value contains all other fields.
    # ------------------------------------------------------------------
    split_samples = {
        "train": [s for s in all_samples if s["split"] == "train"],
        "valid": [s for s in all_samples if s["split"] == "valid"],
        "test":  [s for s in all_samples if s["split"] == "test"],
    }

    for split_name, samples in split_samples.items():
        manifest = {}
        for s in samples:
            entry = {k: v for k, v in s.items() if k not in ("sample_id", "split")}
            manifest[s["sample_id"]] = entry
        out_path = metadata_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} manifest ({len(samples)} samples): {out_path}")

    # ------------------------------------------------------------------
    # Write all_samples.csv (handy for inspection)
    # ------------------------------------------------------------------
    csv_path = metadata_dir / "all_samples.csv"
    fieldnames = [
        "sample_id", "wav", "duration", "split", "actor",
        "emotion_code", "intensity_code", "statement_code",
        "repetition_code", "label_name", "label_id",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)
    print(f"Saved all_samples CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Print summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RAVDESS preparation complete!")
    print(f"  Total valid files : {len(all_samples)}")
    print(f"  Skipped files     : {skipped}")
    print()

    label_order = list(LABEL_NAME_TO_ID.keys())  # preserved insertion order
    for split_name, samples in split_samples.items():
        counts = {name: 0 for name in label_order}
        for s in samples:
            counts[s["label_name"]] += 1
        print(f"  {split_name:5s} ({len(samples):4d} samples):")
        for name in label_order:
            lid = LABEL_NAME_TO_ID[name]
            print(f"    id={lid}  {name:10s}: {counts[name]:4d}")
        print()

    print("=" * 60)
    return all_samples


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare RAVDESS dataset manifests for 8-class SER baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        default="C:/Users/NannanLi/Desktop/Ravdess/Audio_Speech_Actors_01-24",
        help="Path to RAVDESS Audio_Speech_Actors_01-24 directory",
    )
    parser.add_argument(
        "--metadata_dir",
        default="C:/Users/NannanLi/Desktop/speechbrain-develop/metadata",
        help="Directory where train.json / valid.json / test.json will be saved",
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help=(
            "Path to split_by_actor.json "
            "(defaults to {metadata_dir}/split_by_actor.json)"
        ),
    )
    args = parser.parse_args()
    prepare_ravdess(args.data_root, args.metadata_dir, args.split_file)


if __name__ == "__main__":
    main()
