"""CLI for building manifest CSVs from the 200-sample custom dataset.

Usage — verify files only (no writing):
    python -m src.preprocess.build_custom_manifests \\
        --labels dataset_200_labeled.csv \\
        --audio-root data/raw/custom/audio \\
        --video-root data/raw/custom/video \\
        --ppg-root data/raw/custom/ppg \\
        --verify-only

Usage — build manifests (default LOSO: test=4, valid=3, train=1+2):
    python -m src.preprocess.build_custom_manifests \\
        --labels dataset_200_labeled.csv \\
        --audio-root data/raw/custom/audio \\
        --video-root data/raw/custom/video \\
        --ppg-root data/raw/custom/ppg \\
        --output-root data/manifests/custom

Usage — different LOSO fold (test=3, valid=2, train=1+4):
    python -m src.preprocess.build_custom_manifests \\
        --labels dataset_200_labeled.csv \\
        --output-root data/manifests/custom_fold2 \\
        --test-participant 3 \\
        --valid-participant 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocess.custom_manifests import (
    CUSTOM_LABEL_ORDER,
    build_sample_list,
    summarize_split,
    verify_files,
    write_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build manifest CSVs for the custom 200-sample multimodal emotion dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        default="dataset_200_labeled.csv",
        help="Path to the labeled CSV (default: dataset_200_labeled.csv in cwd).",
    )
    parser.add_argument(
        "--audio-root",
        default="data/raw/custom/audio",
        help="Directory containing audio_<p>_<n>.wav files.",
    )
    parser.add_argument(
        "--video-root",
        default="data/raw/custom/video",
        help="Directory containing video_<p>_<n>.mp4 files.",
    )
    parser.add_argument(
        "--ppg-root",
        default="data/raw/custom/ppg",
        help="Directory containing ppg_<p>_<n>.csv files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/manifests/custom",
        help="Directory where av_train.csv / av_valid.csv / av_test.csv are written.",
    )
    parser.add_argument(
        "--test-participant",
        type=int,
        default=4,
        help="Participant ID assigned to the test split (default: 4).",
    )
    parser.add_argument(
        "--valid-participant",
        type=int,
        default=3,
        help="Participant ID assigned to the valid split (default: 3).",
    )
    parser.add_argument(
        "--include-missing-av",
        action="store_true",
        help=(
            "Write manifest rows even when audio or video files are absent. "
            "The dataset loader will raise an error at training time for missing files. "
            "By default, rows with missing audio or video are silently skipped."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check file existence and print a report without writing any manifests.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    labels_csv = _resolve(repo_root, args.labels)
    audio_root = _resolve(repo_root, args.audio_root)
    video_root = _resolve(repo_root, args.video_root)
    ppg_root = _resolve(repo_root, args.ppg_root)
    output_root = _resolve(repo_root, args.output_root)

    if not labels_csv.is_file():
        print(f"ERROR: Labels CSV not found: {labels_csv}", file=sys.stderr)
        return 1

    if args.test_participant == args.valid_participant:
        print(
            f"ERROR: --test-participant and --valid-participant cannot be the same ({args.test_participant}).",
            file=sys.stderr,
        )
        return 1

    # ---- verify-only mode ------------------------------------------------
    if args.verify_only:
        print("=" * 60)
        print("FILE VERIFICATION REPORT")
        print("=" * 60)
        report = verify_files(
            labels_csv,
            audio_root=audio_root,
            video_root=video_root,
            ppg_root=ppg_root,
        )
        total = report["total_samples"]
        print(f"Total samples in labels CSV : {total}")
        print(f"Audio files present         : {report['audio_present']}/{total}")
        print(f"Video files present         : {report['video_present']}/{total}")
        print(f"PPG files present           : {report['ppg_present']}/{total}")

        if report["missing_audio"]:
            print(f"\nMissing audio ({len(report['missing_audio'])}):")
            for name in report["missing_audio"][:20]:
                print(f"  {name}")
            if len(report["missing_audio"]) > 20:
                print(f"  ... and {len(report['missing_audio']) - 20} more")

        if report["missing_video"]:
            print(f"\nMissing video ({len(report['missing_video'])}):")
            for name in report["missing_video"][:20]:
                print(f"  {name}")
            if len(report["missing_video"]) > 20:
                print(f"  ... and {len(report['missing_video']) - 20} more")

        if report["missing_ppg"]:
            print(f"\nMissing PPG ({len(report['missing_ppg'])}):")
            for name in report["missing_ppg"][:20]:
                print(f"  {name}")
            if len(report["missing_ppg"]) > 20:
                print(f"  ... and {len(report['missing_ppg']) - 20} more")

        all_present = (
            not report["missing_audio"]
            and not report["missing_video"]
            and not report["missing_ppg"]
        )
        if all_present:
            print("\nAll files are present. Ready to build manifests.")
        else:
            n_missing_av = len(report["missing_audio"]) + len(report["missing_video"])
            print(f"\n{n_missing_av} required file(s) missing (audio/video).")
            if report["missing_ppg"]:
                print(f"{len(report['missing_ppg'])} PPG file(s) missing (will load as zero vectors).")
        return 0

    # ---- manifest-building mode ------------------------------------------
    print("=" * 60)
    print("BUILDING CUSTOM DATASET MANIFESTS")
    print("=" * 60)
    print(f"Labels CSV     : {labels_csv}")
    print(f"Audio root     : {audio_root}")
    print(f"Video root     : {video_root}")
    print(f"PPG root       : {ppg_root}")
    print(f"Output root    : {output_root}")
    print(f"Split (LOSO)   : test=P{args.test_participant}, valid=P{args.valid_participant}")
    all_participants = {1, 2, 3, 4}
    train_participants = sorted(
        all_participants - {args.test_participant, args.valid_participant}
    )
    print(f"                 train=P{'P'.join(str(p) for p in train_participants)}")

    samples, warnings = build_sample_list(
        labels_csv,
        audio_root=audio_root,
        video_root=video_root,
        ppg_root=ppg_root,
        test_participant=args.test_participant,
        valid_participant=args.valid_participant,
    )

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(w)

    counts = write_manifests(
        samples,
        output_root=output_root,
        audio_root=audio_root,
        video_root=video_root,
        ppg_root=ppg_root,
        repo_root=repo_root,
        skip_missing_av=not args.include_missing_av,
    )

    print(f"\nManifests written to: {output_root}")
    for split in ("train", "valid", "test"):
        summary = summarize_split(
            [s for s in samples if counts.get("skipped_missing_av", 0) == 0 or s.audio_path.is_file()],
            split,
        )
        n = counts.get(split, 0)
        print(f"\n  {split:5s}: {n} rows")
        for label_name in CUSTOM_LABEL_ORDER:
            count = summary["label_distribution"].get(label_name, 0)
            if count > 0:
                print(f"           {label_name:<10s}: {count}")

    if counts.get("skipped_missing_av", 0):
        print(
            f"\n  NOTE: {counts['skipped_missing_av']} sample(s) skipped due to missing audio/video."
        )
        print("  Use --include-missing-av to include them anyway.")

    total_written = counts.get("train", 0) + counts.get("valid", 0) + counts.get("test", 0)
    print(f"\nTotal rows written: {total_written}")
    return 0


def _resolve(repo_root: Path, path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


if __name__ == "__main__":
    raise SystemExit(main())
