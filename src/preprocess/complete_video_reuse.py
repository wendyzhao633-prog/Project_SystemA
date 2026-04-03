from __future__ import annotations

import argparse

from src.preprocess.video_reuse import (
    RAVDESS_SPLIT_ORDER,
    audit_video_reuse,
    build_fusion_ready_video_tables,
    build_video_reuse_layout,
    normalize_video_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and normalize reusable RAVDESS video artifacts.")
    parser.add_argument(
        "command",
        choices=("audit", "normalize", "build-fusion-tables", "complete"),
        help="Operation to run.",
    )
    args = parser.parse_args()

    layout = build_video_reuse_layout()

    if args.command == "audit":
        audit = audit_video_reuse(layout)
        print_audit_summary(audit)
        return 0

    if args.command == "normalize":
        result = normalize_video_artifacts(layout)
        print_audit_summary(result["audit"])
        print(f"audit_path: {result['audit_path']}")
        print(f"missing_artifacts_path: {result['missing_artifacts_path']}")
        return 0

    if args.command == "build-fusion-tables":
        normalize_video_artifacts(layout)
        split_paths = build_fusion_ready_video_tables(layout)
        for split in (*RAVDESS_SPLIT_ORDER, "all"):
            print(f"{split}: {split_paths[split]}")
        return 0

    result = normalize_video_artifacts(layout)
    split_paths = build_fusion_ready_video_tables(layout)
    print_audit_summary(result["audit"])
    print(f"audit_path: {result['audit_path']}")
    print(f"missing_artifacts_path: {result['missing_artifacts_path']}")
    for split in (*RAVDESS_SPLIT_ORDER, "all"):
        print(f"{split}: {split_paths[split]}")
    return 0


def print_audit_summary(audit: dict) -> None:
    print(f"package_root: {audit['package_root']}")
    print(f"frame_predictions_are_frame_level: {audit['predictions']['frame_level']}")
    expected_counts = audit["canonical_expected_counts"]

    for split in RAVDESS_SPLIT_ORDER:
        prediction_info = audit["predictions"]["source_files"][split]
        sequence_info = audit["selected_against_canonical_manifest"][split]
        print(
            f"{split}: frame_rows={prediction_info['frame_rows']} "
            f"unique_video_ids={prediction_info['unique_video_ids']} "
            f"speech_av_keys={prediction_info['unique_speech_av_keys']}"
        )
        print(
            f"{split}: sequence_rows={sequence_info['normalized_sequence_rows']}/{expected_counts[split]} "
            f"prediction_rows={sequence_info['normalized_prediction_rows']}/{expected_counts[split]} "
            f"seq_fallback_to_01={sequence_info['sequence_fallback_to_01']} "
            f"pred_fallback_to_01={sequence_info['prediction_fallback_to_01']}"
        )
        print(
            f"{split}: missing_sequence={sequence_info['missing_sequence_count']} "
            f"missing_prediction={sequence_info['missing_prediction_count']} "
            f"late_fusion_ready_rows={sequence_info['late_fusion_ready_rows']}/{expected_counts[split]}"
        )

    print(f"late_fusion_ready: {audit['late_fusion_ready']}")


if __name__ == "__main__":
    raise SystemExit(main())
