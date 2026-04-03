from __future__ import annotations

import argparse

from src.preprocess.complete_video_reuse import print_audit_summary
from src.preprocess.video_actor15_backfill import (
    build_actor15_backfill_layout,
    run_actor15_backfill,
)
from src.preprocess.video_reuse import (
    audit_video_reuse,
    build_fusion_ready_video_tables,
    build_video_reuse_layout,
    normalize_video_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill the missing actor-15 reusable video artifacts.")
    parser.add_argument(
        "command",
        choices=("backfill", "refresh-video-reuse", "complete"),
        help="Operation to run.",
    )
    args = parser.parse_args()

    if args.command == "backfill":
        summary = run_actor15_backfill(build_actor15_backfill_layout())
        print(f"device: {summary['device']}")
        print(f"predictions_written: {summary['predictions_written']}")
        print(f"sequences_written: {summary['sequences_written']}")
        print(f"prediction_index_path: {summary['prediction_index_path']}")
        print(f"sequence_index_path: {summary['sequence_index_path']}")
        return 0

    if args.command == "refresh-video-reuse":
        layout = build_video_reuse_layout()
        result = normalize_video_artifacts(layout)
        split_paths = build_fusion_ready_video_tables(layout)
        print_audit_summary(result["audit"])
        for split, path in split_paths.items():
            print(f"{split}: {path}")
        return 0

    backfill_summary = run_actor15_backfill(build_actor15_backfill_layout())
    print(f"device: {backfill_summary['device']}")
    print(f"predictions_written: {backfill_summary['predictions_written']}")
    print(f"sequences_written: {backfill_summary['sequences_written']}")

    layout = build_video_reuse_layout()
    result = normalize_video_artifacts(layout)
    split_paths = build_fusion_ready_video_tables(layout)
    print_audit_summary(result["audit"])
    for split, path in split_paths.items():
        print(f"{split}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
