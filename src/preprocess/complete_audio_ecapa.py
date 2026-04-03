from __future__ import annotations

import argparse
from pathlib import Path

from .audio_ecapa import (
    build_ecapa_layout,
    build_fusion_ready_audio_tables,
    detect_missing_artifacts,
    expected_counts_from_manifests,
    find_best_checkpoint,
    generate_missing_audio_artifacts,
    load_audio_manifest_rows,
    normalize_existing_audio_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize delivered ECAPA audio artifacts, regenerate only missing "
            "predictions/embeddings from the best checkpoint, and build fusion-ready tables."
        )
    )
    parser.add_argument(
        "command",
        choices=("normalize-existing", "generate-missing", "build-fusion-tables", "complete", "inventory"),
        nargs="?",
        default="complete",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    args = parser.parse_args()

    layout = build_ecapa_layout(args.repo_root)
    manifest_rows = load_audio_manifest_rows(layout.audio_manifest_dir)
    expected_counts = expected_counts_from_manifests(manifest_rows)
    best_checkpoint = find_best_checkpoint(layout.results_save_root)

    print(f"Best checkpoint: {best_checkpoint.checkpoint_dir.as_posix()}  uar={best_checkpoint.uar:.6f}")

    if args.command == "inventory":
        missing = detect_missing_artifacts(layout, expected_counts=expected_counts)
        for artifact_type, splits in missing.items():
            print(f"{artifact_type}: {', '.join(splits) if splits else 'none'}")
        return 0

    if args.command in {"normalize-existing", "complete"}:
        summary = normalize_existing_audio_artifacts(layout)
        print("Normalized existing artifacts.")
        print(summary)

    if args.command in {"generate-missing", "complete"}:
        summary = generate_missing_audio_artifacts(layout)
        print("Generated missing artifacts.")
        print(summary)

    if args.command in {"build-fusion-tables", "complete"}:
        split_paths = build_fusion_ready_audio_tables(layout, expected_counts=expected_counts)
        print("Fusion-ready tables:")
        for split, path in split_paths.items():
            print(f"  {split}: {path.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
