from __future__ import annotations

import argparse
from pathlib import Path

from .manifests import build_and_write_manifests


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical RAVDESS audio, video, and AV manifests while supporting "
            "either canonical flattened raw roots or the currently wrapped local layout."
        )
    )
    parser.add_argument(
        "mode",
        choices=("all", "audio", "video", "av"),
        nargs="?",
        default="all",
        help="Which manifest set to write.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/raw/ravdess/audio_speech"),
        help="Raw audio root. Supports Actor_XX directly or Audio_Speech_Actors_01-24/Actor_XX.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/raw/ravdess/video_speech"),
        help="Raw video root. Supports Actor_XX directly or Video_Speech_Actor_XX/Actor_XX.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/manifests"),
        help="Root directory for canonical audio/video/av manifest folders.",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Write AV manifests from the audio/video intersection instead of failing on missing pairs.",
    )
    args = parser.parse_args()

    outputs = build_and_write_manifests(
        audio_root=args.audio_root,
        video_root=args.video_root,
        output_root=args.output_root,
        strict_av=not args.allow_unpaired,
        mode=args.mode,
    )

    for manifest_type, split_paths in outputs.items():
        print(f"{manifest_type}:")
        for split, path in split_paths.items():
            print(f"  {split}: {path.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
