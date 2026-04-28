"""Preprocessing utilities.

Keep package import lightweight so evaluation scripts can import
``src.preprocess.ppg_features`` without pulling optional heavy dependencies
such as ``torch`` or ``cv2`` at package import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PPG_FEATURE_NAMES_ALL": ("src.preprocess.ppg_features", "PPG_FEATURE_NAMES_ALL"),
    "PPG_FEATURE_NAMES_FREQ": ("src.preprocess.ppg_features", "PPG_FEATURE_NAMES_FREQ"),
    "PPG_FEATURE_NAMES_TIME": ("src.preprocess.ppg_features", "PPG_FEATURE_NAMES_TIME"),
    "extract_ppg_features": ("src.preprocess.ppg_features", "extract_ppg_features"),
    "features_to_array": ("src.preprocess.ppg_features", "features_to_array"),
    "CUSTOM_LABEL_ORDER": ("src.preprocess.custom_manifests", "CUSTOM_LABEL_ORDER"),
    "CUSTOM_LABEL_NAME_TO_ID": ("src.preprocess.custom_manifests", "CUSTOM_LABEL_NAME_TO_ID"),
    "MANIFEST_FIELDNAMES": ("src.preprocess.custom_manifests", "MANIFEST_FIELDNAMES"),
    "build_sample_list": ("src.preprocess.custom_manifests", "build_sample_list"),
    "verify_files": ("src.preprocess.custom_manifests", "verify_files"),
    "write_manifests": ("src.preprocess.custom_manifests", "write_manifests"),
    "EXPECTED_SPLIT_COUNTS": ("src.preprocess.audio_ecapa", "EXPECTED_SPLIT_COUNTS"),
    "EcapaLayout": ("src.preprocess.audio_ecapa", "EcapaLayout"),
    "build_ecapa_layout": ("src.preprocess.audio_ecapa", "build_ecapa_layout"),
    "build_fusion_ready_audio_tables": ("src.preprocess.audio_ecapa", "build_fusion_ready_audio_tables"),
    "create_local_speechbrain_metadata": ("src.preprocess.audio_ecapa", "create_local_speechbrain_metadata"),
    "detect_missing_artifacts": ("src.preprocess.audio_ecapa", "detect_missing_artifacts"),
    "find_best_checkpoint": ("src.preprocess.audio_ecapa", "find_best_checkpoint"),
    "generate_missing_audio_artifacts": ("src.preprocess.audio_ecapa", "generate_missing_audio_artifacts"),
    "normalize_existing_audio_artifacts": ("src.preprocess.audio_ecapa", "normalize_existing_audio_artifacts"),
    "AUDIO_MANIFEST_FIELDS": ("src.preprocess.manifests", "AUDIO_MANIFEST_FIELDS"),
    "AV_MANIFEST_FIELDS": ("src.preprocess.manifests", "AV_MANIFEST_FIELDS"),
    "VIDEO_MANIFEST_FIELDS": ("src.preprocess.manifests", "VIDEO_MANIFEST_FIELDS"),
    "build_and_write_manifests": ("src.preprocess.manifests", "build_and_write_manifests"),
    "build_audio_manifest_rows": ("src.preprocess.manifests", "build_audio_manifest_rows"),
    "build_av_manifest_rows": ("src.preprocess.manifests", "build_av_manifest_rows"),
    "build_video_manifest_rows": ("src.preprocess.manifests", "build_video_manifest_rows"),
    "write_split_manifest": ("src.preprocess.manifests", "write_split_manifest"),
    "CANONICAL_SPLIT_COUNTS": ("src.preprocess.video_reuse", "CANONICAL_SPLIT_COUNTS"),
    "EXPECTED_SEQUENCE_SHAPE": ("src.preprocess.video_reuse", "EXPECTED_SEQUENCE_SHAPE"),
    "VideoReuseLayout": ("src.preprocess.video_reuse", "VideoReuseLayout"),
    "audit_video_reuse": ("src.preprocess.video_reuse", "audit_video_reuse"),
    "build_fusion_ready_video_tables": ("src.preprocess.video_reuse", "build_fusion_ready_video_tables"),
    "build_video_reuse_layout": ("src.preprocess.video_reuse", "build_video_reuse_layout"),
    "normalize_video_artifacts": ("src.preprocess.video_reuse", "normalize_video_artifacts"),
    "Actor15BackfillLayout": ("src.preprocess.video_actor15_backfill", "Actor15BackfillLayout"),
    "build_actor15_backfill_layout": ("src.preprocess.video_actor15_backfill", "build_actor15_backfill_layout"),
    "run_actor15_backfill": ("src.preprocess.video_actor15_backfill", "run_actor15_backfill"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

