"""Shared dataset contracts and helpers."""

from .ravdess import (
    EMOTION_CODE_TO_LABEL_ID,
    EMOTION_CODE_TO_LABEL_NAME,
    LABEL_ORDER,
    RAVDESS_SPLIT_ORDER,
    RavdessFilename,
    build_av_key,
    parse_ravdess_stem,
    split_for_actor,
)

__all__ = [
    "EMOTION_CODE_TO_LABEL_ID",
    "EMOTION_CODE_TO_LABEL_NAME",
    "LABEL_ORDER",
    "RAVDESS_SPLIT_ORDER",
    "RavdessFilename",
    "build_av_key",
    "parse_ravdess_stem",
    "split_for_actor",
]
