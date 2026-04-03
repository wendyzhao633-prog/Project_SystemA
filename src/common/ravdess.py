from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

LABEL_ORDER = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprise",
)

EMOTION_CODE_TO_LABEL_ID = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
    "05": 4,
    "06": 5,
    "07": 6,
    "08": 7,
}

EMOTION_CODE_TO_LABEL_NAME = {
    code: LABEL_ORDER[label_id]
    for code, label_id in EMOTION_CODE_TO_LABEL_ID.items()
}

RAVDESS_SPLIT_ORDER = ("train", "valid", "test")
_STEM_RE = re.compile(r"^\d{2}(?:-\d{2}){6}$")
_ACTOR_DIR_RE = re.compile(r"^Actor_(\d{2})$")


def split_for_actor(actor_id: int) -> str:
    if 1 <= actor_id <= 16:
        return "train"
    if 17 <= actor_id <= 20:
        return "valid"
    if 21 <= actor_id <= 24:
        return "test"
    raise ValueError(f"Actor id must be in 1..24, got {actor_id}.")


def build_av_key(stem_or_filename: str | Path | "RavdessFilename") -> str:
    if isinstance(stem_or_filename, RavdessFilename):
        return stem_or_filename.av_key

    stem = Path(stem_or_filename).stem if isinstance(stem_or_filename, Path) else str(stem_or_filename)
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 dash-separated fields in stem '{stem}'.")
    return "-".join(parts[1:])


@dataclass(frozen=True)
class RavdessFilename:
    stem: str
    modality: str
    vocal_channel: str
    emotion_code: str
    intensity_code: str
    statement_code: str
    repetition_code: str
    actor_code: str

    @property
    def actor_id(self) -> int:
        return int(self.actor_code)

    @property
    def label_id(self) -> int:
        return EMOTION_CODE_TO_LABEL_ID[self.emotion_code]

    @property
    def label_name(self) -> str:
        return EMOTION_CODE_TO_LABEL_NAME[self.emotion_code]

    @property
    def split(self) -> str:
        return split_for_actor(self.actor_id)

    @property
    def av_key(self) -> str:
        return "-".join(
            [
                self.vocal_channel,
                self.emotion_code,
                self.intensity_code,
                self.statement_code,
                self.repetition_code,
                self.actor_code,
            ]
        )

    @property
    def actor_dirname(self) -> str:
        return f"Actor_{self.actor_code}"


def parse_ravdess_stem(stem: str | Path) -> RavdessFilename:
    stem_text = Path(stem).stem if isinstance(stem, Path) else str(stem)
    if not _STEM_RE.match(stem_text):
        raise ValueError(
            "RAVDESS stem must match MM-VV-EE-II-SS-RR-AA with two-digit fields; "
            f"got '{stem_text}'."
        )

    parts = stem_text.split("-")
    modality, vocal_channel, emotion_code, intensity_code, statement_code, repetition_code, actor_code = parts

    if modality not in {"01", "02", "03"}:
        raise ValueError(f"Unsupported modality code '{modality}' in stem '{stem_text}'.")
    if vocal_channel not in {"01", "02"}:
        raise ValueError(f"Unsupported vocal channel code '{vocal_channel}' in stem '{stem_text}'.")
    if emotion_code not in EMOTION_CODE_TO_LABEL_ID:
        raise ValueError(f"Unsupported emotion code '{emotion_code}' in stem '{stem_text}'.")
    if intensity_code not in {"01", "02"}:
        raise ValueError(f"Unsupported intensity code '{intensity_code}' in stem '{stem_text}'.")
    if statement_code not in {"01", "02"}:
        raise ValueError(f"Unsupported statement code '{statement_code}' in stem '{stem_text}'.")
    if repetition_code not in {"01", "02"}:
        raise ValueError(f"Unsupported repetition code '{repetition_code}' in stem '{stem_text}'.")

    actor_id = int(actor_code)
    split_for_actor(actor_id)

    return RavdessFilename(
        stem=stem_text,
        modality=modality,
        vocal_channel=vocal_channel,
        emotion_code=emotion_code,
        intensity_code=intensity_code,
        statement_code=statement_code,
        repetition_code=repetition_code,
        actor_code=actor_code,
    )


def actor_id_from_dirname(dirname: str) -> int:
    match = _ACTOR_DIR_RE.match(dirname)
    if not match:
        raise ValueError(f"Expected actor directory name like 'Actor_01', got '{dirname}'.")
    return int(match.group(1))
