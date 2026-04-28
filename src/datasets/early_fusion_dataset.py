from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import wave

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from src.common.ravdess import RAVDESS_SPLIT_ORDER, build_av_key


@dataclass(frozen=True)
class EarlyFusionDataConfig:
    train_manifest: Path = Path("data/manifests/av/av_train.csv")
    valid_manifest: Path = Path("data/manifests/av/av_valid.csv")
    test_manifest: Path = Path("data/manifests/av/av_test.csv")
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    audio_num_frames: int = 256
    num_video_frames: int = 5
    image_size: int = 224
    use_face_crop: bool = False
    enable_cache: bool = True
    audio_cache_root: Path = Path("data/processed/av/early_fusion_cache/audio_logmel_sr16000_mel80_t256")
    video_cache_root: Path = Path("data/processed/av/early_fusion_cache/video_frames_5f_224")
    train_audio_gain_jitter_db: float = 0.0
    train_specaug_freq_mask_width: int = 0
    train_specaug_time_mask_width: int = 0
    train_video_random_crop_scale_min: float = 1.0
    train_video_horizontal_flip_prob: float = 0.0
    train_video_brightness_jitter: float = 0.0
    train_video_contrast_jitter: float = 0.0
    # Set to False for non-RAVDESS datasets to skip filename-based av_key validation.
    validate_ravdess_av_key: bool = True

    def manifest_for_split(self, split: str) -> Path:
        if split == "train":
            return self.train_manifest
        if split == "valid":
            return self.valid_manifest
        if split == "test":
            return self.test_manifest
        raise ValueError(f"Unsupported split '{split}'.")


@dataclass(frozen=True)
class EarlyFusionExample:
    av_key: str
    split: str
    actor_id: int
    label_id: int
    label_name: str
    emotion_code: str
    audio_sample_id: str
    video_sample_id: str
    audio_path: Path
    video_path: Path


class EarlyFusionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        repo_root: Path,
        examples: list[EarlyFusionExample],
        config: EarlyFusionDataConfig,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.examples = list(examples)
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            center=True,
            power=2.0,
        )
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self._face_detector = _load_face_detector() if config.use_face_crop else None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        audio_tensor = self._load_audio_tensor(example)
        video_tensor = self._load_video_tensor(example)
        if example.split == "train":
            audio_tensor = _apply_train_audio_augmentations(audio_tensor, self.config)
            video_tensor = _apply_train_video_augmentations(video_tensor, self.config)
        batch: dict[str, Any] = {
            "av_key": example.av_key,
            "split": example.split,
            "actor_id": example.actor_id,
            "label_id": example.label_id,
            "label_name": example.label_name,
            "emotion_code": example.emotion_code,
            "audio_sample_id": example.audio_sample_id,
            "video_sample_id": example.video_sample_id,
            "audio_path": str(example.audio_path),
            "video_path": str(example.video_path),
            "audio_input": audio_tensor,
            "video_input": video_tensor,
        }
        return batch

    def _load_audio_tensor(self, example: EarlyFusionExample) -> torch.Tensor:
        cache_path = self._audio_cache_path(example)
        if self.config.enable_cache and cache_path.is_file():
            cached = _load_cached_array(cache_path)
            if cached is not None:
                return torch.from_numpy(cached).to(torch.float32)

        waveform, sample_rate = _read_wav_tensor(example.audio_path)
        waveform = waveform.to(torch.float32)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            resampler = self._resamplers.get(sample_rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(sample_rate, self.config.sample_rate)
                self._resamplers[sample_rate] = resampler
            waveform = resampler(waveform)

        max_abs = float(waveform.abs().max().item())
        if max_abs > 0.0:
            waveform = waveform / max_abs

        mel = self.mel_transform(waveform)
        mel = torch.log(mel.clamp_min(1e-6))
        mel = _pad_or_truncate_last_dim(mel, self.config.audio_num_frames)

        if self.config.enable_cache:
            _write_cached_array(cache_path, mel.numpy().astype(np.float32))
        return mel

    def _load_video_tensor(self, example: EarlyFusionExample) -> torch.Tensor:
        cache_path = self._video_cache_path(example)
        if self.config.enable_cache and cache_path.is_file():
            cached = _load_cached_array(cache_path)
            if cached is not None:
                return torch.from_numpy(cached).to(torch.float32)

        frames = _read_sampled_video_frames(
            example.video_path,
            num_frames=self.config.num_video_frames,
        )
        processed = [
            _prepare_frame(
                frame,
                image_size=self.config.image_size,
                face_detector=self._face_detector,
            )
            for frame in frames
        ]
        video = np.stack(processed, axis=0).astype(np.float32)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()

        if self.config.enable_cache:
            _write_cached_array(cache_path, video_tensor.numpy().astype(np.float32))
        return video_tensor

    def _audio_cache_path(self, example: EarlyFusionExample) -> Path:
        return self.repo_root / self.config.audio_cache_root / example.split / f"{example.audio_sample_id}.npy"

    def _video_cache_path(self, example: EarlyFusionExample) -> Path:
        return self.repo_root / self.config.video_cache_root / example.split / f"{example.video_sample_id}.npy"


def build_split_datasets(
    *,
    repo_root: Path,
    config: EarlyFusionDataConfig,
    split_limits: dict[str, int | None] | None = None,
) -> dict[str, EarlyFusionDataset]:
    datasets: dict[str, EarlyFusionDataset] = {}
    for split in RAVDESS_SPLIT_ORDER:
        limit = None if split_limits is None else split_limits.get(split)
        examples = load_manifest_examples(
            repo_root / config.manifest_for_split(split),
            repo_root=repo_root,
            limit=limit,
            validate_av_key=config.validate_ravdess_av_key,
        )
        datasets[split] = EarlyFusionDataset(
            repo_root=repo_root,
            examples=examples,
            config=config,
        )
    return datasets


def load_manifest_examples(
    manifest_path: Path,
    *,
    repo_root: Path,
    limit: int | None = None,
    validate_av_key: bool = True,
) -> list[EarlyFusionExample]:
    rows: list[dict[str, str]]
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    examples: list[EarlyFusionExample] = []
    resolved_root = Path(repo_root).resolve()
    for row in rows:
        audio_path = _resolve_repo_path(resolved_root, row["audio_path"])
        video_path = _resolve_repo_path(resolved_root, row["video_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio file for av_key '{row['av_key']}': {audio_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video file for av_key '{row['av_key']}': {video_path}")

        if validate_av_key:
            if build_av_key(row["audio_sample_id"]) != row["av_key"]:
                raise ValueError(f"Audio av_key mismatch in manifest row: {row}")
            if build_av_key(row["video_sample_id"]) != row["av_key"]:
                raise ValueError(f"Video av_key mismatch in manifest row: {row}")

        examples.append(
            EarlyFusionExample(
                av_key=row["av_key"],
                split=row["split"],
                actor_id=int(row["actor_id"]),
                label_id=int(row["label_id"]),
                label_name=row["label_name"],
                emotion_code=row["emotion_code"],
                audio_sample_id=row["audio_sample_id"],
                video_sample_id=row["video_sample_id"],
                audio_path=audio_path,
                video_path=video_path,
            )
        )

    examples.sort(key=lambda item: item.av_key)
    if limit is not None:
        examples = examples[:limit]
    return examples


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _load_cached_array(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except (EOFError, ValueError, OSError):
        path.unlink(missing_ok=True)
        return None


def _write_cached_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    temp_path.replace(path)


def _read_wav_tensor(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    data = _pcm_bytes_to_float(raw, sample_width)
    if data.size % max(channels, 1) != 0:
        raise ValueError(f"Unexpected PCM payload size for '{path}'.")
    waveform = data.reshape(-1, channels).transpose(1, 0)
    return torch.from_numpy(waveform.copy()), sample_rate


def _pcm_bytes_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_width == 3:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            triplets[:, 0].astype(np.int32)
            | (triplets[:, 1].astype(np.int32) << 8)
            | (triplets[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        values = (values ^ sign_mask) - sign_mask
        return values.astype(np.float32) / float(1 << 23)
    if sample_width == 4:
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")


def _pad_or_truncate_last_dim(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    current = tensor.shape[-1]
    if current == target_size:
        return tensor
    if current > target_size:
        return tensor[..., :target_size]
    return F.pad(tensor, (0, target_size - current))


def _apply_train_audio_augmentations(
    mel: torch.Tensor,
    config: EarlyFusionDataConfig,
) -> torch.Tensor:
    augmented = mel.clone()
    if config.train_audio_gain_jitter_db > 0.0:
        delta_db = (torch.rand(1).item() * 2.0 - 1.0) * config.train_audio_gain_jitter_db
        augmented = augmented + float(delta_db * np.log(10.0) / 10.0)
    if config.train_specaug_freq_mask_width > 0:
        augmented = _apply_frequency_mask(augmented, max_width=config.train_specaug_freq_mask_width)
    if config.train_specaug_time_mask_width > 0:
        augmented = _apply_time_mask(augmented, max_width=config.train_specaug_time_mask_width)
    return augmented.contiguous()


def _apply_frequency_mask(mel: torch.Tensor, *, max_width: int) -> torch.Tensor:
    _, num_mels, _ = mel.shape
    width = min(int(torch.randint(low=0, high=max_width + 1, size=(1,)).item()), num_mels)
    if width <= 0 or width >= num_mels:
        return mel
    start = int(torch.randint(low=0, high=num_mels - width + 1, size=(1,)).item())
    masked = mel.clone()
    masked[:, start : start + width, :] = float(masked.mean().item())
    return masked


def _apply_time_mask(mel: torch.Tensor, *, max_width: int) -> torch.Tensor:
    _, _, num_frames = mel.shape
    width = min(int(torch.randint(low=0, high=max_width + 1, size=(1,)).item()), num_frames)
    if width <= 0 or width >= num_frames:
        return mel
    start = int(torch.randint(low=0, high=num_frames - width + 1, size=(1,)).item())
    masked = mel.clone()
    masked[:, :, start : start + width] = float(masked.mean().item())
    return masked


def _apply_train_video_augmentations(
    video: torch.Tensor,
    config: EarlyFusionDataConfig,
) -> torch.Tensor:
    augmented = video.clone()
    if config.train_video_random_crop_scale_min < 1.0:
        augmented = _apply_random_resized_crop(
            augmented,
            scale_min=config.train_video_random_crop_scale_min,
        )
    if config.train_video_horizontal_flip_prob > 0.0 and torch.rand(1).item() < config.train_video_horizontal_flip_prob:
        augmented = torch.flip(augmented, dims=(3,))
    if config.train_video_brightness_jitter > 0.0:
        brightness = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * config.train_video_brightness_jitter
        augmented = augmented * float(brightness)
    if config.train_video_contrast_jitter > 0.0:
        contrast = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * config.train_video_contrast_jitter
        frame_mean = augmented.mean(dim=(2, 3), keepdim=True)
        augmented = (augmented - frame_mean) * float(contrast) + frame_mean
    return augmented.clamp(0.0, 1.0).contiguous()


def _apply_random_resized_crop(video: torch.Tensor, *, scale_min: float) -> torch.Tensor:
    if not (0.0 < scale_min <= 1.0):
        raise ValueError(f"scale_min must be in (0, 1], got {scale_min}.")
    num_frames, channels, height, width = video.shape
    scale = float(torch.empty(1).uniform_(scale_min, 1.0).item())
    crop_height = max(1, int(round(height * scale)))
    crop_width = max(1, int(round(width * scale)))
    if crop_height == height and crop_width == width:
        return video
    top = int(torch.randint(low=0, high=height - crop_height + 1, size=(1,)).item())
    left = int(torch.randint(low=0, high=width - crop_width + 1, size=(1,)).item())
    cropped = video[:, :, top : top + crop_height, left : left + crop_width]
    resized = F.interpolate(
        cropped,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.view(num_frames, channels, height, width)


def _load_face_detector() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        return None
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None
    return detector


def _read_sampled_video_frames(video_path: Path, *, num_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file '{video_path}'.")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            indices = np.linspace(0, max(frame_count - 1, 0), num_frames).round().astype(int).tolist()
            frames = []
            last_good_frame: np.ndarray | None = None
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = capture.read()
                if success:
                    last_good_frame = frame
                    frames.append(frame)
                elif last_good_frame is not None:
                    frames.append(last_good_frame.copy())
            if frames:
                return _ensure_frame_count(frames, num_frames)

        sequential_frames: list[np.ndarray] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            sequential_frames.append(frame)
        if not sequential_frames:
            raise RuntimeError(f"Unable to decode any frames from '{video_path}'.")
        indices = np.linspace(0, len(sequential_frames) - 1, num_frames).round().astype(int).tolist()
        return [sequential_frames[index] for index in indices]
    finally:
        capture.release()


def _ensure_frame_count(frames: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
    if not frames:
        raise ValueError("Expected at least one frame to replicate.")
    if len(frames) >= num_frames:
        return frames[:num_frames]
    last_frame = frames[-1]
    while len(frames) < num_frames:
        frames.append(last_frame.copy())
    return frames


def _prepare_frame(
    frame_bgr: np.ndarray,
    *,
    image_size: int,
    face_detector: cv2.CascadeClassifier | None,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cropped = _crop_face_or_center(frame_rgb, face_detector=face_detector)
    resized = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _crop_face_or_center(
    frame_rgb: np.ndarray,
    *,
    face_detector: cv2.CascadeClassifier | None,
) -> np.ndarray:
    if face_detector is not None:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            margin = int(0.15 * max(w, h))
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(frame_rgb.shape[1], x + w + margin)
            y1 = min(frame_rgb.shape[0], y + h + margin)
            crop = frame_rgb[y0:y1, x0:x1]
            if crop.size > 0:
                return _center_square_crop(crop)
    return _center_square_crop(frame_rgb)


def _center_square_crop(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    side = min(height, width)
    top = max((height - side) // 2, 0)
    left = max((width - side) // 2, 0)
    return frame_rgb[top : top + side, left : left + side]
