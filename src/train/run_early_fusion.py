from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.common.ravdess import LABEL_ORDER, RAVDESS_SPLIT_ORDER
from src.datasets.early_fusion_dataset import EarlyFusionDataConfig, build_split_datasets
from src.fusion.metrics import compute_classification_metrics, write_confusion_matrix_csv
from src.models.early_fusion_3cnn import EarlyFusion3CNN, EarlyFusion3CNNConfig
from src.models.early_fusion_bilinear import EarlyFusionBilinear, EarlyFusionBilinearConfig
from src.models.early_fusion_xattn import EarlyFusionXAttn, EarlyFusionXAttnConfig


@dataclass(frozen=True)
class EarlyFusionTrainConfig:
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    grad_clip_norm: float = 1.0
    scheduler_name: str = "none"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    scheduler_min_lr: float = 1e-6


@dataclass(frozen=True)
class EarlyFusionRunConfig:
    model_name: str
    run_name: str
    seed: int
    data: EarlyFusionDataConfig
    train: EarlyFusionTrainConfig
    model: dict[str, Any]
    config_path: Path


@dataclass(frozen=True)
class EarlyFusionLayout:
    repo_root: Path
    checkpoints_root: Path
    metrics_root: Path
    predictions_root: Path
    logs_root: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate early-fusion baselines.")
    parser.add_argument("--config", required=True, help="Path to a YAML config under configs/early_fusion/.")
    parser.add_argument("--run-name", help="Optional override for the configured run name.")
    parser.add_argument("--epochs", type=int, help="Optional override for training epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for batch size.")
    parser.add_argument("--num-workers", type=int, help="Optional override for DataLoader workers.")
    parser.add_argument("--max-train-items", type=int, help="Optional cap for smoke runs.")
    parser.add_argument("--max-valid-items", type=int, help="Optional cap for smoke runs.")
    parser.add_argument("--max-test-items", type=int, help="Optional cap for smoke runs.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable audio/video cache reads and writes.")
    parser.add_argument("--device", help="Explicit device string, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config = load_run_config(Path(args.config), repo_root=repo_root)
    if args.run_name:
        config = replace(config, run_name=args.run_name)
    if args.epochs is not None:
        config = replace(config, train=replace(config.train, epochs=args.epochs))
    if args.batch_size is not None:
        config = replace(config, train=replace(config.train, batch_size=args.batch_size))
    if args.num_workers is not None:
        config = replace(config, train=replace(config.train, num_workers=args.num_workers))
    if args.disable_cache:
        config = replace(config, data=replace(config.data, enable_cache=False))

    split_limits = {
        "train": args.max_train_items,
        "valid": args.max_valid_items,
        "test": args.max_test_items,
    }
    layout = build_early_fusion_layout(config.run_name, repo_root=repo_root)
    device = select_device(args.device)

    set_seed(config.seed)
    datasets = build_split_datasets(
        repo_root=repo_root,
        config=config.data,
        split_limits=split_limits,
    )
    dataloaders = build_dataloaders(datasets, config.train, device=device)
    model = build_model(config.model_name, config.model).to(device)

    checkpoint_path = layout.checkpoints_root / "best_model.pt"
    history_path = layout.logs_root / "training_history.json"
    config_path = layout.logs_root / "resolved_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(run_config_to_dict(config), indent=2), encoding="utf-8")

    parameter_count = count_trainable_parameters(model)
    dataset_summary = summarize_datasets(datasets)
    train_result = train_model(
        model=model,
        dataloaders=dataloaders,
        device=device,
        train_config=config.train,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        config=config,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary = {
        "model_name": config.model_name,
        "run_name": config.run_name,
        "device": str(device),
        "parameter_count": parameter_count,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "history_path": history_path.as_posix(),
        "resolved_config_path": config_path.as_posix(),
        "best_epoch": train_result["best_epoch"],
        "best_valid_uar": train_result["best_valid_uar"],
        "best_checkpoint_criterion": "max_valid_uar",
        "timing": {
            "total_training_time_sec": train_result["total_training_time_sec"],
            "average_epoch_time_sec": train_result["average_epoch_time_sec"],
        },
        "dataset": dataset_summary,
        "splits": {},
    }
    for split in RAVDESS_SPLIT_ORDER:
        split_summary = evaluate_split(
            model=model,
            dataloader=dataloaders[split],
            device=device,
            layout=layout,
            split=split,
            model_name=config.model_name,
        )
        summary["splits"][split] = split_summary

    summary_path = layout.metrics_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {summary_path}")
    test_metrics = summary["splits"]["test"]
    print(
        f"{config.model_name}: test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} test_uar={test_metrics['uar']:.4f}"
    )
    return 0


def load_run_config(config_path: Path, *, repo_root: Path) -> EarlyFusionRunConfig:
    resolved_config_path = config_path if config_path.is_absolute() else repo_root / config_path
    raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in config '{resolved_config_path}'.")

    data_raw = raw.get("data", {})
    train_raw = raw.get("train", {})
    model_name = str(raw["model_name"])
    run_name = str(raw.get("run_name", model_name))
    data_config = EarlyFusionDataConfig(
        train_manifest=Path(data_raw.get("train_manifest", "data/manifests/av/av_train.csv")),
        valid_manifest=Path(data_raw.get("valid_manifest", "data/manifests/av/av_valid.csv")),
        test_manifest=Path(data_raw.get("test_manifest", "data/manifests/av/av_test.csv")),
        sample_rate=int(data_raw.get("sample_rate", 16000)),
        n_fft=int(data_raw.get("n_fft", 400)),
        hop_length=int(data_raw.get("hop_length", 160)),
        win_length=int(data_raw.get("win_length", 400)),
        n_mels=int(data_raw.get("n_mels", 80)),
        audio_num_frames=int(data_raw.get("audio_num_frames", 256)),
        num_video_frames=int(data_raw.get("num_video_frames", 5)),
        image_size=int(data_raw.get("image_size", 224)),
        use_face_crop=bool(data_raw.get("use_face_crop", False)),
        enable_cache=bool(data_raw.get("enable_cache", True)),
        audio_cache_root=Path(
            data_raw.get(
                "audio_cache_root",
                "data/processed/av/early_fusion_cache/audio_logmel_sr16000_mel80_t256",
            )
        ),
        video_cache_root=Path(
            data_raw.get(
                "video_cache_root",
                "data/processed/av/early_fusion_cache/video_frames_5f_224",
            )
        ),
        train_audio_gain_jitter_db=float(data_raw.get("train_audio_gain_jitter_db", 0.0)),
        train_specaug_freq_mask_width=int(data_raw.get("train_specaug_freq_mask_width", 0)),
        train_specaug_time_mask_width=int(data_raw.get("train_specaug_time_mask_width", 0)),
        train_video_random_crop_scale_min=float(data_raw.get("train_video_random_crop_scale_min", 1.0)),
        train_video_horizontal_flip_prob=float(data_raw.get("train_video_horizontal_flip_prob", 0.0)),
        train_video_brightness_jitter=float(data_raw.get("train_video_brightness_jitter", 0.0)),
        train_video_contrast_jitter=float(data_raw.get("train_video_contrast_jitter", 0.0)),
        validate_ravdess_av_key=bool(data_raw.get("validate_ravdess_av_key", True)),
    )
    if data_config.n_mels != 80:
        raise ValueError(f"Expected 80 mel bins for the canonical baseline, got {data_config.n_mels}.")
    if data_config.num_video_frames != 5:
        raise ValueError(
            f"Expected 5 video frames for the canonical early-fusion baseline, got {data_config.num_video_frames}."
        )

    train_config = EarlyFusionTrainConfig(
        batch_size=int(train_raw.get("batch_size", 16)),
        num_workers=int(train_raw.get("num_workers", 0)),
        epochs=int(train_raw.get("epochs", 30)),
        lr=float(train_raw.get("lr", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        patience=int(train_raw.get("patience", 8)),
        grad_clip_norm=float(train_raw.get("grad_clip_norm", 1.0)),
        scheduler_name=str(train_raw.get("scheduler_name", "none")),
        scheduler_factor=float(train_raw.get("scheduler_factor", 0.5)),
        scheduler_patience=int(train_raw.get("scheduler_patience", 4)),
        scheduler_min_lr=float(train_raw.get("scheduler_min_lr", 1e-6)),
    )
    model_config = dict(raw.get("model", {}))
    if int(model_config.get("num_classes", len(LABEL_ORDER))) != len(LABEL_ORDER):
        raise ValueError("The canonical baseline must use the fixed 8-class label order.")

    return EarlyFusionRunConfig(
        model_name=model_name,
        run_name=run_name,
        seed=int(raw.get("seed", 42)),
        data=data_config,
        train=train_config,
        model=model_config,
        config_path=resolved_config_path,
    )


def build_early_fusion_layout(run_name: str, *, repo_root: Path | None = None) -> EarlyFusionLayout:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    return EarlyFusionLayout(
        repo_root=resolved_repo_root,
        checkpoints_root=resolved_repo_root / "checkpoints/early_fusion" / run_name,
        metrics_root=resolved_repo_root / "outputs/metrics/early_fusion" / run_name,
        predictions_root=resolved_repo_root / "outputs/predictions/early_fusion" / run_name,
        logs_root=resolved_repo_root / "outputs/logs/early_fusion" / run_name,
    )


def build_model(model_name: str, model_config: dict[str, Any]) -> nn.Module:
    if model_name == "early_fusion_3cnn":
        return EarlyFusion3CNN(EarlyFusion3CNNConfig(**model_config))
    if model_name == "early_fusion_bilinear":
        return EarlyFusionBilinear(EarlyFusionBilinearConfig(**model_config))
    if model_name == "early_fusion_xattn":
        return EarlyFusionXAttn(EarlyFusionXAttnConfig(**model_config))
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def build_dataloaders(
    datasets: dict[str, Any],
    train_config: EarlyFusionTrainConfig,
    *,
    device: torch.device,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=(split == "train"),
            num_workers=train_config.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    return loaders


def train_model(
    *,
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    device: torch.device,
    train_config: EarlyFusionTrainConfig,
    checkpoint_path: Path,
    history_path: Path,
    config: EarlyFusionRunConfig,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, train_config)

    best_valid_uar = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    total_training_start = time.perf_counter()

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_items = 0
        for batch in dataloaders["train"]:
            audio_input = batch["audio_input"].to(device)
            video_input = batch["video_input"].to(device)
            labels = batch["label_id"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(audio_input, video_input)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if train_config.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip_norm)
            optimizer.step()

            batch_size = audio_input.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            epoch_items += batch_size

        valid_probs, valid_labels, _, valid_loss = collect_probabilities(
            model,
            dataloaders["valid"],
            device=device,
            compute_loss=True,
        )
        valid_metrics = compute_classification_metrics(valid_labels, valid_probs)
        average_loss = epoch_loss / max(epoch_items, 1)
        epoch_time_sec = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_uar": valid_metrics["uar"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": epoch_time_sec,
            }
        )

        step_scheduler(scheduler, monitor_value=valid_metrics["uar"])
        if valid_metrics["uar"] > best_valid_uar:
            best_valid_uar = valid_metrics["uar"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": config.model_name,
                    "run_name": config.run_name,
                    "epoch": epoch,
                    "best_valid_uar": best_valid_uar,
                    "config": run_config_to_dict(config),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.patience:
                break

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    total_training_time_sec = time.perf_counter() - total_training_start
    average_epoch_time_sec = (
        float(sum(entry["epoch_time_sec"] for entry in history) / len(history))
        if history
        else 0.0
    )
    return {
        "best_epoch": best_epoch,
        "best_valid_uar": best_valid_uar,
        "history": history,
        "total_training_time_sec": total_training_time_sec,
        "average_epoch_time_sec": average_epoch_time_sec,
    }


def evaluate_split(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layout: EarlyFusionLayout,
    split: str,
    model_name: str,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    probabilities, labels, metadata_rows, average_loss = collect_probabilities(
        model,
        dataloader,
        device=device,
        compute_loss=True,
    )
    metrics = compute_classification_metrics(labels, probabilities)
    evaluation_time_sec = time.perf_counter() - eval_start

    split_metrics = {
        "split": split,
        "model_name": model_name,
        "num_rows": int(probabilities.shape[0]),
        "loss": average_loss,
        "evaluation_time_sec": evaluation_time_sec,
        **{key: metrics[key] for key in metrics if key != "pred_ids"},
    }

    metrics_path = layout.metrics_root / f"{split}_metrics.json"
    confusion_path = layout.metrics_root / f"{split}_confusion_matrix.csv"
    predictions_path = layout.predictions_root / f"{split}_predictions.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path.write_text(json.dumps(split_metrics, indent=2), encoding="utf-8")
    write_confusion_matrix_csv(confusion_path, np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    write_prediction_jsonl(
        predictions_path,
        metadata_rows=metadata_rows,
        probabilities=probabilities,
        true_labels=labels,
    )
    return split_metrics


def collect_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    compute_loss: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float | None]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in dataloader:
            audio_input = batch["audio_input"].to(device)
            video_input = batch["video_input"].to(device)
            batch_labels = batch["label_id"].to(device)
            logits = model(audio_input, video_input)
            if compute_loss:
                total_loss += float(F.cross_entropy(logits, batch_labels, reduction="sum").item())
                total_items += audio_input.shape[0]
            probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            labels = batch_labels.cpu().numpy().astype(np.int64)

            all_probabilities.append(probabilities)
            all_labels.append(labels)
            metadata_rows.extend(build_batch_metadata(batch))

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_labels, axis=0),
        metadata_rows,
        (total_loss / max(total_items, 1)) if compute_loss else None,
    )


def build_batch_metadata(batch: dict[str, Any]) -> list[dict[str, Any]]:
    batch_size = len(batch["av_key"])
    metadata_rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        metadata_rows.append(
            {
                "av_key": batch["av_key"][index],
                "split": batch["split"][index],
                "actor_id": int(batch["actor_id"][index]),
                "label_name": batch["label_name"][index],
                "emotion_code": batch["emotion_code"][index],
                "audio_sample_id": batch["audio_sample_id"][index],
                "video_sample_id": batch["video_sample_id"][index],
                "audio_path": batch["audio_path"][index],
                "video_path": batch["video_path"][index],
            }
        )
    return metadata_rows


def write_prediction_jsonl(
    path: Path,
    *,
    metadata_rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    true_labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_ids = probabilities.argmax(axis=1)
    with path.open("w", encoding="utf-8") as handle:
        for metadata, prob_vector, label_id, pred_id in zip(
            metadata_rows,
            probabilities,
            true_labels,
            pred_ids,
            strict=True,
        ):
            record = {
                "av_key": metadata["av_key"],
                "split": metadata["split"],
                "actor_id": metadata["actor_id"],
                "label_id": int(label_id),
                "label_name": LABEL_ORDER[int(label_id)],
                "emotion_code": metadata["emotion_code"],
                "audio_sample_id": metadata["audio_sample_id"],
                "video_sample_id": metadata["video_sample_id"],
                "audio_path": metadata["audio_path"],
                "video_path": metadata["video_path"],
                "pred_id": int(pred_id),
                "pred_name": LABEL_ORDER[int(pred_id)],
            }
            for label_name, value in zip(LABEL_ORDER, prob_vector.tolist(), strict=True):
                record[f"prob_{label_name}"] = float(value)
            handle.write(json.dumps(record) + "\n")


def run_config_to_dict(config: EarlyFusionRunConfig) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "run_name": config.run_name,
        "seed": config.seed,
        "data": _dataclass_to_serializable_dict(config.data),
        "train": _dataclass_to_serializable_dict(config.train),
        "model": config.model,
        "config_path": str(config.config_path),
    }


def _dataclass_to_serializable_dict(instance: Any) -> dict[str, Any]:
    raw = asdict(instance)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in raw.items()
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def summarize_datasets(datasets: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"split_sizes": {}, "class_distribution": {}}
    for split, dataset in datasets.items():
        summary["split_sizes"][split] = len(dataset)
        distribution = {label_name: 0 for label_name in LABEL_ORDER}
        for example in dataset.examples:
            distribution[example.label_name] += 1
        summary["class_distribution"][split] = distribution
    return summary


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: EarlyFusionTrainConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = train_config.scheduler_name.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=train_config.scheduler_factor,
            patience=train_config.scheduler_patience,
            min_lr=train_config.scheduler_min_lr,
        )
    raise ValueError(f"Unsupported scheduler_name '{train_config.scheduler_name}'.")


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    *,
    monitor_value: float,
) -> None:
    if scheduler is None:
        return
    scheduler.step(monitor_value)


def select_device(explicit_device: str | None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    raise SystemExit(main())
