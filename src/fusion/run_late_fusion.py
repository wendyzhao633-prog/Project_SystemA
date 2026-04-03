from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common.ravdess import LABEL_ORDER, RAVDESS_SPLIT_ORDER
from src.fusion.data import (
    LateFusionLayout,
    build_av_late_fusion_tables,
    build_late_fusion_layout,
    load_audio_embeddings,
    load_join_rows,
    load_labels,
    load_prob_matrix,
    load_video_sequence_embeddings,
)
from src.fusion.metrics import compute_classification_metrics, write_confusion_matrix_csv
from src.fusion.models import EmbeddingMLP, EmbeddingMLPConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run late-fusion baselines on prepared AV tables.")
    parser.add_argument(
        "mode",
        choices=("prob_avg", "weighted_prob_avg", "embedding_mlp", "all"),
        help="Fusion mode to run.",
    )
    parser.add_argument("--run-name", default="ecapa_video_reuse_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    layout = build_late_fusion_layout(run_name=args.run_name)
    build_av_late_fusion_tables(layout)
    joined_rows = {split: load_join_rows(layout, split) for split in RAVDESS_SPLIT_ORDER}

    summary: dict[str, Any] = {}
    if args.mode in {"prob_avg", "all"}:
        summary["prob_avg"] = run_prob_avg(layout, joined_rows)
    if args.mode in {"weighted_prob_avg", "all"}:
        summary["weighted_prob_avg"] = run_weighted_prob_avg(
            layout,
            joined_rows,
            weight_step=args.weight_step,
        )
    if args.mode in {"embedding_mlp", "all"}:
        summary["embedding_mlp"] = run_embedding_mlp(
            layout,
            joined_rows,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

    summary_path = layout.metrics_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")
    for mode, mode_summary in summary.items():
        test_metrics = mode_summary["splits"]["test"]
        print(
            f"{mode}: test_acc={test_metrics['accuracy']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f} test_uar={test_metrics['uar']:.4f}"
        )
    return 0


def run_prob_avg(layout: LateFusionLayout, joined_rows: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    mode = "prob_avg"
    mode_start = time.perf_counter()
    outputs = {}
    for split, rows in joined_rows.items():
        audio_probs = load_prob_matrix(rows, "audio")
        video_probs = load_prob_matrix(rows, "video")
        fused_probs = 0.5 * (audio_probs + video_probs)
        outputs[split] = evaluate_and_write_mode_outputs(
            layout,
            mode=mode,
            split=split,
            rows=rows,
            fused_probs=fused_probs,
            fused_loss=compute_probability_loss(load_labels(rows), fused_probs),
            extra_metadata={"audio_weight": 0.5, "video_weight": 0.5},
        )

    checkpoint_dir = layout.checkpoints_root / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "fusion_config.json").write_text(
        json.dumps({"mode": mode, "audio_weight": 0.5, "video_weight": 0.5}, indent=2),
        encoding="utf-8",
    )
    config_path = checkpoint_dir / "fusion_config.json"
    return {
        "mode": mode,
        "parameter_count": 0,
        "checkpoint_path": config_path.as_posix(),
        "checkpoint_size_bytes": config_path.stat().st_size,
        "timing": {"total_training_time_sec": 0.0, "average_epoch_time_sec": 0.0, "mode_runtime_sec": time.perf_counter() - mode_start},
        "best_checkpoint_criterion": "not_applicable",
        "splits": outputs,
    }


def run_weighted_prob_avg(
    layout: LateFusionLayout,
    joined_rows: dict[str, list[dict[str, str]]],
    *,
    weight_step: float,
) -> dict[str, Any]:
    mode = "weighted_prob_avg"
    mode_start = time.perf_counter()
    train_rows = joined_rows["train"]
    train_labels = load_labels(train_rows)
    train_audio = load_prob_matrix(train_rows, "audio")
    train_video = load_prob_matrix(train_rows, "video")
    valid_rows = joined_rows["valid"]
    valid_labels = load_labels(valid_rows)
    valid_audio = load_prob_matrix(valid_rows, "audio")
    valid_video = load_prob_matrix(valid_rows, "video")

    candidate_weights = build_weight_grid(weight_step)
    best_weight = 0.5
    best_score: tuple[float, float, float] | None = None
    search_history = []
    valid_search_history = []
    for weight in candidate_weights:
        fused = weight * train_audio + (1.0 - weight) * train_video
        metrics = compute_classification_metrics(train_labels, fused)
        score = (metrics["uar"], metrics["macro_f1"], metrics["accuracy"])
        search_history.append({"audio_weight": weight, **{k: metrics[k] for k in ("accuracy", "macro_f1", "uar")}})
        valid_fused = weight * valid_audio + (1.0 - weight) * valid_video
        valid_metrics = compute_classification_metrics(valid_labels, valid_fused)
        valid_search_history.append(
            {"audio_weight": weight, **{k: valid_metrics[k] for k in ("accuracy", "macro_f1", "uar")}}
        )
        if best_score is None or score > best_score:
            best_score = score
            best_weight = weight

    outputs = {}
    for split, rows in joined_rows.items():
        audio_probs = load_prob_matrix(rows, "audio")
        video_probs = load_prob_matrix(rows, "video")
        fused_probs = best_weight * audio_probs + (1.0 - best_weight) * video_probs
        outputs[split] = evaluate_and_write_mode_outputs(
            layout,
            mode=mode,
            split=split,
            rows=rows,
            fused_probs=fused_probs,
            fused_loss=compute_probability_loss(load_labels(rows), fused_probs),
            extra_metadata={"audio_weight": best_weight, "video_weight": 1.0 - best_weight},
        )

    checkpoint_dir = layout.checkpoints_root / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "fitted_weight.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "audio_weight": best_weight,
                "video_weight": 1.0 - best_weight,
                "train_search_history": search_history,
                "valid_search_history": valid_search_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint_path = checkpoint_dir / "fitted_weight.json"
    return {
        "mode": mode,
        "audio_weight": best_weight,
        "video_weight": 1.0 - best_weight,
        "parameter_count": 0,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "best_checkpoint_criterion": "max_train_uar_then_macro_f1_then_accuracy",
        "timing": {"total_training_time_sec": 0.0, "average_epoch_time_sec": 0.0, "mode_runtime_sec": time.perf_counter() - mode_start},
        "valid_search_history": valid_search_history,
        "splits": outputs,
    }


def run_embedding_mlp(
    layout: LateFusionLayout,
    joined_rows: dict[str, list[dict[str, str]]],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> dict[str, Any]:
    mode = "embedding_mlp"
    set_seed(seed)
    mode_start = time.perf_counter()

    train_features, train_labels = load_embedding_features_and_labels(joined_rows["train"])
    valid_features, valid_labels = load_embedding_features_and_labels(joined_rows["valid"])
    test_features, test_labels = load_embedding_features_and_labels(joined_rows["test"])

    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0

    train_features = (train_features - feature_mean) / feature_std
    valid_features = (valid_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std

    train_loader = build_dataloader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    valid_loader = build_dataloader(valid_features, valid_labels, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = EmbeddingMLPConfig(input_dim=train_features.shape[1])
    model = EmbeddingMLP(config).to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = compute_class_weights(train_labels).to(device)

    best_state: dict[str, Any] | None = None
    best_valid_uar = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_items = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_features.shape[0]
            epoch_items += batch_features.shape[0]

        train_metrics = compute_classification_metrics(train_labels, predict_probabilities(model, train_features, device))
        valid_probs, valid_loss = evaluate_embedding_features(
            model,
            valid_features,
            valid_labels,
            device=device,
            batch_size=batch_size,
            class_weights=class_weights,
        )
        valid_metrics = compute_classification_metrics(valid_labels, valid_probs)
        avg_loss = epoch_loss / max(epoch_items, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "valid_loss": valid_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_uar": train_metrics["uar"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_uar": valid_metrics["uar"],
                "epoch_time_sec": time.perf_counter() - epoch_start,
            }
        )

        if valid_metrics["uar"] > best_valid_uar:
            best_valid_uar = valid_metrics["uar"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "config": asdict(config),
                "epoch": epoch,
                "best_valid_uar": best_valid_uar,
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state["model_state_dict"])

    checkpoint_dir = layout.checkpoints_root / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    total_training_time_sec = time.perf_counter() - training_start
    average_epoch_time_sec = (
        float(sum(entry["epoch_time_sec"] for entry in history) / len(history))
        if history
        else 0.0
    )

    outputs = {}
    for split, rows, features, labels in (
        ("train", joined_rows["train"], train_features, train_labels),
        ("valid", joined_rows["valid"], valid_features, valid_labels),
        ("test", joined_rows["test"], test_features, test_labels),
    ):
        fused_probs, fused_loss = evaluate_embedding_features(
            model,
            features,
            labels,
            device=device,
            batch_size=batch_size,
            class_weights=class_weights,
        )
        outputs[split] = evaluate_and_write_mode_outputs(
            layout,
            mode=mode,
            split=split,
            rows=rows,
            fused_probs=fused_probs,
            fused_loss=fused_loss,
            extra_metadata={"checkpoint_path": checkpoint_path.as_posix(), "best_epoch": best_epoch},
        )

    return {
        "mode": mode,
        "device": str(device),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_valid_uar": best_valid_uar,
        "best_checkpoint_criterion": "max_valid_uar",
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "timing": {
            "total_training_time_sec": total_training_time_sec,
            "average_epoch_time_sec": average_epoch_time_sec,
            "mode_runtime_sec": time.perf_counter() - mode_start,
        },
        "splits": outputs,
    }


def evaluate_and_write_mode_outputs(
    layout: LateFusionLayout,
    *,
    mode: str,
    split: str,
    rows: list[dict[str, str]],
    fused_probs: np.ndarray,
    fused_loss: float | None,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    labels = load_labels(rows)
    evaluation_start = time.perf_counter()
    metrics = compute_classification_metrics(labels, fused_probs)
    evaluation_time_sec = time.perf_counter() - evaluation_start

    mode_predictions_root = layout.predictions_root / mode
    mode_metrics_root = layout.metrics_root / mode
    mode_predictions_root.mkdir(parents=True, exist_ok=True)
    mode_metrics_root.mkdir(parents=True, exist_ok=True)

    prediction_path = mode_predictions_root / f"{split}_predictions.jsonl"
    metrics_path = mode_metrics_root / f"{split}_metrics.json"
    confusion_path = mode_metrics_root / f"{split}_confusion_matrix.csv"

    write_prediction_jsonl(prediction_path, rows, fused_probs, metrics["pred_ids"], mode=mode, extra_metadata=extra_metadata)
    write_confusion_matrix_csv(confusion_path, np.asarray(metrics["confusion_matrix"], dtype=np.int64))

    metrics_payload = {
        "split": split,
        "mode": mode,
        "loss": fused_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "uar": metrics["uar"],
        "evaluation_time_sec": evaluation_time_sec,
        "confusion_matrix_path": confusion_path.as_posix(),
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_f1": metrics["per_class_f1"],
        **extra_metadata,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def write_prediction_jsonl(
    path: Path,
    rows: list[dict[str, str]],
    probs: np.ndarray,
    pred_ids: list[int],
    *,
    mode: str,
    extra_metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, prob_vector, pred_id in zip(rows, probs.tolist(), pred_ids, strict=True):
            record = {
                "mode": mode,
                "av_key": row["av_key"],
                "split": row["split"],
                "actor_id": int(row["actor_id"]),
                "label_id": int(row["label_id"]),
                "label_name": row["label_name"],
                "audio_sample_id": row["audio_sample_id"],
                "video_sample_id": row["video_sample_id"],
                "pred_id": int(pred_id),
                "pred_name": LABEL_ORDER[int(pred_id)],
                "probs": [float(value) for value in prob_vector],
                **extra_metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_embedding_features_and_labels(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    audio_embeddings = load_audio_embeddings(rows)
    video_embeddings = load_video_sequence_embeddings(rows)
    features = np.concatenate([audio_embeddings, video_embeddings], axis=1)
    labels = load_labels(rows)
    return features.astype(np.float32), labels


def build_dataloader(features: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def predict_probabilities(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return probs


def evaluate_embedding_features(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    class_weights: torch.Tensor | None,
) -> tuple[np.ndarray, float]:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weights, reduction="sum")
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            all_probs.append(probs)
            total_loss += float(loss.item())
            total_items += batch_features.shape[0]
    return np.concatenate(all_probs, axis=0), total_loss / max(total_items, 1)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABEL_ORDER)).astype(np.float32)
    weights = counts.sum() / (len(LABEL_ORDER) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def build_weight_grid(weight_step: float) -> list[float]:
    if weight_step <= 0 or weight_step > 1:
        raise ValueError(f"weight_step must be in (0, 1], got {weight_step}.")
    num_steps = int(round(1.0 / weight_step))
    return [round(step * weight_step, 10) for step in range(num_steps + 1)]


def compute_probability_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probs, 1e-12, 1.0)
    losses = -np.log(clipped[np.arange(labels.shape[0]), labels])
    return float(losses.mean())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
