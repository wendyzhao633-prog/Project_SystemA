from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.ravdess import LABEL_ORDER, RAVDESS_SPLIT_ORDER
from src.datasets.early_fusion_dataset import EarlyFusionDataConfig, build_split_datasets
from src.fusion.data import (
    build_late_fusion_layout,
    load_join_rows,
    load_labels,
    load_prob_matrix,
)
from src.fusion.metrics import compute_classification_metrics
from src.fusion.models import EmbeddingMLP, EmbeddingMLPConfig
from src.fusion.run_late_fusion import (
    build_weight_grid,
    compute_probability_loss,
    evaluate_embedding_features,
    load_embedding_features_and_labels,
)
from src.train.run_early_fusion import (
    EarlyFusionTrainConfig,
    build_dataloaders,
    build_early_fusion_layout,
    build_model,
    collect_probabilities,
    count_trainable_parameters,
    select_device,
)


EARLY_RUNS = ("early_fusion_3cnn", "early_fusion_bilinear", "early_fusion_xattn")
LATE_MODELS = ("prob_avg", "weighted_prob_avg", "embedding_mlp")


@dataclass(frozen=True)
class ReportLayout:
    repo_root: Path
    report_root: Path
    plots_root: Path
    models_root: Path


def export_baseline_report_artifacts(
    *,
    repo_root: Path | None = None,
    late_run_name: str = "ecapa_video_reuse_baseline",
    early_run_names: tuple[str, ...] = EARLY_RUNS,
    report_name: str = "baseline_diagnostic",
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    layout = build_report_layout(report_name, repo_root=resolved_repo_root)
    dataset_summary = build_dataset_summary(resolved_repo_root)
    device = select_device(None)

    model_reports: list[dict[str, Any]] = []
    for mode in LATE_MODELS:
        model_reports.append(
            collect_late_model_report(
                repo_root=resolved_repo_root,
                late_run_name=late_run_name,
                mode=mode,
                layout=layout,
                dataset_summary=dataset_summary,
                device=device,
            )
        )
    for run_name in early_run_names:
        model_reports.append(
            collect_early_model_report(
                repo_root=resolved_repo_root,
                run_name=run_name,
                layout=layout,
                dataset_summary=dataset_summary,
                device=device,
            )
        )

    write_global_tables_and_plots(layout, dataset_summary=dataset_summary, model_reports=model_reports)
    audit = build_audit_summary(dataset_summary=dataset_summary, model_reports=model_reports)
    write_json(layout.report_root / "audit_summary.json", audit)
    return audit


def build_report_model_id(*, family: str, model_name: str, run_name: str, mode: str | None = None) -> str:
    if family == "late_fusion":
        if mode is None:
            raise ValueError("Expected late-fusion mode when building a late-fusion report model id.")
        return f"late_{mode}"
    if family == "early_fusion":
        return run_name
    raise ValueError(f"Unsupported report family '{family}'.")


def build_display_name(*, family: str, model_name: str, run_name: str, mode: str | None = None) -> str:
    if family == "late_fusion":
        if mode is None:
            raise ValueError("Expected late-fusion mode when building a late-fusion display name.")
        return mode
    if family == "early_fusion":
        if run_name == model_name:
            return model_name
        return f"{model_name} ({run_name})"
    raise ValueError(f"Unsupported report family '{family}'.")


def build_report_layout(report_name: str, *, repo_root: Path | None = None) -> ReportLayout:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    return ReportLayout(
        repo_root=resolved_repo_root,
        report_root=resolved_repo_root / "outputs/reports" / report_name,
        plots_root=resolved_repo_root / "outputs/plots" / report_name,
        models_root=resolved_repo_root / "outputs/reports" / report_name / "models",
    )


def build_dataset_summary(repo_root: Path) -> dict[str, Any]:
    split_sizes: dict[str, int] = {}
    class_distribution: dict[str, dict[str, int]] = {}
    for split in RAVDESS_SPLIT_ORDER:
        path = repo_root / f"data/manifests/av/av_{split}.csv"
        rows = load_csv_rows(path)
        split_sizes[split] = len(rows)
        distribution = {label_name: 0 for label_name in LABEL_ORDER}
        for row in rows:
            distribution[row["label_name"]] += 1
        class_distribution[split] = distribution
    return {"split_sizes": split_sizes, "class_distribution": class_distribution}


def collect_early_model_report(
    *,
    repo_root: Path,
    run_name: str,
    layout: ReportLayout,
    dataset_summary: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    metrics_root = repo_root / "outputs/metrics/early_fusion" / run_name
    logs_root = repo_root / "outputs/logs/early_fusion" / run_name
    checkpoint_path = repo_root / "checkpoints/early_fusion" / run_name / "best_model.pt"
    resolved_config_path = logs_root / "resolved_config.json"
    history_path = logs_root / "training_history.json"

    summary = read_json(metrics_root / "summary.json")
    history = read_json(history_path) if history_path.is_file() else []
    resolved_config = read_json(resolved_config_path)
    model_name = str(summary["model_name"])
    per_split_metrics = {
        split: read_json(metrics_root / f"{split}_metrics.json")
        for split in RAVDESS_SPLIT_ORDER
    }

    config_data = resolved_config["data"]
    config_model = resolved_config["model"]
    data_config = EarlyFusionDataConfig(
        train_manifest=Path(config_data["train_manifest"]),
        valid_manifest=Path(config_data["valid_manifest"]),
        test_manifest=Path(config_data["test_manifest"]),
        sample_rate=int(config_data["sample_rate"]),
        n_fft=int(config_data["n_fft"]),
        hop_length=int(config_data["hop_length"]),
        win_length=int(config_data["win_length"]),
        n_mels=int(config_data["n_mels"]),
        audio_num_frames=int(config_data["audio_num_frames"]),
        num_video_frames=int(config_data["num_video_frames"]),
        image_size=int(config_data["image_size"]),
        use_face_crop=bool(config_data["use_face_crop"]),
        enable_cache=bool(config_data["enable_cache"]),
        audio_cache_root=Path(config_data["audio_cache_root"]),
        video_cache_root=Path(config_data["video_cache_root"]),
    )
    train_config = EarlyFusionTrainConfig(
        batch_size=int(resolved_config["train"]["batch_size"]),
        num_workers=int(resolved_config["train"]["num_workers"]),
        epochs=int(resolved_config["train"]["epochs"]),
        lr=float(resolved_config["train"]["lr"]),
        weight_decay=float(resolved_config["train"]["weight_decay"]),
        patience=int(resolved_config["train"]["patience"]),
        grad_clip_norm=float(resolved_config["train"]["grad_clip_norm"]),
    )
    model = build_model(model_name, config_model)
    parameter_count = int(summary.get("parameter_count", count_trainable_parameters(model)))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_eval_time_sec, test_loss = measure_early_test_evaluation(
        repo_root=repo_root,
        model=model,
        data_config=data_config,
        train_config=train_config,
        device=device,
    )
    per_split_metrics["test"]["evaluation_time_sec_measured"] = test_eval_time_sec
    per_split_metrics["test"]["loss_measured"] = test_loss

    model_id = build_report_model_id(
        family="early_fusion",
        model_name=model_name,
        run_name=run_name,
    )
    model_report = build_standardized_model_report(
        family="early_fusion",
        model_id=model_id,
        display_name=build_display_name(
            family="early_fusion",
            model_name=model_name,
            run_name=run_name,
        ),
        run_name=run_name,
        summary=summary,
        per_split_metrics=per_split_metrics,
        dataset_summary=dataset_summary,
        parameter_count=parameter_count,
        checkpoint_path=checkpoint_path,
        history=history,
        resolved_config=resolved_config,
        layout=layout,
    )
    model_report["diagnostic"] = diagnose_training_dynamics(model_report, history)
    write_model_artifacts(layout, model_report, history=history)
    return model_report


def collect_late_model_report(
    *,
    repo_root: Path,
    late_run_name: str,
    mode: str,
    layout: ReportLayout,
    dataset_summary: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    late_layout = build_late_fusion_layout(run_name=late_run_name, repo_root=repo_root)
    summary = read_json(repo_root / "outputs/metrics/late_fusion" / late_run_name / "summary.json")[mode]
    metrics_root = repo_root / "outputs/metrics/late_fusion" / late_run_name / mode
    per_split_metrics = {
        split: read_json(metrics_root / f"{split}_metrics.json")
        for split in RAVDESS_SPLIT_ORDER
    }
    history: list[dict[str, Any]] = []
    checkpoint_path: Path | None = None
    parameter_count = 0

    if mode == "embedding_mlp":
        checkpoint_path = repo_root / "checkpoints/late_fusion" / late_run_name / mode / "best_model.pt"
        history_path = repo_root / "checkpoints/late_fusion" / late_run_name / mode / "history.json"
        if history_path.is_file():
            history = read_json(history_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = EmbeddingMLP(EmbeddingMLPConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        parameter_count = count_trainable_parameters(model)
        test_eval_time_sec, test_loss = measure_late_embedding_test_evaluation(
            repo_root=repo_root,
            late_layout=late_layout,
            model=model,
            checkpoint=checkpoint,
            device=device,
        )
        per_split_metrics["test"]["evaluation_time_sec_measured"] = test_eval_time_sec
        per_split_metrics["test"]["loss_measured"] = test_loss
    else:
        checkpoint_path = (
            repo_root / "checkpoints/late_fusion" / late_run_name / mode / (
                "fusion_config.json" if mode == "prob_avg" else "fitted_weight.json"
            )
        )
        test_eval_time_sec, test_loss = measure_late_prob_mode_test_evaluation(
            repo_root=repo_root,
            late_layout=late_layout,
            mode=mode,
            summary=summary,
        )
        per_split_metrics["test"]["evaluation_time_sec_measured"] = test_eval_time_sec
        per_split_metrics["test"]["loss_measured"] = test_loss

    resolved_config = {"mode": mode, "run_name": late_run_name}
    if mode == "weighted_prob_avg":
        fitted = read_json(checkpoint_path)
        valid_search_history = fitted.get("valid_search_history")
        if valid_search_history is None:
            valid_search_history = compute_weighted_valid_sweep(late_layout)
        summary["valid_search_history"] = valid_search_history
    model_report = build_standardized_model_report(
        family="late_fusion",
        model_id=build_report_model_id(
            family="late_fusion",
            model_name=mode,
            run_name=late_run_name,
            mode=mode,
        ),
        display_name=build_display_name(
            family="late_fusion",
            model_name=mode,
            run_name=late_run_name,
            mode=mode,
        ),
        run_name=late_run_name,
        summary=summary,
        per_split_metrics=per_split_metrics,
        dataset_summary=dataset_summary,
        parameter_count=parameter_count,
        checkpoint_path=checkpoint_path,
        history=history,
        resolved_config=resolved_config,
        layout=layout,
    )
    model_report["diagnostic"] = diagnose_training_dynamics(model_report, history)
    if mode == "weighted_prob_avg":
        model_report["weight_sweep"] = summary["valid_search_history"]
    write_model_artifacts(layout, model_report, history=history)
    return model_report


def build_standardized_model_report(
    *,
    family: str,
    model_id: str,
    display_name: str,
    run_name: str,
    summary: dict[str, Any],
    per_split_metrics: dict[str, dict[str, Any]],
    dataset_summary: dict[str, Any],
    parameter_count: int,
    checkpoint_path: Path | None,
    history: list[dict[str, Any]],
    resolved_config: dict[str, Any],
    layout: ReportLayout,
) -> dict[str, Any]:
    checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_path is not None and checkpoint_path.is_file() else None
    timing = summary.get("timing", {})
    runtime = {
        "total_training_time_sec": timing.get("total_training_time_sec"),
        "average_epoch_time_sec": timing.get("average_epoch_time_sec"),
        "mode_runtime_sec": timing.get("mode_runtime_sec"),
        "test_evaluation_time_sec": per_split_metrics["test"].get("evaluation_time_sec_measured")
        or per_split_metrics["test"].get("evaluation_time_sec"),
    }

    report = {
        "family": family,
        "model_id": model_id,
        "model_name": display_name,
        "run_name": run_name,
        "best_epoch": summary.get("best_epoch"),
        "best_checkpoint_criterion": summary.get("best_checkpoint_criterion", "max_valid_uar"),
        "best_valid_uar": summary.get("best_valid_uar"),
        "parameter_count": parameter_count,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_path": checkpoint_path.as_posix() if checkpoint_path is not None else None,
        "core_metrics": {
            "valid_accuracy": per_split_metrics["valid"]["accuracy"],
            "valid_macro_f1": per_split_metrics["valid"]["macro_f1"],
            "valid_uar": per_split_metrics["valid"]["uar"],
            "test_accuracy": per_split_metrics["test"]["accuracy"],
            "test_macro_f1": per_split_metrics["test"]["macro_f1"],
            "test_uar": per_split_metrics["test"]["uar"],
        },
        "timing": runtime,
        "dataset": dataset_summary,
        "per_split_metrics": per_split_metrics,
        "history_path": None,
        "resolved_config": resolved_config,
        "plots": {},
    }
    if history:
        report["history_path"] = "captured"
    return report


def write_model_artifacts(layout: ReportLayout, model_report: dict[str, Any], *, history: list[dict[str, Any]]) -> None:
    model_root = layout.models_root / model_report["model_id"]
    plots_root = layout.plots_root / model_report["model_id"]
    model_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    core_metrics_row = {"model_id": model_report["model_id"], **model_report["core_metrics"]}
    write_csv(model_root / "core_metrics.csv", [core_metrics_row], fieldnames=list(core_metrics_row))
    write_json(model_root / "core_metrics.json", core_metrics_row)
    write_json(model_root / "artifact_summary.json", model_report)

    for split in RAVDESS_SPLIT_ORDER:
        metrics = model_report["per_split_metrics"][split]
        per_class_rows = build_per_class_rows(metrics)
        write_csv(
            model_root / f"per_class_{split}.csv",
            per_class_rows,
            fieldnames=["label_name", "precision", "recall", "f1", "support"],
        )
        matrix = extract_confusion_matrix(metrics)
        plot_confusion_heatmap(
            matrix,
            title=f"{model_report['model_name']} {split} confusion matrix",
            output_path=plots_root / f"{split}_confusion_matrix_heatmap.png",
        )

    plot_history_or_placeholder(
        history,
        x_field="epoch",
        y_field="train_loss",
        title=f"{model_report['model_name']} train loss",
        ylabel="train loss",
        output_path=plots_root / "train_loss_vs_epoch.png",
        missing_message="Training history not available.",
    )
    plot_history_or_placeholder(
        history,
        x_field="epoch",
        y_field="valid_loss",
        title=f"{model_report['model_name']} valid loss",
        ylabel="valid loss",
        output_path=plots_root / "valid_loss_vs_epoch.png",
        missing_message="Validation loss was not captured for this legacy run.",
    )
    plot_history_or_placeholder(
        history,
        x_field="epoch",
        y_field="valid_uar",
        title=f"{model_report['model_name']} valid UAR",
        ylabel="valid UAR",
        output_path=plots_root / "valid_uar_vs_epoch.png",
        missing_message="Validation UAR history not available.",
    )
    plot_history_or_placeholder(
        history,
        x_field="epoch",
        y_field="valid_macro_f1",
        title=f"{model_report['model_name']} valid macro F1",
        ylabel="valid macro F1",
        output_path=plots_root / "valid_macro_f1_vs_epoch.png",
        missing_message="Validation macro F1 history not available.",
    )

    if model_report["model_id"] == "late_weighted_prob_avg":
        weight_sweep = model_report.get("weight_sweep", [])
        if weight_sweep:
            plot_weight_sweep(
                weight_sweep,
                output_path=plots_root / "audio_weight_vs_valid_uar.png",
            )
        else:
            plot_placeholder(
                plots_root / "audio_weight_vs_valid_uar.png",
                title="Weighted late fusion valid UAR sweep",
                message="Weight sweep history not available.",
            )


def write_global_tables_and_plots(
    layout: ReportLayout,
    *,
    dataset_summary: dict[str, Any],
    model_reports: list[dict[str, Any]],
) -> None:
    layout.report_root.mkdir(parents=True, exist_ok=True)
    layout.plots_root.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    runtime_rows = []
    for report in model_reports:
        comparison_rows.append(
            {
                "model_id": report["model_id"],
                "model_name": report["model_name"],
                "family": report["family"],
                "run_name": report["run_name"],
                **report["core_metrics"],
            }
        )
        runtime_rows.append(
            {
                "model_id": report["model_id"],
                "model_name": report["model_name"],
                "family": report["family"],
                "parameter_count": report["parameter_count"],
                "checkpoint_size_bytes": report["checkpoint_size_bytes"],
                "total_training_time_sec": report["timing"]["total_training_time_sec"],
                "average_epoch_time_sec": report["timing"]["average_epoch_time_sec"],
                "test_evaluation_time_sec": report["timing"]["test_evaluation_time_sec"],
            }
        )

    write_csv(
        layout.report_root / "model_comparison.csv",
        comparison_rows,
        fieldnames=list(comparison_rows[0]),
    )
    write_json(layout.report_root / "model_comparison.json", comparison_rows)
    write_csv(
        layout.report_root / "runtime_comparison.csv",
        runtime_rows,
        fieldnames=list(runtime_rows[0]),
    )
    write_json(layout.report_root / "runtime_comparison.json", runtime_rows)

    split_rows = [
        {"split": split, "num_rows": count}
        for split, count in dataset_summary["split_sizes"].items()
    ]
    write_csv(layout.report_root / "dataset_split_sizes.csv", split_rows, fieldnames=["split", "num_rows"])
    distribution_rows = []
    for split, distribution in dataset_summary["class_distribution"].items():
        row = {"split": split, **distribution}
        distribution_rows.append(row)
    write_csv(
        layout.report_root / "dataset_class_distribution.csv",
        distribution_rows,
        fieldnames=["split", *LABEL_ORDER],
    )
    write_json(layout.report_root / "dataset_summary.json", dataset_summary)

    plot_model_comparison_bars(
        comparison_rows,
        metric_key="test_uar",
        output_path=layout.plots_root / "model_comparison_test_uar.png",
        title="Test UAR comparison",
    )
    plot_model_comparison_bars(
        comparison_rows,
        metric_key="test_macro_f1",
        output_path=layout.plots_root / "model_comparison_test_macro_f1.png",
        title="Test macro F1 comparison",
    )


def build_audit_summary(*, dataset_summary: dict[str, Any], model_reports: list[dict[str, Any]]) -> dict[str, Any]:
    best_model = max(model_reports, key=lambda report: report["core_metrics"]["test_uar"])
    diagnostics = {
        report["model_id"]: report["diagnostic"]
        for report in model_reports
    }
    training_recommendation = "Longer training is not justified yet."
    if any(report["diagnostic"]["longer_training_recommended"] for report in model_reports if report["family"] == "early_fusion"):
        training_recommendation = (
            "Longer training may be justified for the flagged runs, but only with early stopping and the captured diagnostics."
        )
    return {
        "dataset": dataset_summary,
        "best_model_by_test_uar": {
            "model_id": best_model["model_id"],
            "test_uar": best_model["core_metrics"]["test_uar"],
        },
        "diagnostics": diagnostics,
        "longer_training_recommendation": training_recommendation,
    }


def diagnose_training_dynamics(model_report: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "status": "no_training_curve",
            "summary": "No epoch history exists for this model.",
            "longer_training_recommended": False,
        }

    best_epoch = int(model_report["best_epoch"] or 0)
    total_epochs = len(history)
    best_valid_uar = float(model_report["best_valid_uar"] or 0.0)
    last_valid_uar = float(history[-1].get("valid_uar", best_valid_uar))
    last_train_loss = float(history[-1].get("train_loss", 0.0))
    best_history = max(history, key=lambda entry: entry.get("valid_uar", float("-inf")))
    best_train_loss = float(best_history.get("train_loss", last_train_loss))
    valid_drop = best_valid_uar - last_valid_uar

    flags: list[str] = []
    status = "stable"
    longer_training_recommended = False
    if best_epoch >= max(int(0.9 * total_epochs), 1) and valid_drop <= 0.01:
        status = "possible_undertraining"
        flags.append("best epoch occurred at the end of training")
        longer_training_recommended = True
    if valid_drop >= 0.03 and last_train_loss <= best_train_loss:
        status = "overfitting_after_peak"
        flags.append("validation UAR fell after the best epoch while train loss kept decreasing")
        longer_training_recommended = False
    if model_report["family"] == "early_fusion" and model_report["core_metrics"]["test_uar"] < 0.35:
        flags.append("test UAR remains low, suggesting architecture or optimization limits")
        longer_training_recommended = False

    if not flags:
        flags.append("no strong overfitting or undertraining signal was detected")
    return {
        "status": status,
        "best_epoch": best_epoch,
        "total_epochs_recorded": total_epochs,
        "best_valid_uar": best_valid_uar,
        "last_valid_uar": last_valid_uar,
        "valid_uar_drop_after_best": valid_drop,
        "flags": flags,
        "summary": "; ".join(flags),
        "longer_training_recommended": longer_training_recommended,
    }


def measure_early_test_evaluation(
    *,
    repo_root: Path,
    model: torch.nn.Module,
    data_config: EarlyFusionDataConfig,
    train_config: EarlyFusionTrainConfig,
    device: torch.device,
) -> tuple[float, float | None]:
    datasets = build_split_datasets(repo_root=repo_root, config=data_config, split_limits=None)
    dataloaders = build_dataloaders(datasets, train_config, device=device)
    start = time.perf_counter()
    _, _, _, average_loss = collect_probabilities(model, dataloaders["test"], device=device, compute_loss=True)
    return time.perf_counter() - start, average_loss


def measure_late_embedding_test_evaluation(
    *,
    repo_root: Path,
    late_layout: Any,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[float, float]:
    rows = load_join_rows(late_layout, "test")
    features, labels = load_embedding_features_and_labels(rows)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = (features - feature_mean) / feature_std
    start = time.perf_counter()
    _, loss = evaluate_embedding_features(
        model,
        features,
        labels,
        device=device,
        batch_size=128,
        class_weights=None,
    )
    return time.perf_counter() - start, loss


def measure_late_prob_mode_test_evaluation(
    *,
    repo_root: Path,
    late_layout: Any,
    mode: str,
    summary: dict[str, Any],
) -> tuple[float, float]:
    rows = load_join_rows(late_layout, "test")
    audio_probs = load_prob_matrix(rows, "audio")
    video_probs = load_prob_matrix(rows, "video")
    start = time.perf_counter()
    if mode == "prob_avg":
        fused_probs = 0.5 * (audio_probs + video_probs)
    else:
        fused_probs = float(summary["audio_weight"]) * audio_probs + float(summary["video_weight"]) * video_probs
    loss = compute_probability_loss(load_labels(rows), fused_probs)
    _ = compute_classification_metrics(load_labels(rows), fused_probs)
    return time.perf_counter() - start, loss


def compute_weighted_valid_sweep(late_layout: Any, *, weight_step: float = 0.01) -> list[dict[str, Any]]:
    rows = load_join_rows(late_layout, "valid")
    labels = load_labels(rows)
    audio_probs = load_prob_matrix(rows, "audio")
    video_probs = load_prob_matrix(rows, "video")
    history = []
    for weight in build_weight_grid(weight_step):
        fused = weight * audio_probs + (1.0 - weight) * video_probs
        metrics = compute_classification_metrics(labels, fused)
        history.append({"audio_weight": weight, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "uar": metrics["uar"]})
    return history


def build_per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    support = np.asarray(extract_confusion_matrix(metrics), dtype=np.int64).sum(axis=1).tolist()
    rows = []
    for index, label_name in enumerate(LABEL_ORDER):
        rows.append(
            {
                "label_name": label_name,
                "precision": metrics["per_class_precision"][label_name],
                "recall": metrics["per_class_recall"][label_name],
                "f1": metrics["per_class_f1"][label_name],
                "support": int(support[index]),
            }
        )
    return rows


def extract_confusion_matrix(metrics: dict[str, Any]) -> list[list[int]]:
    if "confusion_matrix" in metrics:
        return metrics["confusion_matrix"]
    confusion_path = Path(metrics["confusion_matrix_path"])
    with confusion_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    matrix = []
    for row in rows[1:]:
        matrix.append([int(value) for value in row[1:]])
    return matrix


def plot_history_or_placeholder(
    history: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
    missing_message: str,
) -> None:
    if history and all(y_field in entry for entry in history):
        x_values = [entry[x_field] for entry in history]
        y_values = [entry[y_field] for entry in history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_values, y_values, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        save_figure(fig, output_path)
        return
    plot_placeholder(output_path, title=title, message=missing_message)


def plot_confusion_heatmap(matrix: list[list[int]], *, title: str, output_path: Path) -> None:
    array = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(array, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for row_index in range(array.shape[0]):
        for col_index in range(array.shape[1]):
            ax.text(col_index, row_index, int(array[row_index, col_index]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_path)


def plot_model_comparison_bars(
    comparison_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    output_path: Path,
    title: str,
) -> None:
    labels = [row["model_id"] for row in comparison_rows]
    values = [row[metric_key] for row in comparison_rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, values, color="#4472c4")
    ax.set_title(title)
    ax.set_ylabel(metric_key)
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, output_path)


def plot_weight_sweep(weight_history: list[dict[str, Any]], *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [row["audio_weight"] for row in weight_history],
        [row["uar"] for row in weight_history],
        marker="o",
        linewidth=1.5,
    )
    ax.set_title("Weighted late fusion audio weight vs valid UAR")
    ax.set_xlabel("audio weight")
    ax.set_ylabel("valid UAR")
    ax.grid(True, alpha=0.3)
    save_figure(fig, output_path)


def plot_placeholder(output_path: Path, *, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_figure(fig, output_path)


def save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
