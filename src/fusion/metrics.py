from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.common.ravdess import LABEL_ORDER


def compute_classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(LABEL_ORDER):
        raise ValueError(f"Expected probability matrix with shape (N, {len(LABEL_ORDER)}), got {probs.shape}.")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError(f"Mismatched y_true/probs sizes: {y_true.shape[0]} vs {probs.shape[0]}.")

    y_pred = np.argmax(probs, axis=1)
    confusion = compute_confusion_matrix(y_true, y_pred, num_classes=len(LABEL_ORDER))
    per_class_recall = safe_divide(np.diag(confusion), confusion.sum(axis=1))
    per_class_precision = safe_divide(np.diag(confusion), confusion.sum(axis=0))
    per_class_f1 = safe_divide(
        2.0 * per_class_precision * per_class_recall,
        per_class_precision + per_class_recall,
    )

    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "macro_f1": float(np.mean(per_class_f1)),
        "uar": float(np.mean(per_class_recall)),
        "confusion_matrix": confusion.tolist(),
        "per_class_precision": {label: float(value) for label, value in zip(LABEL_ORDER, per_class_precision)},
        "per_class_recall": {label: float(value) for label, value in zip(LABEL_ORDER, per_class_recall)},
        "per_class_f1": {label: float(value) for label, value in zip(LABEL_ORDER, per_class_f1)},
        "pred_ids": y_pred.tolist(),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=True):
        confusion[int(truth), int(pred)] += 1
    return confusion


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *LABEL_ORDER])
        for label_name, row in zip(LABEL_ORDER, matrix.tolist(), strict=True):
            writer.writerow([label_name, *row])


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    output = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator != 0
    output[mask] = numerator[mask] / denominator[mask]
    return output
