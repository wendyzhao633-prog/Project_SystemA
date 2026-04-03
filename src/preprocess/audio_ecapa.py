from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from src.common.ravdess import LABEL_ORDER, build_av_key, parse_ravdess_stem

EXPECTED_SPLIT_COUNTS = {
    "train": 960,
    "valid": 240,
    "test": 240,
}

FUSION_TABLE_FIELDS = [
    "av_key",
    "sample_id",
    "stem",
    "wav_path",
    "actor_id",
    "split",
    "emotion_code",
    "label_id",
    "label_name",
    "intensity_code",
    "statement_code",
    "repetition_code",
    "pred_id",
    "pred_name",
    "prob_neutral",
    "prob_calm",
    "prob_happy",
    "prob_sad",
    "prob_angry",
    "prob_fearful",
    "prob_disgust",
    "prob_surprise",
    "embedding_path",
    "embedding_dim",
]


@dataclass(frozen=True)
class EcapaLayout:
    repo_root: Path
    audio_branch_root: Path
    upstream_hparams_path: Path
    upstream_evaluate_path: Path
    upstream_export_embeddings_path: Path
    results_parent: Path
    results_save_root: Path
    nested_predictions_root: Path
    nested_embeddings_root: Path
    canonical_predictions_root: Path
    canonical_embeddings_root: Path
    audio_manifest_dir: Path
    processed_audio_root: Path
    local_metadata_root: Path
    fusion_ready_root: Path


@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint_dir: Path
    ckpt_yaml_path: Path
    uar: float
    acc: float | None


def ensure_audio_runtime_dependencies(layout: EcapaLayout) -> None:
    missing: list[str] = []
    for module_name in ("hyperpyyaml", "torchaudio", "huggingface_hub", "sentencepiece"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        raise RuntimeError(
            "Missing Python dependencies required for ECAPA artifact generation: "
            + ", ".join(missing)
            + ". Install them in the active environment before running generate-missing. "
            + "Recommended commands:\n"
            + "  python -m pip install hyperpyyaml huggingface_hub sentencepiece\n"
            + "  python -m pip install torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128"
        )


def build_ecapa_layout(repo_root: Path | None = None) -> EcapaLayout:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    audio_branch_root = resolved_repo_root / "external/audio_ser_speechbrain/speechbrain-develop-main"
    recipe_root = audio_branch_root / "recipes/RAVDESS_emotion_recognition"
    outputs_root = audio_branch_root / "outputs"
    results_parent = recipe_root / "results/ECAPA-TDNN/ECAPA-TDNN/1968"

    return EcapaLayout(
        repo_root=resolved_repo_root,
        audio_branch_root=audio_branch_root,
        upstream_hparams_path=recipe_root / "hparams/train_ravdess.yaml",
        upstream_evaluate_path=recipe_root / "evaluate.py",
        upstream_export_embeddings_path=recipe_root / "export_embeddings.py",
        results_parent=results_parent,
        results_save_root=results_parent / "save",
        nested_predictions_root=outputs_root / "outputs/predictions_ECAPA",
        nested_embeddings_root=outputs_root / "outputs/embeddings_ECAPA",
        canonical_predictions_root=outputs_root / "predictions_ECAPA",
        canonical_embeddings_root=outputs_root / "embeddings_ECAPA",
        audio_manifest_dir=resolved_repo_root / "data/manifests/audio",
        processed_audio_root=resolved_repo_root / "data/processed/audio",
        local_metadata_root=resolved_repo_root / "data/processed/audio/ecapa_metadata",
        fusion_ready_root=resolved_repo_root / "data/processed/audio/fusion_ready",
    )


def find_best_checkpoint(save_root: Path) -> CheckpointInfo:
    ckpt_yaml_paths = sorted(Path(save_root).rglob("CKPT.yaml"))
    if not ckpt_yaml_paths:
        raise FileNotFoundError(f"No CKPT.yaml files found under '{save_root}'.")

    best_info: CheckpointInfo | None = None
    for ckpt_yaml_path in ckpt_yaml_paths:
        values = _parse_simple_yaml_scalars(ckpt_yaml_path)
        if "uar" not in values:
            raise ValueError(f"Checkpoint metadata missing 'uar': '{ckpt_yaml_path}'.")

        info = CheckpointInfo(
            checkpoint_dir=ckpt_yaml_path.parent,
            ckpt_yaml_path=ckpt_yaml_path,
            uar=float(values["uar"]),
            acc=float(values["acc"]) if "acc" in values else None,
        )
        if best_info is None or info.uar > best_info.uar:
            best_info = info

    assert best_info is not None
    return best_info


def create_local_speechbrain_metadata(layout: EcapaLayout) -> dict[str, Path]:
    manifest_rows = load_audio_manifest_rows(layout.audio_manifest_dir)
    expected_counts = expected_counts_from_manifests(manifest_rows)
    validate_expected_counts(expected_counts)

    layout.local_metadata_root.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, Path] = {}
    for split, rows in manifest_rows.items():
        payload = {}
        for row in rows:
            payload[row["sample_id"]] = {
                "wav": row["wav_path"],
                "actor": row["actor_id"],
                "emotion_code": row["emotion_code"],
                "label_id": row["label_id"],
                "label_name": row["label_name"],
                "split": row["split"],
            }

        output_path = layout.local_metadata_root / f"{split}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written_paths[split] = output_path

    return written_paths


def normalize_existing_audio_artifacts(layout: EcapaLayout) -> dict[str, Any]:
    manifest_rows = load_audio_manifest_rows(layout.audio_manifest_dir)
    expected_counts = expected_counts_from_manifests(manifest_rows)
    validate_expected_counts(expected_counts)
    sample_index = sample_index_from_manifests(manifest_rows)

    layout.canonical_predictions_root.mkdir(parents=True, exist_ok=True)
    layout.canonical_embeddings_root.mkdir(parents=True, exist_ok=True)

    normalized_prediction_counts: dict[str, int] = {}
    normalized_embedding_counts: dict[str, int] = {}

    for split in ("valid", "test"):
        source_predictions_path = layout.nested_predictions_root / f"{split}_predictions.jsonl"
        if source_predictions_path.is_file():
            destination_predictions_path = layout.canonical_predictions_root / f"{split}_predictions.jsonl"
            records = load_jsonl(source_predictions_path)
            normalized_records = normalize_prediction_records(records, split=split, sample_index=sample_index)
            write_jsonl(destination_predictions_path, normalized_records)
            normalized_prediction_counts[split] = len(normalized_records)

            source_metrics_path = layout.nested_predictions_root / f"{split}_metrics.json"
            destination_metrics_path = layout.canonical_predictions_root / f"{split}_metrics.json"
            if source_metrics_path.is_file():
                shutil.copy2(source_metrics_path, destination_metrics_path)
            else:
                destination_metrics_path.write_text(
                    json.dumps(compute_prediction_metrics(normalized_records), indent=2),
                    encoding="utf-8",
                )

    test_source_index = layout.nested_embeddings_root / "test_embedding_index.jsonl"
    test_source_dir = layout.nested_embeddings_root / "test"
    if test_source_dir.is_dir():
        normalized_embedding_counts["test"] = normalize_embedding_artifacts(
            split="test",
            source_dir=test_source_dir,
            destination_root=layout.canonical_embeddings_root,
            sample_index=sample_index,
            source_index_path=test_source_index if test_source_index.is_file() else None,
        )

    return {
        "prediction_counts": normalized_prediction_counts,
        "embedding_counts": normalized_embedding_counts,
        "best_checkpoint": find_best_checkpoint(layout.results_save_root),
    }


def generate_missing_audio_artifacts(layout: EcapaLayout) -> dict[str, Any]:
    ensure_audio_runtime_dependencies(layout)
    create_local_speechbrain_metadata(layout)
    manifest_rows = load_audio_manifest_rows(layout.audio_manifest_dir)
    expected_counts = expected_counts_from_manifests(manifest_rows)
    validate_expected_counts(expected_counts)
    missing = detect_missing_artifacts(layout, expected_counts=expected_counts)

    generated: dict[str, Any] = {"predictions": [], "embeddings": [], "indexes": []}

    if "train" in missing["predictions"]:
        generate_predictions_for_split(layout, split="train")
        generated["predictions"].append("train")

    for split in ("train", "valid", "test"):
        if split in missing["embedding_indexes"] and split not in missing["embeddings"]:
            rebuild_embedding_index_from_split_dir(layout, split=split, expected_count=expected_counts[split])
            generated["indexes"].append(split)

    for split in ("train", "valid"):
        if split in missing["embeddings"] or split in missing["embedding_indexes"]:
            generate_embeddings_for_split(layout, split=split)
            generated["embeddings"].append(split)

    for split in ("train", "valid", "test"):
        if split in missing["prediction_metrics"] and split not in missing["predictions"]:
            predictions_path = layout.canonical_predictions_root / f"{split}_predictions.jsonl"
            metrics_path = layout.canonical_predictions_root / f"{split}_metrics.json"
            metrics_path.write_text(
                json.dumps(compute_prediction_metrics(load_jsonl(predictions_path)), indent=2),
                encoding="utf-8",
            )

    return {
        "generated": generated,
        "best_checkpoint": find_best_checkpoint(layout.results_save_root),
    }


def build_fusion_ready_audio_tables(
    layout: EcapaLayout,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Path]:
    manifest_rows = load_audio_manifest_rows(layout.audio_manifest_dir)
    resolved_expected_counts = expected_counts or expected_counts_from_manifests(manifest_rows)
    if expected_counts is None:
        validate_expected_counts(resolved_expected_counts)

    split_paths: dict[str, Path] = {}
    layout.fusion_ready_root.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        manifest_index = {row["sample_id"]: row for row in manifest_rows[split]}
        predictions_path = layout.canonical_predictions_root / f"{split}_predictions.jsonl"
        embedding_index_path = layout.canonical_embeddings_root / f"{split}_embedding_index.jsonl"
        prediction_rows = load_jsonl(predictions_path)
        embedding_rows = load_jsonl(embedding_index_path)

        validate_split_counts({split: len(manifest_index)}, {split: resolved_expected_counts[split]})
        validate_split_counts({split: len(prediction_rows)}, {split: resolved_expected_counts[split]})
        validate_split_counts({split: len(embedding_rows)}, {split: resolved_expected_counts[split]})

        predictions_by_sample = {row["sample_id"]: row for row in prediction_rows}
        embeddings_by_sample = {row["sample_id"]: row for row in embedding_rows}

        if set(manifest_index) != set(predictions_by_sample):
            raise ValueError(f"Prediction sample ids do not match manifest sample ids for split '{split}'.")
        if set(manifest_index) != set(embeddings_by_sample):
            raise ValueError(f"Embedding sample ids do not match manifest sample ids for split '{split}'.")

        joined_rows: list[dict[str, Any]] = []
        for sample_id in sorted(manifest_index):
            manifest_row = manifest_index[sample_id]
            prediction_row = predictions_by_sample[sample_id]
            embedding_row = embeddings_by_sample[sample_id]
            parsed = parse_ravdess_stem(sample_id)

            if build_av_key(sample_id) != manifest_row["av_key"]:
                raise ValueError(f"av_key mismatch for sample '{sample_id}' in audio manifest.")

            row = {
                "av_key": manifest_row["av_key"],
                "sample_id": sample_id,
                "stem": manifest_row["stem"],
                "wav_path": manifest_row["wav_path"],
                "actor_id": manifest_row["actor_id"],
                "split": split,
                "emotion_code": manifest_row["emotion_code"],
                "label_id": manifest_row["label_id"],
                "label_name": manifest_row["label_name"],
                "intensity_code": manifest_row["intensity_code"],
                "statement_code": manifest_row["statement_code"],
                "repetition_code": manifest_row["repetition_code"],
                "pred_id": prediction_row["pred_id"],
                "pred_name": prediction_row["pred_name"],
                "embedding_path": embedding_row["embedding_path"],
                "embedding_dim": embedding_row["embedding_dim"],
            }

            probs = prediction_row["probs"]
            for label_name, prob in zip(LABEL_ORDER, probs):
                row[f"prob_{label_name}"] = prob

            if parsed.av_key != row["av_key"]:
                raise ValueError(f"Parsed av_key does not match row av_key for '{sample_id}'.")

            joined_rows.append(row)

        split_path = layout.fusion_ready_root / f"audio_ecapa_{split}.csv"
        write_csv(split_path, joined_rows, fieldnames=FUSION_TABLE_FIELDS)
        split_paths[split] = split_path
        combined_rows.extend(joined_rows)

    all_path = layout.fusion_ready_root / "audio_ecapa_all.csv"
    write_csv(all_path, combined_rows, fieldnames=FUSION_TABLE_FIELDS)
    split_paths["all"] = all_path
    return split_paths


def detect_missing_artifacts(
    layout: EcapaLayout,
    *,
    expected_counts: dict[str, int],
) -> dict[str, list[str]]:
    missing = {
        "predictions": [],
        "prediction_metrics": [],
        "embeddings": [],
        "embedding_indexes": [],
    }

    for split, expected_count in expected_counts.items():
        prediction_path = layout.canonical_predictions_root / f"{split}_predictions.jsonl"
        metrics_path = layout.canonical_predictions_root / f"{split}_metrics.json"
        embedding_dir = layout.canonical_embeddings_root / split
        embedding_index_path = layout.canonical_embeddings_root / f"{split}_embedding_index.jsonl"

        if not prediction_path.is_file() or count_jsonl_rows(prediction_path) != expected_count:
            missing["predictions"].append(split)
        if not metrics_path.is_file():
            missing["prediction_metrics"].append(split)
        if not embedding_dir.is_dir() or count_files(embedding_dir, ".npy") != expected_count:
            missing["embeddings"].append(split)
        if not embedding_index_path.is_file() or count_jsonl_rows(embedding_index_path) != expected_count:
            missing["embedding_indexes"].append(split)

    return missing


def load_audio_manifest_rows(audio_manifest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        csv_path = Path(audio_manifest_dir) / f"audio_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Audio manifest not found: '{csv_path}'.")

        rows = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_row = dict(row)
                normalized_row["actor_id"] = int(row["actor_id"])
                normalized_row["label_id"] = int(row["label_id"])
                rows.append(normalized_row)
        rows_by_split[split] = rows
    return rows_by_split


def sample_index_from_manifests(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    sample_index: dict[str, dict[str, Any]] = {}
    for rows in rows_by_split.values():
        for row in rows:
            sample_index[row["sample_id"]] = row
    return sample_index


def expected_counts_from_manifests(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {split: len(rows) for split, rows in rows_by_split.items()}


def validate_expected_counts(actual_counts: dict[str, int]) -> None:
    validate_split_counts(actual_counts, EXPECTED_SPLIT_COUNTS)


def validate_split_counts(actual_counts: dict[str, int], expected_counts: dict[str, int]) -> None:
    for split, expected_count in expected_counts.items():
        actual_count = actual_counts.get(split)
        if actual_count != expected_count:
            raise ValueError(
                f"Count mismatch for split '{split}': expected {expected_count}, got {actual_count}."
            )


def normalize_prediction_records(
    records: Iterable[dict[str, Any]],
    *,
    split: str,
    sample_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in sample_index:
            raise KeyError(f"Prediction sample_id '{sample_id}' not found in audio manifests.")

        manifest_row = sample_index[sample_id]
        probs = record["probs"]
        validate_probability_vector(probs, sample_id=sample_id)

        if split != manifest_row["split"]:
            raise ValueError(
                f"Prediction split mismatch for '{sample_id}': expected {manifest_row['split']}, got {split}."
            )

        pred_id = int(record["pred_id"])
        if pred_id < 0 or pred_id >= len(LABEL_ORDER):
            raise ValueError(f"pred_id out of range for sample '{sample_id}': {pred_id}.")

        normalized_records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "actor": manifest_row["actor_id"],
                "actor_id": manifest_row["actor_id"],
                "av_key": manifest_row["av_key"],
                "wav": manifest_row["wav_path"],
                "wav_path": manifest_row["wav_path"],
                "emotion_code": manifest_row["emotion_code"],
                "label_id": manifest_row["label_id"],
                "label_name": manifest_row["label_name"],
                "pred_id": pred_id,
                "pred_name": str(record["pred_name"]),
                "probs": [float(prob) for prob in probs],
            }
        )

    return sorted(normalized_records, key=lambda row: row["sample_id"])


def normalize_embedding_artifacts(
    *,
    split: str,
    source_dir: Path,
    destination_root: Path,
    sample_index: dict[str, dict[str, Any]],
    source_index_path: Path | None = None,
) -> int:
    source_dir = Path(source_dir)
    destination_root = Path(destination_root)
    destination_split_dir = destination_root / split
    destination_split_dir.mkdir(parents=True, exist_ok=True)

    if source_index_path is not None and Path(source_index_path).is_file():
        source_records = load_jsonl(source_index_path)
        sample_ids = [str(record["sample_id"]) for record in source_records]
    else:
        sample_ids = sorted(path.stem for path in source_dir.glob("*.npy"))

    index_records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        if sample_id not in sample_index:
            raise KeyError(f"Embedding sample_id '{sample_id}' not found in audio manifests.")

        source_path = source_dir / f"{sample_id}.npy"
        if not source_path.is_file():
            raise FileNotFoundError(f"Embedding file missing for '{sample_id}': '{source_path}'.")

        destination_path = destination_split_dir / f"{sample_id}.npy"
        shutil.copy2(source_path, destination_path)
        validate_embedding_file(destination_path)

        manifest_row = sample_index[sample_id]
        index_records.append(
            {
                "sample_id": sample_id,
                "av_key": manifest_row["av_key"],
                "embedding_path": destination_path.as_posix(),
                "embedding_dim": 96,
                "label_id": manifest_row["label_id"],
                "split": split,
            }
        )

    index_path = destination_root / f"{split}_embedding_index.jsonl"
    write_jsonl(index_path, sorted(index_records, key=lambda row: row["sample_id"]))
    return len(index_records)


def rebuild_embedding_index_from_split_dir(
    layout: EcapaLayout,
    *,
    split: str,
    expected_count: int,
) -> Path:
    manifest_rows = load_audio_manifest_rows(layout.audio_manifest_dir)
    sample_index = sample_index_from_manifests(manifest_rows)
    split_dir = layout.canonical_embeddings_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Canonical embedding split directory not found: '{split_dir}'.")

    index_records: list[dict[str, Any]] = []
    for embedding_path in sorted(split_dir.glob("*.npy")):
        validate_embedding_file(embedding_path)
        sample_id = embedding_path.stem
        manifest_row = sample_index[sample_id]
        index_records.append(
            {
                "sample_id": sample_id,
                "av_key": manifest_row["av_key"],
                "embedding_path": embedding_path.as_posix(),
                "embedding_dim": 96,
                "label_id": manifest_row["label_id"],
                "split": split,
            }
        )

    validate_split_counts({split: len(index_records)}, {split: expected_count})
    index_path = layout.canonical_embeddings_root / f"{split}_embedding_index.jsonl"
    write_jsonl(index_path, index_records)
    return index_path


def generate_predictions_for_split(layout: EcapaLayout, *, split: str) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported prediction split '{split}'.")

    create_local_speechbrain_metadata(layout)
    evaluate_module = _load_upstream_module("ravdess_eval_upstream", layout.upstream_evaluate_path)
    hparams = _load_runtime_hparams(layout)

    import speechbrain as sb
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_opts = {"device": device}
    brain = evaluate_module.build_model_and_checkpointer(hparams, run_opts)
    brain.on_evaluate_start(max_key="uar")
    modules = brain.modules

    manifest_path = layout.local_metadata_root / f"{split}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Local metadata missing: '{manifest_path}'.")

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        if sig.ndim == 2:
            sig = sig.mean(dim=1)
        return sig

    @sb.utils.data_pipeline.takes("label_id")
    @sb.utils.data_pipeline.provides("label_id", "label_id_encoded")
    def label_pipeline(label_id):
        yield label_id
        yield torch.LongTensor([label_id])

    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=manifest_path.as_posix(),
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig", "label_id_encoded"],
    )

    results = evaluate_module.run_inference(
        modules=modules,
        dataset=dataset,
        hparams=hparams,
        device=device,
        split_name=split,
        manifest_path=manifest_path.as_posix(),
    )
    metrics = evaluate_module.compute_metrics(
        [row["pred_id"] for row in results],
        [row["label_id"] for row in results],
    )

    output_path = layout.canonical_predictions_root / f"{split}_predictions.jsonl"
    metrics_path = layout.canonical_predictions_root / f"{split}_metrics.json"
    layout.canonical_predictions_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, results)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def generate_embeddings_for_split(layout: EcapaLayout, *, split: str) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported embedding split '{split}'.")

    create_local_speechbrain_metadata(layout)
    export_module = _load_upstream_module("ravdess_export_upstream", layout.upstream_export_embeddings_path)
    hparams = _load_runtime_hparams(layout)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_opts = {"device": device}
    brain = export_module.build_brain_for_restore(hparams, run_opts)
    brain.on_evaluate_start(max_key="uar")

    export_module.export_embeddings_for_split(
        modules=brain.modules,
        hparams=hparams,
        device=device,
        split_name=split,
        manifest_path=(layout.local_metadata_root / f"{split}.json").as_posix(),
        embeddings_root=layout.canonical_embeddings_root.as_posix(),
    )

    return layout.canonical_embeddings_root / f"{split}_embedding_index.jsonl"


def compute_prediction_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(record["label_id"]) for record in records]
    preds = [int(record["pred_id"]) for record in records]
    if not labels:
        raise ValueError("Cannot compute metrics from an empty prediction set.")

    confusion_matrix = [[0 for _ in LABEL_ORDER] for _ in LABEL_ORDER]
    per_class_recall: dict[str, float | None] = {}

    correct = 0
    for label_id, pred_id in zip(labels, preds):
        confusion_matrix[label_id][pred_id] += 1
        if label_id == pred_id:
            correct += 1

    recalls: list[float] = []
    for label_id, label_name in enumerate(LABEL_ORDER):
        row_total = sum(confusion_matrix[label_id])
        if row_total == 0:
            per_class_recall[label_name] = None
            continue
        recall = confusion_matrix[label_id][label_id] / row_total
        per_class_recall[label_name] = round(recall, 4)
        recalls.append(recall)

    accuracy = correct / len(labels)
    uar = sum(recalls) / len(recalls)
    return {
        "accuracy": round(accuracy, 4),
        "uar": round(uar, 4),
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion_matrix,
    }


def validate_probability_vector(probs: Any, *, sample_id: str) -> None:
    if not isinstance(probs, list) or len(probs) != 8:
        raise ValueError(f"Probability vector for '{sample_id}' must have length 8.")


def validate_embedding_file(path: Path) -> None:
    embedding = np.load(path)
    if embedding.shape != (96,):
        raise ValueError(f"Embedding at '{path}' must have shape (96,), got {embedding.shape}.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_jsonl_rows(path: Path) -> int:
    return len(load_jsonl(path))


def count_files(directory: Path, suffix: str) -> int:
    return sum(1 for path in Path(directory).glob(f"*{suffix}") if path.is_file())


def _parse_simple_yaml_scalars(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        values[key.strip()] = _coerce_scalar(raw_value.strip())
    return values


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_upstream_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream module from '{module_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime_hparams(layout: EcapaLayout) -> dict[str, Any]:
    from hyperpyyaml import load_hyperpyyaml

    overrides = {
        "metadata_dir": layout.local_metadata_root.as_posix(),
        "train_annotation": (layout.local_metadata_root / "train.json").as_posix(),
        "valid_annotation": (layout.local_metadata_root / "valid.json").as_posix(),
        "test_annotation": (layout.local_metadata_root / "test.json").as_posix(),
        "results_root": layout.results_parent.parent.as_posix(),
        "output_folder": layout.results_parent.as_posix(),
        "save_folder": layout.results_save_root.as_posix(),
        "train_log": (layout.results_parent / "train_log.txt").as_posix(),
        "predictions_dir": layout.canonical_predictions_root.as_posix(),
        "embeddings_dir": layout.canonical_embeddings_root.as_posix(),
    }

    with layout.upstream_hparams_path.open(encoding="utf-8") as handle:
        return load_hyperpyyaml(handle, overrides)
