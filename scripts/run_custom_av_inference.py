"""
scripts/run_custom_av_inference.py

End-to-end inference pipeline for 200 custom audio+video files.

Steps
-----
1. ECAPA-TDNN  — predict 8-class probs + extract 96-dim embeddings for every
                 custom .wav file.
2. Video backbone — extract sequence features (5×2048) + 8-class probs for
                    every custom .mp4 file.
3. Late fusion (weighted_prob_avg) using the fitted audio_weight / video_weight
   from checkpoints/late_fusion/ecapa_video_reuse_baseline/weighted_prob_avg/
   fitted_weight.json.

File naming convention (input)
-------------------------------
  data/raw/custom/audio/audio_P_NN.wav
  data/raw/custom/video/video_P_NN.mp4

  P  = participant number (1–4, 1 digit)
  NN = local sample number within participant (zero-padded to 2 digits, 01–50)

  av_key  = "{P}_{G}"  where G = (P-1)*50 + NN  (1-based global index)
  split   : P1 + P2 → train  |  P3 → valid  |  P4 → test

NOTE: No ground-truth emotion labels are embedded in the filenames.
      All rows use label_id=0 / label_name="neutral" as a placeholder.
      Per-split accuracy / UAR figures therefore have NO diagnostic meaning
      unless you supply a real label CSV via --labels-csv.

Outputs
-------
  data/processed/audio/fusion_ready/audio_ecapa_custom_{split}.csv
  data/processed/video/fusion_ready/video_reuse_custom_{split}.csv
  data/processed/av/late_fusion_custom/av_ecapa_video_reuse_custom_{split}.csv
  outputs/metrics/late_fusion_custom/weighted_prob_avg/{split}_metrics.json
  outputs/predictions/late_fusion_custom/weighted_prob_avg/{split}_predictions.jsonl

Usage
-----
  python scripts/run_custom_av_inference.py
  python scripts/run_custom_av_inference.py --labels-csv data/raw/custom/labels.csv

  Labels CSV columns (required): participant_id, sample_no, label_id  (other columns ignored)
    sample_no is the global 1-based index (1-200).
    e.g.:  1,1,3   → P1 global sample 1 is "sad" (label_id 3)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import random
import torch
import torch.nn.functional as F
from dataclasses import asdict
from torch.utils.data import DataLoader, TensorDataset

# ── Repo root (two levels up from scripts/) ───────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.ravdess import LABEL_ORDER
from src.preprocess.audio_ecapa import (
    EcapaLayout,
    build_ecapa_layout,
    _load_upstream_module,
)
from src.preprocess.video_actor15_backfill import (
    build_face_detector,
    extract_video_frames,
)
from src.models.video_reference_resnet import (
    load_reference_backbone,
    load_reference_expression_model,
)
from src.fusion.data import load_audio_embeddings, load_video_sequence_embeddings
from src.fusion.metrics import compute_classification_metrics, write_confusion_matrix_csv
from src.fusion.models import EmbeddingMLP, EmbeddingMLPConfig
from src.preprocess.ppg_features import extract_ppg_features, features_to_array

# ── Tunable constants ─────────────────────────────────────────────────────────
SAMPLES_PER_PARTICIPANT = 50
FRAMES_PER_VIDEO = 5
IMAGE_SIZE = 256
EXPECTED_SEQUENCE_SHAPE = (5, 2048)

AUDIO_STEM_RE = re.compile(r"^audio_(\d+)_(\d+)$")
VIDEO_STEM_RE = re.compile(r"^video_(\d+)_(\d+)$")

AUDIO_FUSION_FIELDS = [
    "av_key", "sample_id", "stem", "wav_path", "actor_id", "split",
    "emotion_code", "label_id", "label_name", "intensity_code",
    "statement_code", "repetition_code", "pred_id", "pred_name",
    "prob_neutral", "prob_calm", "prob_happy", "prob_sad",
    "prob_angry", "prob_fearful", "prob_disgust", "prob_surprise",
    "embedding_path", "embedding_dim",
]

VIDEO_FUSION_FIELDS = [
    "av_key", "sample_id", "stem", "video_path", "actor_id", "split",
    "emotion_code", "label_id", "label_name", "intensity_code",
    "statement_code", "repetition_code",
    "sequence_video_id", "sequence_video_id_source",
    "sequence_path", "sequence_num_frames", "sequence_feature_dim",
    "prediction_video_id", "prediction_video_id_source",
    "prediction_frame_count", "pred_id", "pred_name",
    "prob_neutral", "prob_calm", "prob_happy", "prob_sad",
    "prob_angry", "prob_fearful", "prob_disgust", "prob_surprise",
    "has_sequence", "has_prediction", "late_fusion_ready", "artifact_status",
]

AV_JOIN_FIELDS = [
    "av_key", "split", "actor_id", "label_id", "label_name",
    "emotion_code", "intensity_code", "statement_code", "repetition_code",
    "audio_sample_id", "audio_wav_path", "audio_pred_id", "audio_pred_name",
    "audio_embedding_path", "audio_embedding_dim",
    "video_sample_id", "video_path", "video_pred_id", "video_pred_name",
    "video_sequence_path", "video_sequence_num_frames", "video_sequence_feature_dim",
    "video_sequence_source", "video_prediction_source",
] + [f"audio_prob_{ln}" for ln in LABEL_ORDER] + [f"video_prob_{ln}" for ln in LABEL_ORDER]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  File discovery & metadata
# ═══════════════════════════════════════════════════════════════════════════════

def participant_to_split(p: int) -> str:
    if p in (1, 2):
        return "train"
    if p == 3:
        return "valid"
    if p == 4:
        return "test"
    raise ValueError(f"Participant {p} out of range 1–4.")


def make_av_key(p: int, local_n: int) -> str:
    global_n = (p - 1) * SAMPLES_PER_PARTICIPANT + local_n
    return f"{p}_{global_n}"


def collect_custom_samples(
    repo_root: Path,
    labels_csv: Path | None,
) -> list[dict[str, Any]]:
    """Scan data/raw/custom/{audio,video}/ and build one record per AV pair."""
    audio_dir = repo_root / "data/raw/custom/audio"
    video_dir = repo_root / "data/raw/custom/video"

    audio_files: dict[tuple[int, int], Path] = {}
    for f in sorted(audio_dir.glob("audio_*.wav")):
        m = AUDIO_STEM_RE.match(f.stem)
        if m:
            audio_files[(int(m.group(1)), int(m.group(2)))] = f

    video_files: dict[tuple[int, int], Path] = {}
    for f in sorted(video_dir.glob("video_*.mp4")):
        m = VIDEO_STEM_RE.match(f.stem)
        if m:
            video_files[(int(m.group(1)), int(m.group(2)))] = f

    # Optional label lookup  {(p, local_n): label_id}
    label_lookup: dict[tuple[int, int], int] = {}
    if labels_csv is not None:
        label_lookup = _load_labels_csv(labels_csv)

    all_keys = sorted(audio_files.keys() | video_files.keys())
    samples: list[dict[str, Any]] = []

    for p, n in all_keys:
        audio_path = audio_files.get((p, n))
        video_path = video_files.get((p, n))
        if audio_path is None:
            print(f"[WARN] No audio file for participant {p} sample {n} – skipping.")
            continue
        if video_path is None:
            print(f"[WARN] No video file for participant {p} sample {n} – skipping.")
            continue

        label_id = label_lookup.get((p, n), 0)
        samples.append({
            "participant": p,
            "local_n": n,
            "av_key": make_av_key(p, n),
            "split": participant_to_split(p),
            "audio_stem": audio_path.stem,       # e.g. "audio_1_01"
            "video_stem": video_path.stem,        # e.g. "video_1_01"
            "audio_path": audio_path,
            "video_path": video_path,
            "actor_id": p,
            "label_id": label_id,
            "label_name": LABEL_ORDER[label_id],
            "emotion_code": f"{label_id + 1:02d}",
            "intensity_code": "01",
            "statement_code": "01",
            "repetition_code": "01",
        })

    print(f"[INFO] Discovered {len(samples)} AV pairs "
          f"({sum(1 for s in samples if s['split']=='train')} train / "
          f"{sum(1 for s in samples if s['split']=='valid')} valid / "
          f"{sum(1 for s in samples if s['split']=='test')} test)")
    return samples


def _load_labels_csv(path: Path) -> dict[tuple[int, int], int]:
    """Load optional label file.

    Expected columns: participant_id, sample_no, label_id  (plus any others).
    sample_no is the global 1-based index (1-200); local_n is derived as
    ((sample_no - 1) % 50) + 1 so that samples 1-50 → 1-50, 51-100 → 1-50, etc.
    """
    lookup: dict[tuple[int, int], int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            p         = int(row["participant_id"])
            sample_no = int(row["sample_no"])
            lid       = int(row["label_id"])
            local_n   = ((sample_no - 1) % SAMPLES_PER_PARTICIPANT) + 1
            lookup[(p, local_n)] = lid
    print(f"[INFO] Loaded {len(lookup)} labels from {path}.")
    return lookup


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ECAPA-TDNN audio inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_ecapa_inference(
    samples: list[dict[str, Any]],
    repo_root: Path,
    *,
    metadata_root: Path | None = None,
    predictions_root: Path | None = None,
    embeddings_root: Path | None = None,
) -> dict[str, Any]:
    """
    Returns
    -------
    {
      "predictions": {split: [{"sample_id": ..., "pred_id": ..., "probs": [...]}]},
      "embeddings":  {split: {"sample_id": {"embedding_path": ..., ...}}},
    }
    """
    base_layout = build_ecapa_layout(repo_root)
    sb_root = str(base_layout.audio_branch_root)
    if sb_root not in sys.path:
        sys.path.insert(0, sb_root)

    from hyperpyyaml import load_hyperpyyaml
    import speechbrain as sb  # noqa: imported here after sys.path is set

    # Add the local SpeechBrain branch to the front of sys.path so that
    # 'import speechbrain' picks it up even if an older SB is installed.

    # Paths for custom artefacts
    meta_root = metadata_root or (repo_root / "data/processed/audio/ecapa_metadata_custom")
    pred_root = predictions_root or (repo_root / "data/processed/audio/predictions_ecapa_custom")
    emb_root  = embeddings_root or (repo_root / "data/processed/audio/embeddings_ecapa_custom")
    meta_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    emb_root.mkdir(parents=True, exist_ok=True)

    # Write SpeechBrain-format metadata JSONs (one per split)
    _write_custom_metadata_jsons(samples, meta_root)

    # Build hparams pointing at the original checkpoint but custom data dirs
    overrides = {
        "metadata_dir":       meta_root.as_posix(),
        "train_annotation":   (meta_root / "train.json").as_posix(),
        "valid_annotation":   (meta_root / "valid.json").as_posix(),
        "test_annotation":    (meta_root / "test.json").as_posix(),
        "results_root":       base_layout.results_parent.parent.as_posix(),
        "output_folder":      base_layout.results_parent.as_posix(),
        "save_folder":        base_layout.results_save_root.as_posix(),
        "train_log":          (base_layout.results_parent / "train_log.txt").as_posix(),
        "predictions_dir":    pred_root.as_posix(),
        "embeddings_dir":     emb_root.as_posix(),
    }
    with base_layout.upstream_hparams_path.open(encoding="utf-8") as fh:
        hparams = load_hyperpyyaml(fh, overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_opts = {"device": device}
    print(f"[ECAPA] device={device}")

    # Load model and restore best checkpoint
    evaluate_module = _load_upstream_module(
        "ravdess_eval_upstream", base_layout.upstream_evaluate_path
    )
    export_module = _load_upstream_module(
        "ravdess_export_upstream", base_layout.upstream_export_embeddings_path
    )

    brain = evaluate_module.build_model_and_checkpointer(hparams, run_opts)
    brain.on_evaluate_start(max_key="uar")
    modules = brain.modules
    print("[ECAPA] Best checkpoint restored.")

    # Define pipelines once (SpeechBrain decorators create independent objects)
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

    all_predictions: dict[str, list[dict]] = {}
    all_embeddings:  dict[str, dict[str, dict]] = {}

    for split in ("train", "valid", "test"):
        manifest_path = meta_root / f"{split}.json"
        if not manifest_path.is_file():
            # Empty split (possible if no files for this split)
            all_predictions[split] = []
            all_embeddings[split] = {}
            continue

        # --- Predictions ---
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=manifest_path.as_posix(),
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "label_id_encoded"],
        )
        pred_results = evaluate_module.run_inference(
            modules=modules,
            dataset=dataset,
            hparams=hparams,
            device=device,
            split_name=split,
            manifest_path=manifest_path.as_posix(),
        )
        all_predictions[split] = pred_results
        _write_jsonl(pred_root / f"{split}_predictions.jsonl", pred_results)

        # --- Embeddings ---
        export_module.export_embeddings_for_split(
            modules=modules,
            hparams=hparams,
            device=device,
            split_name=split,
            manifest_path=manifest_path.as_posix(),
            embeddings_root=emb_root.as_posix(),
        )
        idx_path = emb_root / f"{split}_embedding_index.jsonl"
        emb_records = _load_jsonl(idx_path)
        all_embeddings[split] = {r["sample_id"]: r for r in emb_records}
        print(f"[ECAPA] {split}: {len(pred_results)} predictions, "
              f"{len(emb_records)} embeddings.")

    return {"predictions": all_predictions, "embeddings": all_embeddings}


def _write_custom_metadata_jsons(
    samples: list[dict[str, Any]],
    meta_root: Path,
) -> None:
    """Write SpeechBrain-format annotation JSONs for train / valid / test."""
    split_payloads: dict[str, dict] = {"train": {}, "valid": {}, "test": {}}
    for s in samples:
        split_payloads[s["split"]][s["audio_stem"]] = {
            "wav":          s["audio_path"].as_posix(),
            "actor":        s["actor_id"],
            "emotion_code": s["emotion_code"],
            "label_id":     s["label_id"],
            "label_name":   s["label_name"],
            "split":        s["split"],
        }
    for split, payload in split_payloads.items():
        if not payload:
            continue
        out_path = meta_root / f"{split}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[ECAPA] Metadata JSONs written.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Build audio fusion-ready CSV
# ═══════════════════════════════════════════════════════════════════════════════

def build_audio_fusion_csvs(
    samples: list[dict[str, Any]],
    ecapa_results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}

    for split in ("train", "valid", "test"):
        split_samples = [s for s in samples if s["split"] == split]
        predictions_by_sid = {
            r["sample_id"]: r for r in ecapa_results["predictions"][split]
        }
        embeddings_by_sid = ecapa_results["embeddings"][split]

        rows: list[dict] = []
        for s in split_samples:
            sid = s["audio_stem"]
            pred = predictions_by_sid.get(sid)
            emb  = embeddings_by_sid.get(sid)
            if pred is None or emb is None:
                raise KeyError(
                    f"Missing ECAPA artefact for audio sample '{sid}' (split={split})."
                )
            row: dict[str, Any] = {
                "av_key":          s["av_key"],
                "sample_id":       sid,
                "stem":            sid,
                "wav_path":        s["audio_path"].as_posix(),
                "actor_id":        s["actor_id"],
                "split":           split,
                "emotion_code":    s["emotion_code"],
                "label_id":        s["label_id"],
                "label_name":      s["label_name"],
                "intensity_code":  s["intensity_code"],
                "statement_code":  s["statement_code"],
                "repetition_code": s["repetition_code"],
                "pred_id":         pred["pred_id"],
                "pred_name":       pred["pred_name"],
                "embedding_path":  emb["embedding_path"],
                "embedding_dim":   emb["embedding_dim"],
            }
            for label_name, prob in zip(LABEL_ORDER, pred["probs"]):
                row[f"prob_{label_name}"] = prob
            rows.append(row)

        out_path = output_dir / f"audio_ecapa_custom_{split}.csv"
        _write_csv(out_path, rows, fieldnames=AUDIO_FUSION_FIELDS)
        split_paths[split] = out_path
        print(f"[Audio CSV] {split}: {len(rows)} rows → {out_path}")

    return split_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Video backbone inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_video_inference(
    samples: list[dict[str, Any]],
    repo_root: Path,
    *,
    sequence_root: Path | None = None,
    predictions_root: Path | None = None,
) -> dict[str, Any]:
    """
    Returns
    -------
    {
      "sequences":   {split: {"video_stem": {"sequence_path": ..., ...}}},
      "predictions": {split: {"video_stem": {"pred_id": ..., "probs": [...], ...}}},
    }
    """
    package_root = (
        repo_root / "external/video_face_branch/early_fusion_facial_v2"
    )
    backbone_path = package_root / "best_backbone.pth"
    uar_path      = package_root / "best_uar.pth"

    if not backbone_path.is_file():
        raise FileNotFoundError(f"Video backbone checkpoint not found: {backbone_path}")
    if not uar_path.is_file():
        raise FileNotFoundError(f"Video expression model not found: {uar_path}")

    seq_root  = sequence_root or (repo_root / "data/processed/video/sequences_custom")
    seq_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Video] device={device}")

    backbone_model   = load_reference_backbone(backbone_path, device=device)
    expression_model = load_reference_expression_model(uar_path, device=device)
    face_detector    = build_face_detector()

    all_sequences:   dict[str, dict[str, dict]] = {"train": {}, "valid": {}, "test": {}}
    all_predictions: dict[str, dict[str, dict]] = {"train": {}, "valid": {}, "test": {}}

    for idx, s in enumerate(samples):
        split      = s["split"]
        video_stem = s["video_stem"]
        video_path = s["video_path"]

        frames, _frame_indices, _face_hits = extract_video_frames(
            video_path,
            frames_per_video=FRAMES_PER_VIDEO,
            image_size=IMAGE_SIZE,
            face_detector=face_detector,
        )
        if frames.shape != (FRAMES_PER_VIDEO, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Unexpected frame tensor shape for '{video_stem}': {tuple(frames.shape)}."
            )

        inputs = frames.to(device)
        with torch.no_grad():
            sequence_tensor = backbone_model(inputs).cpu().numpy().astype(np.float32)
            _, logits       = expression_model(inputs)
            frame_probs     = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

        if sequence_tensor.shape != EXPECTED_SEQUENCE_SHAPE:
            raise ValueError(
                f"Sequence shape mismatch for '{video_stem}': "
                f"expected {EXPECTED_SEQUENCE_SHAPE}, got {tuple(sequence_tensor.shape)}."
            )

        mean_probs = frame_probs.mean(axis=0)
        pred_id    = int(np.argmax(mean_probs))

        seq_path = (seq_root / f"{video_stem}.npy").resolve()
        np.save(seq_path, sequence_tensor)

        all_sequences[split][video_stem] = {
            "sequence_path":      seq_path.as_posix(),
            "sequence_num_frames": FRAMES_PER_VIDEO,
            "sequence_feature_dim": EXPECTED_SEQUENCE_SHAPE[1],
        }
        all_predictions[split][video_stem] = {
            "pred_id":   pred_id,
            "pred_name": LABEL_ORDER[pred_id],
            "probs":     [float(v) for v in mean_probs.tolist()],
        }

        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            print(f"[Video] processed {idx + 1}/{len(samples)}")

    # Persist predictions to disk so --skip-inference can reload them later.
    vid_pred_root = predictions_root or (repo_root / "data/processed/video/predictions_custom")
    vid_pred_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        records = [
            {"video_stem": vs, **pred_data}
            for vs, pred_data in all_predictions[split].items()
        ]
        _write_jsonl(vid_pred_root / f"{split}_predictions.jsonl", records)

    return {"sequences": all_sequences, "predictions": all_predictions}


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Build video fusion-ready CSV
# ═══════════════════════════════════════════════════════════════════════════════

def build_video_fusion_csvs(
    samples: list[dict[str, Any]],
    video_results: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}

    for split in ("train", "valid", "test"):
        split_samples = [s for s in samples if s["split"] == split]
        seqs  = video_results["sequences"][split]
        preds = video_results["predictions"][split]

        rows: list[dict] = []
        for s in split_samples:
            vstem = s["video_stem"]
            seq   = seqs.get(vstem)
            pred  = preds.get(vstem)
            if seq is None or pred is None:
                raise KeyError(
                    f"Missing video artefact for '{vstem}' (split={split})."
                )
            row: dict[str, Any] = {
                "av_key":                    s["av_key"],
                "sample_id":                 vstem,
                "stem":                      vstem,
                "video_path":                s["video_path"].as_posix(),
                "actor_id":                  s["actor_id"],
                "split":                     split,
                "emotion_code":              s["emotion_code"],
                "label_id":                  s["label_id"],
                "label_name":                s["label_name"],
                "intensity_code":            s["intensity_code"],
                "statement_code":            s["statement_code"],
                "repetition_code":           s["repetition_code"],
                "sequence_video_id":         vstem,
                "sequence_video_id_source":  "custom_inference",
                "sequence_path":             seq["sequence_path"],
                "sequence_num_frames":       seq["sequence_num_frames"],
                "sequence_feature_dim":      seq["sequence_feature_dim"],
                "prediction_video_id":       vstem,
                "prediction_video_id_source":"custom_inference",
                "prediction_frame_count":    FRAMES_PER_VIDEO,
                "pred_id":                   pred["pred_id"],
                "pred_name":                 pred["pred_name"],
                "has_sequence":              "1",
                "has_prediction":            "1",
                "late_fusion_ready":         "1",
                "artifact_status":           "complete",
            }
            for label_name, prob in zip(LABEL_ORDER, pred["probs"]):
                row[f"prob_{label_name}"] = prob
            rows.append(row)

        out_path = output_dir / f"video_reuse_custom_{split}.csv"
        _write_csv(out_path, rows, fieldnames=VIDEO_FUSION_FIELDS)
        split_paths[split] = out_path
        print(f"[Video CSV] {split}: {len(rows)} rows → {out_path}")

    return split_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Late fusion — weighted probability average
# ═══════════════════════════════════════════════════════════════════════════════

def run_late_fusion(
    repo_root: Path,
    audio_csv_dir: Path,
    video_csv_dir: Path,
) -> dict[str, Any]:
    # Load fitted weights from RAVDESS training
    weight_path = (
        repo_root
        / "checkpoints/late_fusion/ecapa_video_reuse_baseline"
        / "weighted_prob_avg/fitted_weight.json"
    )
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Fitted weight file not found: {weight_path}\n"
            "Run the RAVDESS late fusion first to generate this file."
        )
    weight_cfg   = json.loads(weight_path.read_text(encoding="utf-8"))
    audio_weight = float(weight_cfg["audio_weight"])
    video_weight = float(weight_cfg["video_weight"])
    print(f"[Fusion] Using audio_weight={audio_weight}, video_weight={video_weight} "
          f"(loaded from {weight_path.name})")

    # Output directories
    run_name    = "custom_inference"
    metrics_dir = repo_root / "outputs/metrics/late_fusion_custom/weighted_prob_avg"
    preds_dir   = repo_root / "outputs/predictions/late_fusion_custom/weighted_prob_avg"
    av_join_dir = repo_root / "data/processed/av/late_fusion_custom"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    av_join_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    for split in ("train", "valid", "test"):
        audio_rows = _load_csv_rows(audio_csv_dir / f"audio_ecapa_custom_{split}.csv")
        video_rows = _load_csv_rows(video_csv_dir / f"video_reuse_custom_{split}.csv")

        # Build AV join table
        av_rows = _join_audio_video(audio_rows, video_rows, split=split)
        av_path = av_join_dir / f"av_ecapa_video_reuse_custom_{split}.csv"
        _write_csv(av_path, av_rows, fieldnames=AV_JOIN_FIELDS)

        # Probability matrices
        audio_probs = _load_prob_matrix(av_rows, prefix="audio")
        video_probs = _load_prob_matrix(av_rows, prefix="video")
        fused_probs = audio_weight * audio_probs + video_weight * video_probs

        labels     = np.array([int(r["label_id"]) for r in av_rows], dtype=np.int64)
        pred_ids   = np.argmax(fused_probs, axis=1).tolist()
        metrics    = compute_classification_metrics(labels, fused_probs)

        # Write predictions JSONL
        pred_path = preds_dir / f"{split}_predictions.jsonl"
        _write_fusion_predictions(
            pred_path, av_rows, fused_probs, pred_ids,
            audio_weight=audio_weight, video_weight=video_weight,
        )

        # Write confusion matrix
        cm_path = metrics_dir / f"{split}_confusion_matrix.csv"
        write_confusion_matrix_csv(cm_path, np.asarray(metrics["confusion_matrix"], dtype=np.int64))

        # Write metrics JSON
        metrics_payload: dict[str, Any] = {
            "split":             split,
            "mode":              "weighted_prob_avg",
            "audio_weight":      audio_weight,
            "video_weight":      video_weight,
            "accuracy":          metrics["accuracy"],
            "macro_f1":          metrics["macro_f1"],
            "uar":               metrics["uar"],
            "per_class_recall":  metrics["per_class_recall"],
            "per_class_f1":      metrics["per_class_f1"],
            "labels_are_placeholder": labels_are_placeholder(labels),
        }
        metrics_out = metrics_dir / f"{split}_metrics.json"
        metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        summary[split] = metrics_payload
        print(
            f"[Fusion] {split}: acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} uar={metrics['uar']:.4f}"
            + (" (⚠ placeholder labels)" if metrics_payload["labels_are_placeholder"] else "")
        )

    return summary


def labels_are_placeholder(labels: np.ndarray) -> bool:
    """True when all labels are 0 (the default placeholder)."""
    return bool(np.all(labels == 0))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Late fusion — EmbeddingMLP with PPG features
# ═══════════════════════════════════════════════════════════════════════════════

def _ppg_csv_path(repo_root: Path, participant: int, local_n: int) -> Path:
    return repo_root / "data/raw/custom/ppg" / f"ppg_{participant}_{local_n:02d}.csv"


def extract_ppg_features_for_samples(
    samples: list[dict[str, Any]],
    repo_root: Path,
    *,
    warmup_sec: float = 45.0,
    fs: int = 128,
    ppg_dim: int = 11,
) -> dict[str, np.ndarray]:
    """Extract PPG features from raw CSVs for every sample.

    Returns a dict mapping av_key → float32 array of shape (ppg_dim,).
    """
    ppg_map: dict[str, np.ndarray] = {}
    for s in samples:
        p = s["participant"]
        local_n = s["local_n"]
        csv_path = _ppg_csv_path(repo_root, p, local_n)
        feat_dict = extract_ppg_features(csv_path, warmup_sec=warmup_sec, fs=fs,
                                         include_frequency_features=(ppg_dim >= 11))
        arr = features_to_array(feat_dict, include_frequency_features=(ppg_dim >= 11))
        ppg_map[s["av_key"]] = arr
    return ppg_map


def _build_mlp_features(
    av_rows: list[dict[str, str]],
    ppg_map: dict[str, np.ndarray],
    *,
    ppg_dim: int,
) -> np.ndarray:
    audio = load_audio_embeddings(av_rows)          # (N, 96)
    video = load_video_sequence_embeddings(av_rows)  # (N, 2048)
    ppg = np.stack(
        [ppg_map.get(r["av_key"], np.zeros(ppg_dim, dtype=np.float32)) for r in av_rows],
        axis=0,
    )
    return np.concatenate([audio, video, ppg], axis=1).astype(np.float32)  # (N, 2155)


def _mlp_build_dataloader(
    features: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _mlp_predict(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32)).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)


def _mlp_evaluate(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    class_weights: torch.Tensor,
) -> tuple[np.ndarray, float]:
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for bf, bl in loader:
            bf, bl = bf.to(device), bl.to(device)
            logits = model(bf)
            loss = F.cross_entropy(logits, bl, weight=class_weights, reduction="sum")
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
            total_loss += float(loss.item())
            total_items += bf.shape[0]
    return np.concatenate(all_probs, axis=0), total_loss / max(total_items, 1)


def _mlp_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABEL_ORDER)).astype(np.float32)
    weights = counts.sum() / (len(LABEL_ORDER) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_embedding_mlp_ppg(
    repo_root: Path,
    samples: list[dict[str, Any]],
    audio_csv_dir: Path,
    video_csv_dir: Path,
    *,
    seed: int = 42,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    ppg_dim: int = 11,
    warmup_sec: float = 45.0,
    ppg_fs: int = 128,
) -> dict[str, Any]:
    """Train an EmbeddingMLP on [audio_96 | video_2048 | ppg_11] = 2155-dim features.

    Outputs
    -------
    outputs/predictions/late_fusion_custom/embedding_mlp_ppg/{split}_predictions.jsonl
    outputs/metrics/late_fusion_custom/embedding_mlp_ppg/{split}_metrics.json
    """
    mode = "embedding_mlp_ppg"
    _set_seed(seed)
    mode_start = time.perf_counter()

    # ── Extract PPG features ──────────────────────────────────────────────────
    print(f"[{mode}] Extracting PPG features for {len(samples)} samples...")
    ppg_start = time.perf_counter()
    ppg_map = extract_ppg_features_for_samples(
        samples, repo_root, warmup_sec=warmup_sec, fs=ppg_fs, ppg_dim=ppg_dim
    )
    ppg_time_sec = time.perf_counter() - ppg_start
    print(f"[{mode}] PPG extraction done in {ppg_time_sec:.1f}s.")

    # ── Load AV join rows from the custom AV join directory ───────────────────
    av_join_dir = repo_root / "data/processed/av/late_fusion_custom"
    joined_rows: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "valid", "test"):
        csv_path = av_join_dir / f"av_ecapa_video_reuse_custom_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"AV join CSV not found: {csv_path}\n"
                "Run Step 5 (late fusion weighted_prob_avg) first to build this file."
            )
        joined_rows[split] = _load_csv_rows(csv_path)

    # ── Build feature matrices ────────────────────────────────────────────────
    train_features = _build_mlp_features(joined_rows["train"], ppg_map, ppg_dim=ppg_dim)
    valid_features = _build_mlp_features(joined_rows["valid"], ppg_map, ppg_dim=ppg_dim)
    test_features  = _build_mlp_features(joined_rows["test"],  ppg_map, ppg_dim=ppg_dim)

    train_labels = np.array([int(r["label_id"]) for r in joined_rows["train"]], dtype=np.int64)
    valid_labels = np.array([int(r["label_id"]) for r in joined_rows["valid"]], dtype=np.int64)
    test_labels  = np.array([int(r["label_id"]) for r in joined_rows["test"]],  dtype=np.int64)

    # Normalise using train statistics
    feat_mean = train_features.mean(axis=0, keepdims=True)
    feat_std  = train_features.std(axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0
    train_features = (train_features - feat_mean) / feat_std
    valid_features = (valid_features - feat_mean) / feat_std
    test_features  = (test_features  - feat_mean) / feat_std

    print(f"[{mode}] Feature dim={train_features.shape[1]}  "
          f"train={len(train_labels)} valid={len(valid_labels)} test={len(test_labels)}")

    # ── Build dataloaders ─────────────────────────────────────────────────────
    train_loader = _mlp_build_dataloader(train_features, train_labels,
                                         batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{mode}] device={device}")

    config = EmbeddingMLPConfig(input_dim=train_features.shape[1])
    model  = EmbeddingMLP(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = _mlp_class_weights(train_labels).to(device)

    best_state: dict[str, Any] | None = None
    best_valid_uar   = -1.0
    best_epoch       = -1
    no_improve       = 0
    history: list[dict[str, Any]] = []
    training_start = time.perf_counter()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_items = 0
        for bf, bl in train_loader:
            bf, bl = bf.to(device), bl.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bf)
            loss   = F.cross_entropy(logits, bl, weight=class_weights)
            loss.backward()
            optimizer.step()
            epoch_loss  += float(loss.item()) * bf.shape[0]
            epoch_items += bf.shape[0]

        train_metrics = compute_classification_metrics(
            train_labels, _mlp_predict(model, train_features, device)
        )
        valid_probs, valid_loss = _mlp_evaluate(
            model, valid_features, valid_labels,
            device=device, batch_size=batch_size, class_weights=class_weights,
        )
        valid_metrics = compute_classification_metrics(valid_labels, valid_probs)

        history.append({
            "epoch":          epoch,
            "train_loss":     epoch_loss / max(epoch_items, 1),
            "valid_loss":     valid_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_uar":      train_metrics["uar"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_uar":      valid_metrics["uar"],
            "epoch_time_sec": time.perf_counter() - epoch_start,
        })

        if valid_metrics["uar"] > best_valid_uar:
            best_valid_uar = valid_metrics["uar"]
            best_epoch     = epoch
            no_improve     = 0
            best_state = {
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "feature_mean": feat_mean.tolist(),
                "feature_std":  feat_std.tolist(),
                "config":  asdict(config),
                "epoch":   epoch,
                "best_valid_uar": best_valid_uar,
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state["model_state_dict"])
    total_training_sec = time.perf_counter() - training_start
    avg_epoch_sec = float(sum(e["epoch_time_sec"] for e in history) / len(history)) if history else 0.0

    # Save checkpoint + history
    ckpt_dir = repo_root / "checkpoints/late_fusion_custom/embedding_mlp_ppg"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"
    torch.save(best_state, ckpt_path)
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[{mode}] Training done. best_epoch={best_epoch} best_valid_uar={best_valid_uar:.4f} "
          f"({len(history)} epochs, {total_training_sec:.1f}s)")

    # ── Evaluate and write outputs ────────────────────────────────────────────
    preds_dir   = repo_root / "outputs/predictions/late_fusion_custom/embedding_mlp_ppg"
    metrics_dir = repo_root / "outputs/metrics/late_fusion_custom/embedding_mlp_ppg"
    preds_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    split_outputs: dict[str, Any] = {}
    for split, rows, feats, labels in (
        ("train", joined_rows["train"], train_features, train_labels),
        ("valid", joined_rows["valid"], valid_features, valid_labels),
        ("test",  joined_rows["test"],  test_features,  test_labels),
    ):
        fused_probs, fused_loss = _mlp_evaluate(
            model, feats, labels,
            device=device, batch_size=batch_size, class_weights=class_weights,
        )
        pred_ids = np.argmax(fused_probs, axis=1).tolist()
        metrics  = compute_classification_metrics(labels, fused_probs)

        # Predictions JSONL
        pred_path = preds_dir / f"{split}_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as fh:
            for row, prob_vec, pred_id in zip(rows, fused_probs.tolist(), pred_ids):
                record = {
                    "mode":            mode,
                    "av_key":          row["av_key"],
                    "split":           row["split"],
                    "actor_id":        int(row["actor_id"]),
                    "label_id":        int(row["label_id"]),
                    "label_name":      row["label_name"],
                    "audio_sample_id": row["audio_sample_id"],
                    "video_sample_id": row["video_sample_id"],
                    "pred_id":         int(pred_id),
                    "pred_name":       LABEL_ORDER[int(pred_id)],
                    "probs":           [float(v) for v in prob_vec],
                    "checkpoint_path": ckpt_path.as_posix(),
                    "best_epoch":      best_epoch,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Confusion matrix
        cm_path = metrics_dir / f"{split}_confusion_matrix.csv"
        write_confusion_matrix_csv(cm_path, np.asarray(metrics["confusion_matrix"], dtype=np.int64))

        # Metrics JSON
        metrics_payload: dict[str, Any] = {
            "split":              split,
            "mode":               mode,
            "loss":               fused_loss,
            "accuracy":           metrics["accuracy"],
            "macro_f1":           metrics["macro_f1"],
            "uar":                metrics["uar"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall":   metrics["per_class_recall"],
            "per_class_f1":       metrics["per_class_f1"],
            "best_epoch":         best_epoch,
            "checkpoint_path":    ckpt_path.as_posix(),
            "labels_are_placeholder": labels_are_placeholder(labels),
        }
        (metrics_dir / f"{split}_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        split_outputs[split] = metrics_payload
        print(
            f"[{mode}] {split}: acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} uar={metrics['uar']:.4f}"
            + (" (⚠ placeholder labels)" if metrics_payload["labels_are_placeholder"] else "")
        )

    return {
        "mode":             mode,
        "device":           str(device),
        "parameter_count":  param_count,
        "best_epoch":       best_epoch,
        "best_valid_uar":   best_valid_uar,
        "checkpoint_path":  ckpt_path.as_posix(),
        "timing": {
            "ppg_extraction_sec":     ppg_time_sec,
            "total_training_sec":     total_training_sec,
            "average_epoch_sec":      avg_epoch_sec,
            "mode_runtime_sec":       time.perf_counter() - mode_start,
        },
        "splits": split_outputs,
    }


def _join_audio_video(
    audio_rows: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    *,
    split: str,
) -> list[dict[str, str]]:
    audio_idx = {r["av_key"]: r for r in audio_rows}
    video_idx = {r["av_key"]: r for r in video_rows}

    if set(audio_idx) != set(video_idx):
        missing_v = sorted(set(audio_idx) - set(video_idx))
        missing_a = sorted(set(video_idx) - set(audio_idx))
        raise ValueError(
            f"AV key mismatch for split '{split}'.  "
            f"Only-in-audio={missing_v[:5]}  only-in-video={missing_a[:5]}"
        )

    joined: list[dict[str, str]] = []
    for av_key in sorted(audio_idx):
        ar = audio_idx[av_key]
        vr = video_idx[av_key]
        row: dict[str, str] = {
            "av_key":                    av_key,
            "split":                     split,
            "actor_id":                  ar["actor_id"],
            "label_id":                  ar["label_id"],
            "label_name":                ar["label_name"],
            "emotion_code":              ar["emotion_code"],
            "intensity_code":            ar["intensity_code"],
            "statement_code":            ar["statement_code"],
            "repetition_code":           ar["repetition_code"],
            "audio_sample_id":           ar["sample_id"],
            "audio_wav_path":            ar["wav_path"],
            "audio_pred_id":             ar["pred_id"],
            "audio_pred_name":           ar["pred_name"],
            "audio_embedding_path":      ar["embedding_path"],
            "audio_embedding_dim":       ar["embedding_dim"],
            "video_sample_id":           vr["sample_id"],
            "video_path":                vr["video_path"],
            "video_pred_id":             vr["pred_id"],
            "video_pred_name":           vr["pred_name"],
            "video_sequence_path":       vr["sequence_path"],
            "video_sequence_num_frames": vr["sequence_num_frames"],
            "video_sequence_feature_dim":vr["sequence_feature_dim"],
            "video_sequence_source":     vr["sequence_video_id_source"],
            "video_prediction_source":   vr["prediction_video_id_source"],
        }
        for ln in LABEL_ORDER:
            row[f"audio_prob_{ln}"] = ar[f"prob_{ln}"]
            row[f"video_prob_{ln}"] = vr[f"prob_{ln}"]
        joined.append(row)

    return joined


def _load_prob_matrix(rows: list[dict[str, str]], *, prefix: str) -> np.ndarray:
    fields = [f"{prefix}_prob_{ln}" for ln in LABEL_ORDER]
    return np.array([[float(r[f]) for f in fields] for r in rows], dtype=np.float32)


def _write_fusion_predictions(
    path: Path,
    rows: list[dict[str, str]],
    probs: np.ndarray,
    pred_ids: list[int],
    *,
    audio_weight: float,
    video_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row, prob_vec, pred_id in zip(rows, probs.tolist(), pred_ids):
            record = {
                "mode":              "weighted_prob_avg",
                "av_key":            row["av_key"],
                "split":             row["split"],
                "actor_id":          int(row["actor_id"]),
                "label_id":          int(row["label_id"]),
                "label_name":        row["label_name"],
                "audio_sample_id":   row["audio_sample_id"],
                "video_sample_id":   row["video_sample_id"],
                "pred_id":           int(pred_id),
                "pred_name":         LABEL_ORDER[int(pred_id)],
                "probs":             [float(v) for v in prob_vec],
                "audio_weight":      audio_weight,
                "video_weight":      video_weight,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Disk loaders — used by --skip-inference / --skip-audio / --skip-video
# ═══════════════════════════════════════════════════════════════════════════════

def load_ecapa_artifacts_from_disk(repo_root: Path) -> dict[str, Any]:
    """Reload ECAPA predictions + embedding index written by a previous run.

    Returns the same structure as run_ecapa_inference so it is a drop-in
    replacement when --skip-inference or --skip-audio is set.
    """
    pred_root = repo_root / "data/processed/audio/predictions_ecapa_custom"
    emb_root  = repo_root / "data/processed/audio/embeddings_ecapa_custom"

    all_predictions: dict[str, list[dict]] = {}
    all_embeddings:  dict[str, dict[str, dict]] = {}

    for split in ("train", "valid", "test"):
        pred_path = pred_root / f"{split}_predictions.jsonl"
        if not pred_path.is_file():
            raise FileNotFoundError(
                f"Audio predictions not found: {pred_path}\n"
                "Run without --skip-inference / --skip-audio first to generate them."
            )
        all_predictions[split] = _load_jsonl(pred_path)

        idx_path = emb_root / f"{split}_embedding_index.jsonl"
        if not idx_path.is_file():
            raise FileNotFoundError(
                f"Embedding index not found: {idx_path}\n"
                "Run without --skip-inference / --skip-audio first to generate it."
            )
        all_embeddings[split] = {r["sample_id"]: r for r in _load_jsonl(idx_path)}
        print(f"[ECAPA disk] {split}: {len(all_predictions[split])} predictions, "
              f"{len(all_embeddings[split])} embeddings.")

    return {"predictions": all_predictions, "embeddings": all_embeddings}


def load_video_artifacts_from_disk(repo_root: Path) -> dict[str, Any]:
    """Reload video predictions + sequence paths written by a previous run.

    Returns the same structure as run_video_inference so it is a drop-in
    replacement when --skip-inference or --skip-video is set.
    """
    seq_root      = repo_root / "data/processed/video/sequences_custom"
    vid_pred_root = repo_root / "data/processed/video/predictions_custom"

    all_sequences:   dict[str, dict[str, dict]] = {"train": {}, "valid": {}, "test": {}}
    all_predictions: dict[str, dict[str, dict]] = {"train": {}, "valid": {}, "test": {}}

    # Load predictions
    for split in ("train", "valid", "test"):
        pred_path = vid_pred_root / f"{split}_predictions.jsonl"
        if not pred_path.is_file():
            raise FileNotFoundError(
                f"Video predictions not found: {pred_path}\n"
                "Run without --skip-inference / --skip-video first to generate them."
            )
        for r in _load_jsonl(pred_path):
            vs = r["video_stem"]
            all_predictions[split][vs] = {
                "pred_id":   r["pred_id"],
                "pred_name": r["pred_name"],
                "probs":     r["probs"],
            }

    # Reconstruct sequence info from .npy files already on disk
    if not seq_root.is_dir():
        raise FileNotFoundError(
            f"Video sequence directory not found: {seq_root}\n"
            "Run without --skip-inference / --skip-video first to generate sequences."
        )
    for npy_path in sorted(seq_root.glob("video_*.npy")):
        m = VIDEO_STEM_RE.match(npy_path.stem)
        if not m:
            continue
        p = int(m.group(1))
        split = participant_to_split(p)
        vstem = npy_path.stem
        all_sequences[split][vstem] = {
            "sequence_path":        npy_path.resolve().as_posix(),
            "sequence_num_frames":  FRAMES_PER_VIDEO,
            "sequence_feature_dim": EXPECTED_SEQUENCE_SHAPE[1],
        }

    for split in ("train", "valid", "test"):
        print(f"[Video disk] {split}: {len(all_predictions[split])} predictions, "
              f"{len(all_sequences[split])} sequences.")

    return {"sequences": all_sequences, "predictions": all_predictions}


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ECAPA + video backbone inference and late fusion on 200 custom AV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        metavar="PATH",
        help=(
            "CSV with ground-truth labels.  "
            "Required columns: participant_id, sample_no, label_id "
            "(sample_no is the global 1-based index 1-200; other columns are ignored).  "
            "When omitted every sample is assigned label_id=0 (neutral placeholder)."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Skip both ECAPA and video model inference.  "
            "Loads predictions/embeddings/sequences from the previous run on disk, "
            "then rebuilds the fusion-ready CSVs with whatever labels are in "
            "--labels-csv (or placeholder 0 if omitted).  "
            "Use this to re-apply correct labels without re-running the models."
        ),
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help=(
            "Skip ECAPA model inference only.  "
            "Loads audio predictions and embeddings from disk, "
            "then rebuilds the audio fusion-ready CSV with updated labels."
        ),
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help=(
            "Skip video backbone inference only.  "
            "Loads video predictions and sequences from disk, "
            "then rebuilds the video fusion-ready CSV with updated labels."
        ),
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip late fusion step (weighted_prob_avg).",
    )
    parser.add_argument(
        "--skip-mlp-ppg",
        action="store_true",
        help="Skip Step 6: EmbeddingMLP with PPG features.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0   = time.perf_counter()

    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    if labels_csv is not None and not labels_csv.is_file():
        print(f"[ERROR] --labels-csv not found: {labels_csv}", file=sys.stderr)
        return 1

    samples = collect_custom_samples(REPO_ROOT, labels_csv=labels_csv)
    if not samples:
        print("[ERROR] No AV pairs found.  Check data/raw/custom/audio/ and data/raw/custom/video/.",
              file=sys.stderr)
        return 1

    audio_csv_dir = REPO_ROOT / "data/processed/audio/fusion_ready"
    video_csv_dir = REPO_ROOT / "data/processed/video/fusion_ready"

    skip_audio = args.skip_audio or args.skip_inference
    skip_video = args.skip_video or args.skip_inference

    timing: dict[str, float] = {}

    # ── Step 1: ECAPA-TDNN inference (or reload from disk) ────────────────────
    t1 = time.perf_counter()
    if not skip_audio:
        print("\n── Step 1: ECAPA-TDNN audio inference ──")
        ecapa_results = run_ecapa_inference(samples, REPO_ROOT)
    else:
        print("\n── Step 1: Loading ECAPA artifacts from disk ──")
        ecapa_results = load_ecapa_artifacts_from_disk(REPO_ROOT)
    timing["step1_audio_sec"] = time.perf_counter() - t1

    # ── Step 2: Build audio fusion-ready CSV (always, so labels are current) ──
    print("\n── Step 2: Build audio fusion-ready CSVs ──")
    build_audio_fusion_csvs(samples, ecapa_results, audio_csv_dir)

    # ── Step 3: Video backbone inference (or reload from disk) ────────────────
    t3 = time.perf_counter()
    if not skip_video:
        print("\n── Step 3: Video backbone inference ──")
        video_results = run_video_inference(samples, REPO_ROOT)
    else:
        print("\n── Step 3: Loading video artifacts from disk ──")
        video_results = load_video_artifacts_from_disk(REPO_ROOT)
    timing["step3_video_sec"] = time.perf_counter() - t3

    # ── Step 4: Build video fusion-ready CSV (always, so labels are current) ──
    print("\n── Step 4: Build video fusion-ready CSVs ──")
    build_video_fusion_csvs(samples, video_results, video_csv_dir)

    # ── Step 5: Late fusion (weighted_prob_avg) ───────────────────────────────
    fusion_summary: dict[str, Any] | None = None
    if not args.skip_fusion:
        print("\n── Step 5: Late fusion (weighted_prob_avg) ──")
        t5 = time.perf_counter()
        fusion_summary = run_late_fusion(REPO_ROOT, audio_csv_dir, video_csv_dir)
        timing["step5_weighted_prob_avg_sec"] = time.perf_counter() - t5
    else:
        print("[SKIP] Late fusion (--skip-fusion).")

    # ── Step 6: Late fusion — EmbeddingMLP with PPG ───────────────────────────
    mlp_ppg_summary: dict[str, Any] | None = None
    if not args.skip_mlp_ppg:
        print("\n── Step 6: Late fusion — EmbeddingMLP + PPG ──")
        t6 = time.perf_counter()
        mlp_ppg_summary = run_embedding_mlp_ppg(
            REPO_ROOT,
            samples,
            audio_csv_dir,
            video_csv_dir,
        )
        timing["step6_embedding_mlp_ppg_sec"] = time.perf_counter() - t6
        timing["step6_ppg_extraction_sec"]    = mlp_ppg_summary["timing"]["ppg_extraction_sec"]
        timing["step6_mlp_training_sec"]      = mlp_ppg_summary["timing"]["total_training_sec"]
    else:
        print("[SKIP] EmbeddingMLP+PPG (--skip-mlp-ppg).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──")
    if fusion_summary is not None:
        print("  weighted_prob_avg:")
        for split in ("train", "valid", "test"):
            m = fusion_summary[split]
            ph = " ⚠ placeholder labels" if m["labels_are_placeholder"] else ""
            print(f"    {split:5s}  acc={m['accuracy']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}  uar={m['uar']:.4f}{ph}")

    if mlp_ppg_summary is not None:
        print("  embedding_mlp_ppg:")
        for split in ("train", "valid", "test"):
            m = mlp_ppg_summary["splits"][split]
            ph = " ⚠ placeholder labels" if m["labels_are_placeholder"] else ""
            print(f"    {split:5s}  acc={m['accuracy']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}  uar={m['uar']:.4f}{ph}")

    # ── Timing summary ────────────────────────────────────────────────────────
    total_sec = time.perf_counter() - t0
    timing["total_sec"] = total_sec
    print("\n── Timing ──")
    label_map = {
        "step1_audio_sec":              "  Audio (ECAPA)         ",
        "step3_video_sec":              "  Video (backbone)      ",
        "step5_weighted_prob_avg_sec":  "  Fusion (weighted avg) ",
        "step6_ppg_extraction_sec":     "  PPG extraction        ",
        "step6_mlp_training_sec":       "  MLP training (PPG)    ",
        "step6_embedding_mlp_ppg_sec":  "  Step 6 total          ",
        "total_sec":                    "  TOTAL                 ",
    }
    for key, label in label_map.items():
        if key in timing:
            print(f"{label}: {timing[key]:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
