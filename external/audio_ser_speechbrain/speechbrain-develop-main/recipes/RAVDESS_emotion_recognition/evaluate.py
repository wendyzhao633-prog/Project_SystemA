#!/usr/bin/env python3
"""
RAVDESS SER — Evaluation & Prediction Export Script.

Loads the best checkpoint (selected by highest validation UAR during training),
runs inference on a specified split (val or test), and exports:

  A. Per-sample predictions as JSONL:
       outputs/predictions/{split}_predictions.jsonl
     Each line is a JSON object containing:
       sample_id, split, actor, emotion_code, label_id, label_name,
       pred_id, pred_name, probs (list of 8 floats)

  B. Aggregate metrics printed to stdout:
       Accuracy, UAR, per-class recall, confusion matrix

Usage
-----
  cd C:/Users/NannanLi/Desktop/speechbrain-develop
  conda activate ser_env

  # Evaluate on test set (uses best checkpoint automatically):
  python recipes/RAVDESS_emotion_recognition/evaluate.py ^
      --hparams recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml ^
      --split test

  # Evaluate on val set:
  python recipes/RAVDESS_emotion_recognition/evaluate.py ^
      --hparams recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml ^
      --split valid

Notes
-----
  • The script loads the same hparams yaml used for training.
  • The best checkpoint is located automatically via the Checkpointer
    (max_key="uar" matches what was used in train.py).
  • To point at a different save directory, override the yaml key:
      --overrides "save_folder=/path/to/other/save"
"""

import argparse
import json
import os
import sys

# Ensure the local repo's speechbrain is found before any installed version.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb

# ── Label constants ────────────────────────────────────────────────────────
# Must match label_map.json and train.py  (never reorder)
ID_TO_LABEL_NAME = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprise",
}
LABEL_NAME_TO_ID = {v: k for k, v in ID_TO_LABEL_NAME.items()}
NUM_CLASSES = 8


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_preds, all_labels, num_classes=NUM_CLASSES):
    """Compute accuracy, UAR, per-class recall, and confusion matrix.

    Parameters
    ----------
    all_preds : list[int]
    all_labels : list[int]

    Returns
    -------
    dict with keys: accuracy, uar, per_class_recall (dict), confusion_matrix (2-D list)
    """
    preds  = torch.tensor(all_preds,  dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)

    acc = (preds == labels).float().mean().item()

    per_class_recall = {}
    recalls = []
    for c in range(num_classes):
        mask = labels == c
        n = mask.sum().item()
        if n > 0:
            r = (preds[mask] == c).float().mean().item()
            per_class_recall[ID_TO_LABEL_NAME[c]] = round(r, 4)
            recalls.append(r)
        else:
            per_class_recall[ID_TO_LABEL_NAME[c]] = None

    uar = sum(r for r in recalls) / len(recalls) if recalls else 0.0

    # Build confusion matrix (rows=true, cols=pred)
    cm = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(all_preds, all_labels):
        cm[t][p] += 1

    return {
        "accuracy": round(acc, 4),
        "uar":      round(uar, 4),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
    }


def print_metrics(metrics):
    """Pretty-print metrics to stdout."""
    print(f"\n{'=' * 55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  UAR      : {metrics['uar']:.4f}  ({metrics['uar']*100:.2f}%)")
    print()
    print("  Per-class recall:")
    for name, r in metrics["per_class_recall"].items():
        lid = LABEL_NAME_TO_ID[name]
        r_str = f"{r:.4f}" if r is not None else "  N/A (no samples)"
        print(f"    id={lid}  {name:10s}: {r_str}")

    print()
    print("  Confusion Matrix (rows=true, cols=pred):")
    cm = metrics["confusion_matrix"]
    header = "  True\\Pred " + "".join(f" {i:4d}" for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = f"  {ID_TO_LABEL_NAME[i]:9s} " + "".join(f" {cm[i][j]:4d}" for j in range(NUM_CLASSES))
        print(row)
    print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model_and_checkpointer(hparams, run_opts):
    """Instantiate ECAPA-TDNN modules and Checkpointer from hparams."""

    class _EvalBrain(sb.Brain):
        """Minimal Brain used only for checkpoint restoration."""
        def compute_forward(self, batch, stage):
            return None
        def compute_objectives(self, predictions, batch, stage):
            return torch.tensor(0.0)

    brain = _EvalBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    return brain


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(modules, dataset, hparams, device, split_name, manifest_path):
    """Iterate over a dataset and collect per-sample predictions.

    Returns
    -------
    list[dict] – one dict per sample with all required output fields.
    """
    # Load raw manifest to retrieve metadata fields
    with open(manifest_path, encoding="utf-8") as f:
        raw_manifest = json.load(f)

    # Build DataLoader (no shuffle during evaluation)
    eval_loader_kwargs = {**hparams["dataloader_options"], "shuffle": False}
    loader = sb.dataio.dataloader.make_dataloader(dataset, **eval_loader_kwargs)

    model_type = hparams.get("model_type", "ecapa")

    modules.eval()

    results = []
    for batch in tqdm(loader, desc=f"Inference [{split_name}]", dynamic_ncols=True):
        batch = batch.to(device)
        wavs, lens = batch.sig

        if model_type in ("wav2vec2", "wavlm"):
            # Raw waveform → HF SSL encoder → [B, 1, hidden_size]
            embeddings = modules.embedding_model(wavs, lens)
        else:
            # Fbank → mean-var norm → ECAPA-TDNN → [B, 1, 96]
            feats = modules.compute_features(wavs)
            feats = modules.mean_var_norm(feats, lens)
            embeddings = modules.embedding_model(feats, lens)

        outputs = modules.classifier(embeddings)            # [B, 1, 8]

        # Softmax probabilities over class dimension
        probs_tensor = F.softmax(outputs.squeeze(1), dim=-1)  # [B, 8]
        pred_ids     = torch.argmax(probs_tensor, dim=-1)     # [B]

        sample_ids = batch.id  # list of str

        for i, sid in enumerate(sample_ids):
            meta = raw_manifest.get(sid, {})
            pred_id   = pred_ids[i].item()
            probs_list = probs_tensor[i].cpu().tolist()

            results.append({
                "sample_id":    sid,
                "split":        split_name,
                "actor":        meta.get("actor"),
                "wav":          meta.get("wav"),
                "emotion_code": meta.get("emotion_code"),
                "label_id":     meta.get("label_id"),
                "label_name":   meta.get("label_name"),
                "pred_id":      pred_id,
                "pred_name":    ID_TO_LABEL_NAME[pred_id],
                "probs":        [round(p, 6) for p in probs_list],
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAVDESS SER model and export per-sample predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hparams",
        default="recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml",
        help="Path to hparams YAML (same file used for training)",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="test",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Where to save the JSONL predictions "
            "(defaults to hparams['predictions_dir'])"
        ),
    )
    parser.add_argument(
        "--overrides",
        default="",
        help="Comma-separated key=value overrides for hparams",
    )
    args = parser.parse_args()

    # ── Parse overrides (SpeechBrain style) ──────────────────────────────
    overrides = args.overrides.strip()

    # ── Load hparams ──────────────────────────────────────────────────────
    if not os.path.exists(args.hparams):
        print(f"ERROR: hparams file not found: {args.hparams}", file=sys.stderr)
        sys.exit(1)

    with open(args.hparams, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides if overrides else None)

    # ── Device ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device : {device}")
    if device == "cuda":
        print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Model type   : {hparams.get('model_type', 'ecapa').upper()}")

    run_opts = {"device": device}

    # ── Restore best checkpoint (max UAR) ─────────────────────────────────
    brain = build_model_and_checkpointer(hparams, run_opts)
    brain.on_evaluate_start(max_key="uar")
    print("[INFO] Best checkpoint loaded.")

    modules = brain.modules

    # ── Build dataset ──────────────────────────────────────────────────────
    split_key  = "valid" if args.split == "valid" else "test"
    annot_key  = "valid_annotation" if split_key == "valid" else "test_annotation"
    manifest_path = hparams[annot_key]

    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        print("Run prepare_ravdess.py first.", file=sys.stderr)
        sys.exit(1)

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
        json_path=manifest_path,
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig", "label_id_encoded"],
    )

    # ── Inference ─────────────────────────────────────────────────────────
    results = run_inference(
        modules=modules,
        dataset=dataset,
        hparams=hparams,
        device=device,
        split_name=args.split,
        manifest_path=manifest_path,
    )

    # ── Compute metrics ───────────────────────────────────────────────────
    all_preds  = [r["pred_id"] for r in results]
    all_labels = [r["label_id"] for r in results]
    metrics    = compute_metrics(all_preds, all_labels)
    print_metrics(metrics)

    # ── Save predictions JSONL ────────────────────────────────────────────
    output_dir = args.output_dir or hparams.get("predictions_dir",
                  "C:/Users/NannanLi/Desktop/speechbrain-develop/outputs/predictions")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{args.split}_predictions.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] Predictions saved to: {out_path}")
    print(f"[INFO] Total samples: {len(results)}")

    # ── Save metrics JSON ─────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, f"{args.split}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
