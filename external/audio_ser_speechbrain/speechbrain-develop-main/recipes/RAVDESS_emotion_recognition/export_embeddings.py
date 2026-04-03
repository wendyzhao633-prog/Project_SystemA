#!/usr/bin/env python3
"""
RAVDESS SER — Clip-Level Embedding Export Script.

Loads the best checkpoint (max validation UAR), runs the ECAPA-TDNN encoder
(without the classification head), and saves one embedding .npy file per sample.

Embedding details
-----------------
  Source : output of ECAPA_TDNN.embedding_model before the Classifier.
  Shape  : (96,)   — 96-dim L2-normalised vector (lin_neurons in hparams yaml).
  Files  : outputs/embeddings/{split}/{sample_id}.npy

Also writes an index file:
  outputs/embeddings/{split}_embedding_index.jsonl
  Each line: {"sample_id": ..., "embedding_path": ..., "label_id": ..., "split": ...}

Usage
-----
  cd C:/Users/NannanLi/Desktop/speechbrain-develop
  conda activate ser_env

  # Export test-set embeddings:
  python recipes/RAVDESS_emotion_recognition/export_embeddings.py ^
      --hparams recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml ^
      --split test

  # Export all splits:
  python recipes/RAVDESS_emotion_recognition/export_embeddings.py ^
      --hparams recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml ^
      --split all
"""

import argparse
import json
import os
import sys

# Ensure the local repo's speechbrain is found before any installed version.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb


# ---------------------------------------------------------------------------
# Build minimal Brain for checkpoint restoration
# ---------------------------------------------------------------------------

def build_brain_for_restore(hparams, run_opts):
    class _RestoreBrain(sb.Brain):
        def compute_forward(self, batch, stage):
            return None
        def compute_objectives(self, predictions, batch, stage):
            return torch.tensor(0.0)

    return _RestoreBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_embeddings_for_split(
    modules, hparams, device, split_name, manifest_path, embeddings_root
):
    """Extract and save embeddings for one split.

    Returns path to the index JSONL file.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run prepare_ravdess.py first."
        )

    # Output directory for this split
    split_emb_dir = os.path.join(embeddings_root, split_name)
    os.makedirs(split_emb_dir, exist_ok=True)

    # Build dataset
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

    model_type = hparams.get("model_type", "ecapa")

    eval_loader_kwargs = {**hparams["dataloader_options"], "shuffle": False}
    loader = sb.dataio.dataloader.make_dataloader(dataset, **eval_loader_kwargs)

    modules.eval()

    index_records = []
    for batch in tqdm(loader, desc=f"Exporting embeddings [{split_name}]", dynamic_ncols=True):
        batch = batch.to(device)
        wavs, lens = batch.sig

        if model_type in ("wav2vec2", "wavlm"):
            # Raw waveform → HF SSL encoder → [batch, 1, hidden_size]
            embeddings = modules.embedding_model(wavs, lens)
        else:
            # Fbank → mean-var norm → ECAPA-TDNN → [batch, 1, 96]
            feats = modules.compute_features(wavs)
            feats = modules.mean_var_norm(feats, lens)
            embeddings = modules.embedding_model(feats, lens)

        # Squeeze to [batch, embedding_dim]
        embeddings_2d = embeddings.squeeze(1).cpu()

        label_ids = batch.label_id_encoded.data.squeeze(1).cpu().tolist()
        sample_ids = batch.id

        for i, sid in enumerate(sample_ids):
            emb_np = embeddings_2d[i].numpy()         # shape (96,)
            emb_path = os.path.join(split_emb_dir, f"{sid}.npy")
            np.save(emb_path, emb_np)

            index_records.append({
                "sample_id":       sid,
                "embedding_path":  emb_path.replace("\\", "/"),
                "embedding_dim":   int(emb_np.shape[0]),
                "label_id":        int(label_ids[i]),
                "split":           split_name,
            })

    # Write index JSONL
    index_path = os.path.join(embeddings_root, f"{split_name}_embedding_index.jsonl")
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] {split_name}: {len(index_records)} embeddings saved to {split_emb_dir}")
    print(f"[INFO] Index file: {index_path}")
    return index_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export ECAPA-TDNN clip-level embeddings for RAVDESS SER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hparams",
        default="recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml",
        help="Path to hparams YAML (same file used for training)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "all"],
        default="test",
        help="Which split(s) to export embeddings for",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Root directory for .npy files "
            "(defaults to hparams['embeddings_dir'])"
        ),
    )
    parser.add_argument(
        "--overrides",
        default="",
        help="Comma-separated key=value overrides for hparams",
    )
    args = parser.parse_args()

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

    # ── Restore best checkpoint ───────────────────────────────────────────
    brain = build_brain_for_restore(hparams, run_opts)
    brain.on_evaluate_start(max_key="uar")
    print("[INFO] Best checkpoint loaded.")

    modules = brain.modules

    # ── Output root ───────────────────────────────────────────────────────
    embeddings_root = args.output_dir or hparams.get(
        "embeddings_dir",
        "C:/Users/NannanLi/Desktop/speechbrain-develop/outputs/embeddings",
    )
    os.makedirs(embeddings_root, exist_ok=True)

    # ── Export ────────────────────────────────────────────────────────────
    split_to_annot = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test":  hparams["test_annotation"],
    }

    splits_to_process = (
        list(split_to_annot.keys()) if args.split == "all" else [args.split]
    )

    for split_name in splits_to_process:
        manifest_path = split_to_annot[split_name]
        export_embeddings_for_split(
            modules=modules,
            hparams=hparams,
            device=device,
            split_name=split_name,
            manifest_path=manifest_path,
            embeddings_root=embeddings_root,
        )

    emb_dim  = hparams.get("embedding_dim", 96)
    mdl_type = hparams.get("model_type", "ecapa").upper()
    print("\n[INFO] Embedding export complete.")
    print(f"[INFO] Embedding shape per sample: ({emb_dim},)  [{mdl_type}]")


if __name__ == "__main__":
    main()
