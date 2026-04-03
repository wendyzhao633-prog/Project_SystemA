#!/usr/bin/env python3
"""
RAVDESS Speech Emotion Recognition — Training Script.

8-class clip-level SER using ECAPA-TDNN + Fbank features.
Adapted from recipes/IEMOCAP/emotion_recognition/train.py.

Key differences from IEMOCAP recipe:
  • Fixed label_id mapping (no CategoricalEncoder) — label ordering never changes.
  • 8 output classes instead of 4.
  • Best checkpoint selected by *maximum validation UAR* (Unweighted Average Recall),
    not minimum classification error.
  • Accumulates per-sample predictions for UAR computation at epoch end.

Usage
-----
  cd C:/Users/NannanLi/Desktop/speechbrain-develop
  conda activate ser_env
  python recipes/RAVDESS_emotion_recognition/train.py ^
      recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml

Authors
-------
  Adapted for RAVDESS 8-class SER by project team, 2024.
  Original IEMOCAP recipe by Pierre-Yves Yanni 2021.
"""

import os
import sys

# Ensure the local repo's speechbrain is found before any installed version.
# This is needed when running the script as `python recipes/…/train.py`
# because Python sets sys.path[0] to the script's own directory, not the
# repo root, so the local speechbrain/ package would be missed otherwise.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb


# ---------------------------------------------------------------------------
# Label constants  (single source of truth — must match label_map.json)
# ---------------------------------------------------------------------------
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
NUM_CLASSES = 8


# ---------------------------------------------------------------------------
# UAR helper
# ---------------------------------------------------------------------------

def compute_uar(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> float:
    """Compute Unweighted Average Recall (UAR = macro recall).

    Parameters
    ----------
    preds : 1-D LongTensor of predicted class indices.
    labels : 1-D LongTensor of ground-truth class indices.
    num_classes : total number of classes.

    Returns
    -------
    float in [0, 1].
    """
    recalls = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            recalls.append((preds[mask] == c).float().mean().item())
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# Brain class
# ---------------------------------------------------------------------------

class RavdessEmoBrain(sb.Brain):
    """SpeechBrain Brain for RAVDESS 8-class SER."""

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def compute_forward(self, batch, stage):
        """Forward pass — supports both ECAPA-TDNN and Wav2Vec2 backbones.

        Backbone is selected by ``hparams.model_type``:
          * ``"ecapa"``   (default) : Fbank → mean-var norm → ECAPA-TDNN
          * ``"wav2vec2"``          : raw waveform → Wav2Vec2Encoder (mean-pool)

        Returns
        -------
        outputs : torch.Tensor, shape [batch, 1, out_n_neurons]
            Raw cosine-similarity logits from Classifier.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        model_type = getattr(self.hparams, "model_type", "ecapa")

        if model_type in ("wav2vec2", "wavlm"):
            # Raw waveform → HF SSL encoder → [batch, 1, hidden_size]
            embeddings = self.modules.embedding_model(wavs, lens)
        else:
            # Fbank features → mean-var norm → ECAPA-TDNN → [batch, 1, 96]
            feats = self.modules.compute_features(wavs)
            feats = self.modules.mean_var_norm(feats, lens)
            embeddings = self.modules.embedding_model(feats, lens)

        # classifier output: [batch, 1, out_n_neurons]
        outputs = self.modules.classifier(embeddings)
        return outputs

    # ------------------------------------------------------------------ #
    # Loss + metric accumulation
    # ------------------------------------------------------------------ #
    def compute_objectives(self, predictions, batch, stage):
        """Cross-entropy-style loss + accumulate predictions for UAR.

        Three mutually exclusive loss paths (checked in order):
          1. use_focal_loss = True  → Focal Loss  FL = -(1-pt)^γ · log(pt)
          2. use_class_weights = True → weighted F.cross_entropy
          3. default → LogSoftmaxWrapper + AdditiveAngularMargin (AAM-Softmax)

        All existing versions (V1–V7) are unaffected: they do not define
        use_focal_loss in their yaml, so they fall through to their existing path.
        """
        _, lens = batch.sig
        # label_id_encoded shape: [batch, 1]
        label_id, _ = batch.label_id_encoded

        if getattr(self.hparams, "use_focal_loss", False):
            # ── Focal Loss ────────────────────────────────────────────────
            # predictions [B,1,C] → logits [B,C];  labels [B,1] → [B]
            logits = predictions.squeeze(1)
            targets = label_id.squeeze(1)
            gamma = float(getattr(self.hparams, "focal_gamma", 2.0))
            # per-sample CE gives -log(pt); exp gives pt
            ce_per_sample = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_per_sample)           # probability of correct class
            loss = ((1.0 - pt) ** gamma * ce_per_sample).mean()

        else:
            class_weights = getattr(self, "class_weights", None)
            if getattr(self.hparams, "use_class_weights", False) and class_weights is not None:
                # ── Weighted cross-entropy (V7) ───────────────────────────
                loss = F.cross_entropy(
                    predictions.squeeze(1),
                    label_id.squeeze(1),
                    weight=class_weights.to(predictions.device),
                )
            else:
                # ── Default: AAM-Softmax (V1 and all other versions) ──────
                loss = self.hparams.compute_cost(predictions, label_id, lens)

        if stage != sb.Stage.TRAIN:
            # squeeze to [batch]
            pred_ids  = torch.argmax(predictions.squeeze(1), dim=-1)
            true_ids  = label_id.squeeze(1)
            self._stage_preds.append(pred_ids.detach().cpu())
            self._stage_labels.append(true_ids.detach().cpu())

        return loss

    # ------------------------------------------------------------------ #
    # Optimizer — supports optional param groups for partial fine-tuning
    # ------------------------------------------------------------------ #
    def init_optimizers(self):
        """Create optimizer, with optional per-group LRs for partial FT.

        If ``hparams.backbone_lr`` is defined, two param groups are created:
          * group 0 — trainable backbone params  (lr = backbone_lr)
          * group 1 — classifier params          (lr = lr)

        This is used for the wav2vec2 partial fine-tune variant where the
        last N transformer layers are unfrozen while the classifier uses a
        higher learning rate.

        For ECAPA and frozen wav2vec2, ``backbone_lr`` is NOT defined and
        the method falls through to the default SpeechBrain behaviour.
        """
        backbone_lr = getattr(self.hparams, "backbone_lr", None)

        if backbone_lr is None:
            # Default path — ECAPA and frozen wav2vec2 use this
            super().init_optimizers()
            return

        # ── Partial fine-tune path ──────────────────────────────────────
        classifier_lr = getattr(self.hparams, "lr", 1e-4)

        # Collect trainable backbone parameters (requires_grad=True only)
        trainable_backbone = [
            p for p in self.modules.embedding_model.parameters()
            if p.requires_grad
        ]
        classifier_params = list(self.modules.classifier.parameters())

        param_groups = []
        if trainable_backbone:
            param_groups.append({
                "params": trainable_backbone,
                "lr": backbone_lr,
                "name": "backbone_unfrozen",
            })
        if classifier_params:
            param_groups.append({
                "params": classifier_params,
                "lr": classifier_lr,
                "name": "classifier",
            })

        if not param_groups:
            # Nothing trainable — fall back to default
            super().init_optimizers()
            return

        self.optimizer = self.hparams.opt_class(param_groups)

        # Register optimizer with checkpointer (same as super() does)
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        # ── Log param groups ────────────────────────────────────────────
        total_trainable = sum(
            p.numel() for g in param_groups for p in g["params"]
        )
        total_params = sum(p.numel() for p in self.modules.parameters())
        print(f"[INFO] Optimizer param groups:")
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"  [{g['name']}]  lr={g['lr']:.1e}  params={n_params:,}")
        print(f"[INFO] Trainable params : {total_trainable:,}")
        print(f"[INFO] Total params     : {total_params:,}")

    # ------------------------------------------------------------------ #
    # Stage lifecycle
    # ------------------------------------------------------------------ #
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Step CyclicLRScheduler after each gradient update (if present)."""
        if should_step and hasattr(self.hparams, "lr_annealing"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

    def on_stage_start(self, stage, epoch=None):
        """Initialise per-epoch accumulators."""
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        if stage != sb.Stage.TRAIN:
            self._stage_preds  = []
            self._stage_labels = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Compute metrics, log, and (for VALID) save best checkpoint."""

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        # Compute accuracy and UAR from accumulated predictions
        all_preds  = torch.cat(self._stage_preds)   # [N]
        all_labels = torch.cat(self._stage_labels)  # [N]

        acc = (all_preds == all_labels).float().mean().item()
        uar = compute_uar(all_preds, all_labels, self.hparams.out_n_neurons)

        stats = {"loss": stage_loss, "acc": acc, "uar": uar}

        if stage == sb.Stage.VALID:
            # CyclicLRScheduler updates LR per-batch in on_fit_batch_end;
            # read the current value from the optimizer for logging only.
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": current_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save checkpoint; keep only the one with the highest UAR.
            is_new_best = stats["uar"] > getattr(self, "_best_val_uar", 0.0)
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=["uar"]
            )
            if is_new_best:
                self._best_val_uar = stats["uar"]
                print(
                    f"  [BEST] New best checkpoint  "
                    f"epoch={epoch}  val_uar={stats['uar']:.4f}  val_acc={stats['acc']:.4f}"
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            # Print confusion matrix to stdout for quick inspection
            _print_confusion_matrix(all_preds, all_labels)


# ---------------------------------------------------------------------------
# Confusion matrix helper (stdout only; full export is in evaluate.py)
# ---------------------------------------------------------------------------

def _print_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor):
    """Print a simple ASCII confusion matrix to stdout."""
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for p, t in zip(preds.tolist(), labels.tolist()):
        cm[t][p] += 1

    header = "True\\Pred  " + "  ".join(f"{i:3d}" for i in range(NUM_CLASSES))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(header)
    for i in range(NUM_CLASSES):
        row = f"  {ID_TO_LABEL_NAME[i]:9s}  " + "  ".join(f"{cm[i][j]:3d}" for j in range(NUM_CLASSES))
        print(row)
    print()


# ---------------------------------------------------------------------------
# Class-weight helper (used when hparams.use_class_weights = True)
# ---------------------------------------------------------------------------

def compute_class_weights(train_json_path: str, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compute balanced class weights from the training set.

    Formula: w_c = N / (C * n_c)
      N  = total training samples
      C  = number of classes
      n_c = samples in class c

    Returns
    -------
    torch.Tensor of shape (num_classes,), dtype float32.
    """
    import json
    with open(train_json_path, encoding="utf-8") as f:
        data = json.load(f)

    counts = [0] * num_classes
    for sample in data.values():
        counts[sample["label_id"]] += 1

    N = sum(counts)
    weights = [N / (num_classes * c) if c > 0 else 0.0 for c in counts]

    print(f"[INFO] Class counts (train) : {counts}")
    print(f"[INFO] Class weights        : {[round(w, 4) for w in weights]}")

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def dataio_prep(hparams):
    """Build DynamicItemDatasets from the JSON manifests.

    The label_id field in the manifest is a pre-computed integer
    (0-indexed, fixed order).  We do NOT use CategoricalEncoder so that
    the label ordering is always identical to metadata/label_map.json.
    """

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        # read_audio returns (T,) for mono or (T, C) for multi-channel.
        # A handful of RAVDESS files are stereo; downmix to mono so that
        # all tensors in a batch have the same number of dimensions.
        if sig.ndim == 2:
            sig = sig.mean(dim=1)
        return sig

    @sb.utils.data_pipeline.takes("label_id")
    @sb.utils.data_pipeline.provides("label_id", "label_id_encoded")
    def label_pipeline(label_id):
        # label_id from JSON is an int; wrap in a 1-element LongTensor
        yield label_id
        yield torch.LongTensor([label_id])

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test":  hparams["test_annotation"],
    }
    for split, json_path in data_info.items():
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Manifest not found: {json_path}\n"
                "Please run prepare_ravdess.py first."
            )
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "label_id_encoded"],
        )

    return datasets


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Report device and model type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[INFO] Device     : GPU — {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Device     : CPU (no CUDA GPU detected — training will be slower)")
    model_type_str = hparams.get("model_type", "ecapa").upper()
    print(f"[INFO] Model type : {model_type_str}")
    print(f"[INFO] Epochs     : {hparams['number_of_epochs']}  (override: --number_of_epochs N)")

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Build datasets (manifests must already exist)
    datasets = dataio_prep(hparams)

    emo_brain = RavdessEmoBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # ── Focal Loss logging (only when use_focal_loss: True in yaml) ───────
    if hparams.get("use_focal_loss", False):
        gamma = hparams.get("focal_gamma", 2.0)
        print(f"[INFO] use_focal_loss : True")
        print(f"[INFO] focal_gamma    : {gamma}")

    # ── Class weights (only when use_class_weights: True in yaml) ─────────
    if hparams.get("use_class_weights", False):
        print(f"[INFO] use_class_weights : True  (computing from train set)")
        emo_brain.class_weights = compute_class_weights(hparams["train_annotation"])
    else:
        emo_brain.class_weights = None

    # ── Training ──────────────────────────────────────────────────────────
    emo_brain.fit(
        epoch_counter=emo_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # ── Test evaluation (loads best checkpoint by max UAR) ────────────────
    emo_brain.evaluate(
        test_set=datasets["test"],
        max_key="uar",
        test_loader_kwargs=hparams["dataloader_options"],
    )
