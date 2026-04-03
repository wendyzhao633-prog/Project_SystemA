#!/usr/bin/env python3
"""
HuggingFace SSL encoder wrapper for SpeechBrain — frozen or partial fine-tune SER.

Provides ``Wav2Vec2Encoder``: a thin wrapper around HuggingFace speech SSL
models (Wav2Vec2, WavLM) that supports three training modes:

  1. Fully frozen backbone  (freeze=True, num_unfrozen_layers=0)
       → only the downstream classifier is trained.
  2. Partial fine-tune      (freeze=True, num_unfrozen_layers=N)
       → backbone is first fully frozen, then the last N transformer layers
         are selectively unfrozen.  CNN feature extractor stays frozen.
  3. Full fine-tune         (freeze=False)
       → all backbone parameters are trainable.

In all cases the module:
  * normalises raw waveforms to zero-mean / unit-variance per utterance
  * applies length-masked mean-pooling over frame-level hidden states
  * returns shape ``[batch, 1, hidden_size]`` — same convention as
    ECAPA-TDNN's embedding_model, compatible with the existing pipeline.

Requirements
------------
    pip install transformers

Supported backbones (HuggingFace hub)
--------------------------------------
    Wav2Vec2 (model_arch="wav2vec2"):
        facebook/wav2vec2-base          hidden_size=768,  ~95 M params, 12 layers
        facebook/wav2vec2-large         hidden_size=1024, ~317 M params, 24 layers
        facebook/wav2vec2-base-960h     ASR fine-tuned, 768-dim
        facebook/wav2vec2-large-960h    ASR fine-tuned, 1024-dim

    WavLM (model_arch="wavlm"):
        microsoft/wavlm-base            hidden_size=768,  ~94 M params, 12 layers
        microsoft/wavlm-large           hidden_size=1024, ~316 M params, 24 layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2Vec2Encoder(nn.Module):
    """HuggingFace SSL encoder (Wav2Vec2 / WavLM) with mean-pooling.

    Supports frozen, partial fine-tune, and full fine-tune modes.

    Parameters
    ----------
    source : str
        HuggingFace model hub ID.
        Wav2Vec2: ``"facebook/wav2vec2-base"`` / ``"facebook/wav2vec2-large"``
        WavLM:    ``"microsoft/wavlm-base"``  / ``"microsoft/wavlm-large"``
    model_arch : str
        Architecture family.  ``"wav2vec2"`` (default) or ``"wavlm"``.
        Controls which HuggingFace model class is loaded.  All other logic
        (freeze / unfreeze / pooling) is identical for both families because
        their internal attribute structure is the same.
    freeze : bool
        If ``True`` (default), ALL backbone parameters are frozen first.
        Combined with ``num_unfrozen_layers > 0``, this gives partial FT.
        If ``False``, the entire backbone is trainable (full fine-tune).
    num_unfrozen_layers : int
        Number of transformer encoder layers to unfreeze from the END of
        the encoder stack.  Only effective when ``freeze=True``.
        * 0  → fully frozen backbone (default, original behaviour)
        * 2  → last 2 transformer layers trainable; rest frozen
        * N  → last N transformer layers trainable; rest frozen
        The CNN feature extractor is ALWAYS kept frozen.
    output_norm : bool
        If ``True``, L2-normalise the pooled embedding before returning.
        Leave ``False`` when using the ECAPA ``Classifier`` head (which
        normalises internally).

    Output shape
    ------------
    ``[batch, 1, hidden_size]``
        * 768  for ``*-base`` models
        * 1024 for ``*-large`` models
    """

    def __init__(
        self,
        source: str = "facebook/wav2vec2-base",
        model_arch: str = "wav2vec2",
        freeze: bool = True,
        num_unfrozen_layers: int = 0,
        output_norm: bool = False,
    ):
        super().__init__()

        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for Wav2Vec2Encoder.\n"
                "Install it with:  pip install transformers"
            ) from exc

        # Select model class based on architecture family
        model_arch = model_arch.lower()
        if model_arch == "wavlm":
            from transformers import WavLMModel
            _ModelClass = WavLMModel
            self._log_prefix = "WavLMEncoder"
        else:
            from transformers import Wav2Vec2Model
            _ModelClass = Wav2Vec2Model
            self._log_prefix = "Wav2Vec2Encoder"

        print(f"[{self._log_prefix}] Loading pretrained backbone: {source}")
        self.model = _ModelClass.from_pretrained(source)
        self.output_norm = output_norm
        self.output_dim: int = self.model.config.hidden_size   # 768 or 1024

        # Store for use in train() override
        self._num_unfrozen_layers: int = 0

        if freeze:
            self._freeze_backbone()
            if num_unfrozen_layers > 0:
                self._unfreeze_last_n_layers(num_unfrozen_layers)
        else:
            print(
                f"[{self._log_prefix}] Backbone FULLY TRAINABLE. "
                f"output_dim={self.output_dim}"
            )

        # Always print a clear trainability summary
        self._log_trainability()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _unfreeze_last_n_layers(self, n: int) -> None:
        """Unfreeze the last ``n`` transformer encoder layers.

        The CNN feature extractor (``model.feature_extractor``) is always
        kept frozen, regardless of ``n``.
        """
        layers = self.model.encoder.layers  # ModuleList
        num_layers = len(layers)
        n = min(n, num_layers)
        self._num_unfrozen_layers = n

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze the encoder-level LayerNorm that follows all layers
        # (sits between the last transformer layer and the pooling step).
        for param in self.model.encoder.layer_norm.parameters():
            param.requires_grad = True

        print(
            f"[{self._log_prefix}] Partial fine-tune: "
            f"unfroze last {n}/{num_layers} transformer layers "
            f"+ encoder.layer_norm."
        )

    def _all_backbone_frozen(self) -> bool:
        """True only if every backbone parameter is frozen."""
        return all(not p.requires_grad for p in self.model.parameters())

    # ------------------------------------------------------------------
    # Trainability summary (called once at __init__)
    # ------------------------------------------------------------------

    def _log_trainability(self) -> None:
        """Print a clear summary of which components are trainable."""
        backbone_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        backbone_frozen = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        backbone_total = backbone_trainable + backbone_frozen

        layers = self.model.encoder.layers
        num_layers = len(layers)
        trainable_layer_ids = [
            i for i, layer in enumerate(layers)
            if any(p.requires_grad for p in layer.parameters())
        ]

        if self._all_backbone_frozen():
            mode_str = "FULLY FROZEN"
        elif self._num_unfrozen_layers > 0:
            mode_str = f"PARTIAL FT (last {self._num_unfrozen_layers} / {num_layers} transformer layers)"
        else:
            mode_str = "FULLY TRAINABLE"

        pfx = self._log_prefix
        print(f"[{pfx}] ─────────────────────────────────────────")
        print(f"[{pfx}] Backbone mode       : {mode_str}")
        print(f"[{pfx}] output_dim          : {self.output_dim}")
        print(f"[{pfx}] Backbone params     : {backbone_total:,}")
        print(f"[{pfx}]   → trainable       : {backbone_trainable:,}")
        print(f"[{pfx}]   → frozen          : {backbone_frozen:,}")
        if trainable_layer_ids:
            print(f"[{pfx}] Trainable layers    : {trainable_layer_ids}")
        else:
            print(f"[{pfx}] Trainable layers    : none (backbone frozen)")
        print(f"[{pfx}] ─────────────────────────────────────────")

    # ------------------------------------------------------------------
    # Override train() — keep frozen parts in eval mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Set training mode, keeping frozen submodules in eval mode.

        * Fully frozen backbone       → entire ``self.model`` in eval.
        * Partially frozen backbone   → CNN + frozen transformer layers in
                                        eval; unfrozen transformer layers in
                                        train (activates dropout regularisation
                                        only where gradients flow).
        * Fully trainable backbone    → standard PyTorch behaviour.
        """
        super().train(mode)

        if mode:
            if self._all_backbone_frozen():
                # Case 1: fully frozen — keep everything in eval
                self.model.eval()

            elif self._num_unfrozen_layers > 0:
                # Case 2: partial fine-tune
                # CNN feature extractor (always frozen) → eval
                self.model.feature_extractor.eval()
                self.model.feature_projection.eval()

                # Positional conv embedding (frozen) → eval
                self.model.encoder.pos_conv_embed.eval()

                # Frozen transformer layers → eval
                layers = self.model.encoder.layers
                num_layers = len(layers)
                n = self._num_unfrozen_layers
                for layer in layers[: num_layers - n]:
                    layer.eval()
                # Last n layers: already in train mode from super().train(mode)

            # Case 3 (fully trainable): super().train(mode) already correct

        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Extract a clip-level embedding from a batch of raw waveforms.

        Parameters
        ----------
        wavs : torch.Tensor, shape ``[batch, T]``
            Raw PCM samples in ``[-1, 1]``.
        wav_lens : torch.Tensor, shape ``[batch]``, optional
            Relative lengths in ``(0, 1]`` (SpeechBrain ``PaddedBatch``).
            Used to exclude padded frames from mean-pooling.

        Returns
        -------
        torch.Tensor, shape ``[batch, 1, hidden_size]``
        """
        # ── 1. Per-utterance normalisation ────────────────────────────────
        mean = wavs.mean(dim=-1, keepdim=True)
        std  = wavs.std(dim=-1, keepdim=True).clamp(min=1e-9)
        wavs = (wavs - mean) / std

        # ── 2. Backbone forward ───────────────────────────────────────────
        outputs = self.model(input_values=wavs)
        # hidden_states: [batch, T', hidden_size]
        hidden_states = outputs.last_hidden_state

        # ── 3. Length-masked mean-pooling ─────────────────────────────────
        if wav_lens is not None:
            T_prime = hidden_states.shape[1]
            abs_lens = (
                wav_lens.float().to(hidden_states.device) * T_prime
            ).ceil().long().clamp(min=1, max=T_prime)

            mask = (
                torch.arange(T_prime, device=hidden_states.device)
                .unsqueeze(0)
                < abs_lens.unsqueeze(1)
            )
            mask_f = mask.float().unsqueeze(-1)           # [batch, T', 1]
            pooled = (hidden_states * mask_f).sum(dim=1) \
                     / mask_f.sum(dim=1)                   # [batch, H]
        else:
            pooled = hidden_states.mean(dim=1)             # [batch, H]

        # ── 4. Optional L2 normalisation ──────────────────────────────────
        if self.output_norm:
            pooled = F.normalize(pooled, dim=-1)

        # ── 5. Insert dummy segment dimension to match ECAPA convention ───
        return pooled.unsqueeze(1)                         # [batch, 1, H]
