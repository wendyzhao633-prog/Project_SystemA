# EARLY_FUSION_MODEL_BLUEPRINTS.md

## Purpose

This document defines **three concrete early-fusion / strong feature-fusion model blueprints** for the RAVDESS speech-only audio-video project.

These blueprints are intended for Codex implementation inside the main fusion repo.

They are designed to satisfy the project requirement that **early fusion should be meaningfully different from the already completed late-fusion baseline**.

Important:
- Do **not** define early fusion as concatenating precomputed ECAPA embeddings with precomputed video sequence embeddings only.
- Use **raw or near-raw** modality inputs.
- Reuse the canonical AV manifests and `av_key` pairing.
- Use the fixed 8-class label order.

---

## Fixed project constraints

### Dataset scope
- Use **RAVDESS speech-only only**.
- Do not use song samples.
- Use the canonical paired AV manifests:
  - `data/manifests/av/av_train.csv`
  - `data/manifests/av/av_valid.csv`
  - `data/manifests/av/av_test.csv`

### Fixed split
- train = actors 1-16
- valid = actors 17-20
- test = actors 21-24

### Fixed label order
```text
0 neutral
1 calm
2 happy
3 sad
4 angry
5 fearful
6 disgust
7 surprise
```

### Join key
- The canonical join key is `av_key`
- Never rely on full filename stem equality when modality differs

---

## Shared input pipeline for all three models

These three models should share as much of the data pipeline as practical.

### Audio input
Use **near-raw audio**, not precomputed ECAPA embeddings.

Recommended pipeline:
1. Load waveform from `audio_path`
2. Convert to mono if needed
3. Resample to `16 kHz`
4. Normalize amplitude
5. Compute log-mel spectrogram on the fly

Recommended spectrogram config:
- sample_rate = 16000
- n_fft = 400
- hop_length = 160
- win_length = 400
- n_mels = 80
- log transform = yes

Recommended tensor shape:
- `audio_input`: `(1, 80, T)`

Engineering simplification:
- Fix or pad/truncate to a uniform time axis, e.g. `T = 256` or another repo-wide constant.
- If needed, cache spectrogram tensors under a repo-owned cache path.

### Video input
Use **near-raw visual input**, not precomputed `(5,2048)` sequence embeddings as the main model input.

Recommended pipeline:
1. Load video from `video_path`
2. Sample `5` evenly spaced frames
3. Try face crop if a face detector is available
4. Fallback to center crop if face detection fails
5. Resize to fixed size
6. Normalize to `[0,1]`

Recommended frame config:
- num_frames = 5
- image size = `224 x 224`
- RGB channels

Recommended tensor shape:
- `video_input`: `(5, 3, 224, 224)`

Engineering simplification:
- It is acceptable to implement a repo-owned frame cache under `data/processed/early_fusion_frames_5f/`, but the model should still conceptually consume raw/near-raw frames rather than precomputed high-level embeddings.

### Shared dataset contract
The early-fusion dataset class should yield:
- `av_key`
- `label_id`
- `audio_input`
- `video_input`
- optional metadata for debugging

Recommended dataset module:
- `src/datasets/early_fusion_dataset.py`

---

## Model 1: `early_fusion_3cnn`

## Goal
This is the **main low-risk early-fusion baseline**.

It should be simple, trainable, and easy to compare with late fusion.

## Why it counts as early fusion
It uses separate near-raw modality encoders and fuses the intermediate feature maps **before** either modality collapses into a final unimodal prediction.

## Architecture summary
- **Audio CNN**: log-mel spectrogram -> audio feature map
- **Video CNN**: 5-frame tensor -> video feature map
- **Fusion CNN**: concatenated intermediate feature maps -> fused representation -> classifier

## Detailed sketch

### Audio CNN branch
Input:
- `(1, 80, T)`

Suggested stack:
1. Conv2d(1, 32, kernel=3, stride=1, padding=1) + BN + ReLU
2. MaxPool2d(2)
3. Conv2d(32, 64, kernel=3, stride=1, padding=1) + BN + ReLU
4. MaxPool2d(2)
5. Conv2d(64, 128, kernel=3, stride=1, padding=1) + BN + ReLU
6. AdaptiveAvgPool2d((8, 8))

Output:
- `audio_feat_map`: `(128, 8, 8)`

### Video CNN branch
Input:
- `(5, 3, 224, 224)`

Recommended implementation options:
- shared 2D CNN over each frame, then temporal aggregation
- or a lightweight 3D CNN

Preferred simple option:
1. Apply the same 2D CNN frame encoder to each frame:
   - Conv2d(3, 32, 3, padding=1) + BN + ReLU + MaxPool2d(2)
   - Conv2d(32, 64, 3, padding=1) + BN + ReLU + MaxPool2d(2)
   - Conv2d(64, 128, 3, padding=1) + BN + ReLU
2. Temporal average over the 5 frame feature maps
3. AdaptiveAvgPool2d((8, 8))

Output:
- `video_feat_map`: `(128, 8, 8)`

### Fusion CNN branch
Fusion point:
- Concatenate along channel dimension:
  - `(128, 8, 8)` + `(128, 8, 8)` -> `(256, 8, 8)`

Suggested fusion head:
1. Conv2d(256, 256, kernel=3, padding=1) + BN + ReLU
2. Conv2d(256, 128, kernel=3, padding=1) + BN + ReLU
3. GlobalAveragePooling -> `(128,)`
4. Dropout(0.3)
5. Linear(128, 8)

## Output
- logits `(8,)`

## Notes
- This should be the first early-fusion model to implement.
- It is the safest baseline and should be the reference early-fusion model.

---

## Model 2: `early_fusion_bilinear`

## Goal
This model should capture **explicit audio-video feature interactions** better than plain concatenation.

## Why it counts as strong feature fusion
The audio and video branches still start from near-raw inputs, but the fusion stage uses **low-rank bilinear / tensor-style interaction**, which is a stronger feature-fusion mechanism than concatenation.

## Important constraint
Do **not** implement a full tensor fusion with explosive dimensionality.
Use a compact and practical form:
- low-rank tensor fusion
- or compact bilinear style fusion

## Recommended implementation: low-rank bilinear fusion

### Audio encoder
Use a lightweight CNN on log-mel spectrogram.

Input:
- `(1, 80, T)`

Suggested output after CNN + pooling:
- `audio_vec`: `(256,)`

### Video encoder
Use a lightweight frame encoder + temporal pooling.

Input:
- `(5, 3, 224, 224)`

Suggested output after CNN + temporal pooling:
- `video_vec`: `(256,)`

### Low-rank bilinear fusion block
Use rank `R = 4` and fusion hidden size `H = 128`.

Recommended block:
1. Build `R` audio projections: `Linear(256, 128)`
2. Build `R` video projections: `Linear(256, 128)`
3. For each rank component `r`:
   - `a_r = GELU(Wa_r(audio_vec))`
   - `v_r = GELU(Wv_r(video_vec))`
   - `h_r = a_r * v_r`  (Hadamard product)
4. Fuse:
   - `fused = (h_1 + h_2 + ... + h_R) / R`
5. LayerNorm + Dropout + MLP
6. Final Linear -> 8 logits

### Suggested head
1. LayerNorm(128)
2. Dropout(0.3)
3. Linear(128, 128) + GELU
4. Dropout(0.3)
5. Linear(128, 8)

## Output
- logits `(8,)`

## Why this version is preferred
- Much easier than full TFN
- Stronger than simple concat
- Practical for your dataset size
- Codex can implement it cleanly without huge memory cost

## Notes
- If Codex prefers an equivalent `nn.Bilinear`-style block, that is acceptable as long as it remains compact.
- But the low-rank multi-branch Hadamard design is preferred because it is more stable and explicitly low-rank.

---

## Model 3: `early_fusion_xattn`

## Goal
This is the **higher-capacity early-fusion / multimodal interaction model**.

It should allow the model to learn which audio regions attend to which video cues.

## Why it counts as early fusion
The model still starts from near-raw inputs, converts them into intermediate modality tokens, and fuses them with cross-modal attention **before** final classification.

## Important constraint
Keep it **tiny**.
Do not build a large transformer.

Recommended limits:
- hidden dim = 128
- num heads = 4
- num cross-attention layers = 1 or 2
- small FFN width

## Architecture summary
- Audio CNN backbone -> audio tokens
- Video CNN backbone -> video tokens
- Cross-attention fusion block
- Pool fused tokens -> classifier

## Detailed sketch

### Audio token encoder
Input:
- `(1, 80, T)`

Suggested steps:
1. CNN backbone to produce feature map `(128, H_a, W_a)`
2. Adaptive pooling to a small grid, e.g. `(4, 4)`
3. Flatten spatial grid into tokens

Output:
- `audio_tokens`: `(16, 128)`

### Video token encoder
Input:
- `(5, 3, 224, 224)`

Suggested steps:
1. Shared 2D CNN frame encoder to produce each frame feature map `(128, h, w)`
2. Pool each frame feature map to `(2, 2)`
3. Flatten each frame grid into 4 tokens
4. Across 5 frames this yields `20` tokens

Output:
- `video_tokens`: `(20, 128)`

### Cross-attention fusion block
Preferred simple design:
1. Audio-to-video cross-attention:
   - query = audio tokens
   - key/value = video tokens
2. Video-to-audio cross-attention:
   - query = video tokens
   - key/value = audio tokens
3. Residual + LayerNorm after each block
4. Optional small FFN block

If needed, use just 1 fusion layer first.

### Pooling and classifier
After cross-attention:
1. Mean-pool updated audio tokens -> `(128,)`
2. Mean-pool updated video tokens -> `(128,)`
3. Concatenate -> `(256,)`
4. Dropout(0.3)
5. Linear(256, 128) + GELU
6. Linear(128, 8)

## Output
- logits `(8,)`

## Notes
- This model is likely to be the most expressive, but also the riskiest for overfitting.
- Keep it small and baseline-oriented.
- Do not let Codex build a huge transformer stack.

---

## Shared training plan for the three models

## Loss
Use standard 8-class cross-entropy.

## Metrics
For every model report:
- accuracy
- macro F1
- UAR
- confusion matrix

## Optimization
Recommended baseline settings:
- optimizer: AdamW
- lr: start with `1e-3` for small custom models
- weight_decay: `1e-4`
- epochs: modest, e.g. `20-40`
- early stopping on validation UAR

## Regularization
- dropout in heads
- lightweight augmentation only if easy to implement
- avoid overcomplicating the first pass

## Save outputs under
- `checkpoints/early_fusion/...`
- `outputs/metrics/early_fusion/...`
- `outputs/predictions/early_fusion/...`

---

## Expected code organization

Recommended structure:
- `src/datasets/early_fusion_dataset.py`
- `src/models/early_fusion_3cnn.py`
- `src/models/early_fusion_bilinear.py`
- `src/models/early_fusion_xattn.py`
- `src/train/run_early_fusion.py`
- `configs/early_fusion/early_fusion_3cnn.yaml`
- `configs/early_fusion/early_fusion_bilinear.yaml`
- `configs/early_fusion/early_fusion_xattn.yaml`

Prefer one shared training/evaluation entry point with a model-name switch instead of three unrelated pipelines.

---

## What Codex should avoid

Do NOT:
- use precomputed ECAPA embeddings as the main audio input for early fusion
- use precomputed `(5,2048)` video sequence embeddings as the main visual input for early fusion
- define early fusion as just concatenating those two precomputed embeddings
- build a very heavy transformer or very deep 3D CNN
- ignore the fixed split and label order

---

## Recommended implementation order

Implement in this order:
1. `early_fusion_3cnn`
2. `early_fusion_bilinear`
3. `early_fusion_xattn`

Reason:
- 3CNN is the safest baseline
- bilinear fusion adds stronger feature interaction
- cross-attention is the highest-risk/highest-capacity model

---

## Definition summary for Codex

- `early_fusion_3cnn` = main low-risk true early-fusion baseline
- `early_fusion_bilinear` = compact explicit feature-interaction model
- `early_fusion_xattn` = small cross-modal attention model

All three must be implemented as **comparable early-fusion / feature-fusion baselines** under the same data contract and evaluation scheme.
