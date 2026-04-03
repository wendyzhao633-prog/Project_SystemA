# AGENT.md

## Project identity

This repository is the main WSL repo for a RAVDESS speech-only audio-video fusion project.

Primary goal:
- build a clean multimodal pipeline for RAVDESS speech-only audio + video
- implement a strong late-fusion baseline by reusing the best available unimodal artifacts
- implement a true early-fusion baseline trained from raw or near-raw data
- keep colleague audio/video code as external branches, not as the main repo

This repo is the canonical fusion repo.
Do not treat the audio branch or the video branch as the project root.

---

## Scope

Use only the RAVDESS speech subset.
Do not use the song subset.

Tasks in scope:
1. shared parsing and manifest building
2. audio-video pairing
3. video preprocessing and sequence building
4. late-fusion baselines
5. true early-fusion baseline
6. unified evaluation

Tasks out of scope:
- LLM prompting
- text modality
- physiology modality
- changing the colleague unimodal codebases unless absolutely necessary

---

## Canonical repo layout

Expected structure:

- `external/audio_ser_speechbrain/`
- `external/video_face_branch/`
- `data/raw/ravdess/audio_speech/Actor_01 ... Actor_24`
- `data/raw/ravdess/video_speech/Actor_01 ... Actor_24`
- `data/manifests/audio/`
- `data/manifests/video/`
- `data/manifests/av/`
- `data/processed/audio/`
- `data/processed/video/`
- `data/processed/av/`
- `src/common/`
- `src/datasets/`
- `src/preprocess/`
- `src/models/`
- `src/fusion/`
- `src/train/`
- `src/eval/`
- `configs/`

Do not scatter canonical manifests and processed features inside `external/`.
`external/` is mainly for colleague code and reference artifacts.

---

## Non-negotiable rules

1. Speech-only only.
2. Fixed 8-class label order.
3. Fixed actor split:
   - train = actors 1-16
   - valid = actors 17-20
   - test = actors 21-24
4. Join audio and video using `av_key`, not the full filename stem.
5. Use Linux/WSL-safe paths only.
6. Do not hardcode Windows absolute paths.
7. Use `pathlib`.
8. Prefer reproducible scripts and small composable modules.
9. Do not modify files under `external/` unless absolutely necessary.
10. Do not redefine early fusion as concatenating precomputed audio and video embeddings.

---

## Fixed label order

The label order is fixed and must be used everywhere:

0. neutral
1. calm
2. happy
3. sad
4. angry
5. fearful
6. disgust
7. surprise

Any classifier head, confusion matrix, metrics report, manifest, or JSONL output must respect this order.

---

## RAVDESS filename contract

Canonical filename format:

`MM-VV-EE-II-SS-RR-AA`

Where:
- `MM` = modality
- `VV` = vocal channel / speech-song related code used in the dataset naming convention
- `EE` = emotion
- `II` = emotional intensity
- `SS` = statement
- `RR` = repetition
- `AA` = actor

Important pairing rule:
- audio and video must be paired using `av_key`
- `av_key` = filename without the first `MM` segment
- example:
  - audio: `03-01-07-01-01-02-01`
  - video: `02-01-07-01-01-02-01`
  - shared `av_key`: `01-07-01-01-02-01`

Never join audio and video on the full raw stem when `MM` differs.

---

## Canonical audio branch default

The colleague confirmed that the best audio model for this project is the ECAPA-based run driven by:

- hparams:
  `external/audio_ser_speechbrain/speechbrain-develop-main/recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml`

Treat this as the canonical default audio source.

Important facts:
- model family: ECAPA-TDNN over 80-dim Fbank
- canonical embedding dimension: 96
- fixed 8-class label order
- fixed actor split:
  - train = actors 1-16
  - valid = actors 17-20
  - test = actors 21-24

Preferred audio artifacts for fusion:
- predictions:
  `external/audio_ser_speechbrain/speechbrain-develop-main/outputs/predictions_ECAPA`
- embeddings:
  `external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA`

Use this ECAPA run as the canonical audio source for:
- late-fusion baselines
- feature-level fusion baselines
- audio-side sanity checks

Other audio runs such as wav2vec2 / wavlm / focal / weighted variants are secondary ablations.
Do not use them as the default source unless explicitly requested.

---

## Canonical video branch status

The uploaded video package is a partial reference package, not a clean fully canonical repo.

What is currently known:
- backbone reference: ResNet50
- input image format: RGB
- expected size: 256x256
- value range: [0, 1]
- do not apply ImageNet normalization unless explicitly validated
- frame embedding dimension: 2048
- sequence package uses 5 frames per sample
- canonical video sequence shape target: `(5, 2048)`

Important warnings:
- the uploaded video zip had nested folders and omitted some `.npy` files
- some helper modules used by the reference scripts were missing
- the uploaded reference package must not be treated as the canonical source of truth for manifests
- canonical video manifests must be rebuilt from raw RAVDESS video files
- if complete local video artifacts already exist on disk, they may be reused for late fusion after validation and aggregation

Use the video package mainly as:
- structure reference
- checkpoint reference
- feature format reference

---

## Definitions to preserve

### Late fusion
Late fusion means:
- audio and video are first processed by their own branch
- each branch yields probabilities, logits, and/or compact embeddings
- fusion happens at decision level or post-branch representation level

Examples:
- average probability fusion
- weighted probability fusion
- concatenate audio/video embeddings and train a fusion MLP

### True early fusion
True early fusion means:
- start from raw or near-raw modality inputs
- audio branch operates on waveform-derived or spectrogram-derived tensors
- video branch operates on raw frames or stacked frame tensors
- fusion happens before each branch has fully collapsed into a single final unimodal prediction

Important:
- concatenating precomputed ECAPA 96-d embeddings with precomputed video embeddings is not the requested true early-fusion baseline
- that counts as feature-level / intermediate fusion, not true early fusion

---

## Engineering mode split

### Late fusion mode
Late fusion should prefer artifact reuse:
- reuse existing unimodal audio artifacts whenever available
- reuse existing unimodal video artifacts whenever available
- only write fusion-specific code in this main repo
- do not retrain the unimodal audio branch by default
- do not retrain the unimodal video branch by default unless necessary to produce missing video-level outputs

### Early fusion mode
Early fusion should prefer raw-data training:
- load raw audio or near-raw audio tensors
- load raw video frames or canonical extracted frame tensors
- train a real multimodal early-fusion model from scratch inside this main repo

---

## Implementation priorities

### Priority 1: shared utilities
Implement under `src/common/`:
- RAVDESS filename parser
- label mapping
- actor split helper
- `av_key` builder
- JSONL / CSV helpers

### Priority 2: canonical manifests
Implement scripts to rebuild:
- audio manifests from raw audio
- video manifests from raw video
- AV paired manifests from audio + video manifests

### Priority 3: artifact reuse layer
Implement read-only integration layers for:
- canonical audio predictions and embeddings from ECAPA
- existing video predictions and/or sequence artifacts
- validation that all artifacts match the canonical label order, split rule, and speech-only rule

### Priority 4: video preprocessing
Implement, when needed:
- frame extraction
- optional face crop with fallback
- frame embedding extraction
- sequence building to `(5, 2048)`

### Priority 5: late fusion baseline
Implement:
- probability average baseline
- weighted probability average baseline
- embedding MLP baseline

### Priority 6: true early fusion baseline
Implement a real raw/near-raw multimodal model, ideally:
- audio CNN on log-mel spectrograms
- video CNN on frame tensors
- fusion CNN / MLP head after intermediate fusion

---

## Coding rules

- Use Python modules under `src/`
- Prefer typed functions where practical
- Keep file IO and model code separate
- Use config-driven paths
- Use reproducible random seeds
- Save train/valid/test outputs separately
- Prefer small unit-testable helpers
- Avoid giant monolithic notebooks
- Keep CLI entry points thin
- Document assumptions at the top of each script

---

## Manifest expectations

Canonical manifests must be plain CSV or JSONL and live under:
- `data/manifests/audio/`
- `data/manifests/video/`
- `data/manifests/av/`

They must not depend on the partially uploaded video package.
They must be rebuildable from raw data.

Required contracts are described in `DATA_CONTRACTS.md`.

---

## Evaluation expectations

Minimum evaluation for classifiers:
- accuracy
- macro F1
- UAR
- confusion matrix

Where applicable also save:
- per-class recall
- logits or probabilities
- predictions JSONL
- manifest-linked outputs for traceability

---

## Output locations

Suggested output locations:
- checkpoints: `checkpoints/`
- logs: `outputs/logs/`
- predictions: `outputs/predictions/`
- metrics: `outputs/metrics/`
- processed features: `data/processed/`

Keep paths consistent and easy to rerun.

---

## What Codex should do first

Before writing fusion models:
1. read this file
2. read `PROJECT_STRUCTURE.md`
3. read `DATA_CONTRACTS.md`
4. read `configs/audio/audio_profile_ecapa_best.yaml`
5. inspect the actual local repo layout and existing artifacts
6. implement shared parser + manifest generation first
7. implement artifact integration and validation next
8. only then implement late fusion
9. only after that implement true early fusion

Do not skip the manifest and pairing stage.
That is the foundation of the whole project.
