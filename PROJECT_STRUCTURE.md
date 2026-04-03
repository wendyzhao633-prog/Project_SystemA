# PROJECT_STRUCTURE.md

## Canonical WSL project root

Assume the main project root is:

```bash
~/4thyear/ravdess_avfusion
```

This directory is the project root for the fusion repo.

---

## Final target structure

```text
~/4thyear/ravdess_avfusion/
├── AGENT.md
├── PROJECT_STRUCTURE.md
├── DATA_CONTRACTS.md
├── CODEX_PROMPTS.md
├── README_START_HERE.md
├── configs/
│   ├── audio/
│   │   └── audio_profile_ecapa_best.yaml
│   ├── late_fusion/
│   └── early_fusion/
├── external/
│   ├── audio_ser_speechbrain/
│   │   └── speechbrain-develop-main/
│   │       ├── metadata/
│   │       ├── outputs/
│   │       │   ├── embeddings_ECAPA/
│   │       │   └── predictions_ECAPA/
│   │       └── recipes/
│   │           └── RAVDESS_emotion_recognition/
│   └── video_face_branch/
│       ├── checkpoints/
│       ├── scripts/
│       ├── specs/
│       └── reports/
├── data/
│   ├── raw/
│   │   └── ravdess/
│   │       ├── audio_speech/
│   │       │   ├── Actor_01/
│   │       │   ├── Actor_02/
│   │       │   └── ... Actor_24/
│   │       └── video_speech/
│   │           ├── Actor_01/
│   │           ├── Actor_02/
│   │           └── ... Actor_24/
│   ├── manifests/
│   │   ├── audio/
│   │   ├── video/
│   │   └── av/
│   ├── processed/
│   │   ├── audio/
│   │   ├── video/
│   │   │   ├── frames_5f/
│   │   │   ├── frame_embeddings_5f/
│   │   │   └── sequence_embeddings_5f/
│   │   └── av/
│   └── interim/
├── src/
│   ├── common/
│   ├── datasets/
│   ├── preprocess/
│   ├── models/
│   ├── fusion/
│   ├── train/
│   └── eval/
├── checkpoints/
└── outputs/
    ├── logs/
    ├── metrics/
    └── predictions/
```

---

## What goes where

## 1. Main fusion repo
Everything new that you and Codex write should go into:
- `src/`
- `configs/`
- `data/manifests/`
- `data/processed/`
- `outputs/`
- `checkpoints/`

The main repo is where canonical pairing, validation, late fusion, and early fusion live.

---

## 2. Audio colleague code

Put the full audio zip contents here:

```text
external/audio_ser_speechbrain/speechbrain-develop-main/
```

Keep the repo structure intact inside that folder.

Important existing paths:
- best audio hparams:
  `external/audio_ser_speechbrain/speechbrain-develop-main/recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml`
- best audio predictions:
  `external/audio_ser_speechbrain/speechbrain-develop-main/outputs/predictions_ECAPA`
- best audio embeddings:
  `external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA`

These are read-only inputs for late fusion.
Do not redesign the main fusion repo around the audio repo.

---

## 3. Video colleague package

Simplify the uploaded nested video package and put it under:

```text
external/video_face_branch/
```

Recommended simplified layout:

```text
external/video_face_branch/
├── checkpoints/
│   ├── best_backbone.pth
│   ├── best_uar.pth
│   ├── init_backbone.pth
│   ├── init_model.pth
│   └── last_epoch.pth
├── scripts/
│   ├── video_to_images_for_fusion.py
│   ├── csv_to_backbone_embeddings.py
│   └── manifest_emb_to_sequences.py
├── specs/
│   ├── backbone_fusion_spec.json
│   ├── run_config.json
│   └── Readme.txt
└── reports/
    ├── predictions_train.jsonl
    ├── predictions_val.jsonl
    ├── predictions_test.jsonl
    ├── train_per_class_metrics.json
    ├── val_per_class_metrics.json
    └── test_per_class_metrics.json
```

Important:
- do not keep the redundant double-nested outer folder
- do not treat this package as the canonical manifest source
- use it as a reference/artifact source only

---

## 4. Raw audio data

Put the RAVDESS speech audio files here:

```text
data/raw/ravdess/audio_speech/Actor_01/*.wav
...
data/raw/ravdess/audio_speech/Actor_24/*.wav
```

Use the speech subset only.

---

## 5. Raw video data

Flatten the downloaded raw video data into:

```text
data/raw/ravdess/video_speech/Actor_01/*.mp4
...
data/raw/ravdess/video_speech/Actor_24/*.mp4
```

If the original download looks like:

```text
Video_Speech_Actor_01/Actor_01/*.mp4
```

drop the outer `Video_Speech_Actor_01` folder and keep only the `Actor_01` directory under `video_speech/`.

---

## 6. Canonical manifests

The fusion repo should build its own manifests:

### Audio manifests
- `data/manifests/audio/audio_train.csv`
- `data/manifests/audio/audio_valid.csv`
- `data/manifests/audio/audio_test.csv`

### Video manifests
- `data/manifests/video/video_train.csv`
- `data/manifests/video/video_valid.csv`
- `data/manifests/video/video_test.csv`

### AV paired manifests
- `data/manifests/av/av_train.csv`
- `data/manifests/av/av_valid.csv`
- `data/manifests/av/av_test.csv`

---

## 7. Canonical processed video artifacts

When the main repo generates video features, put them here:

### Extracted frames
- `data/processed/video/frames_5f/...`

### Per-frame embeddings
- `data/processed/video/frame_embeddings_5f/...`

### Per-video sequence embeddings
- `data/processed/video/sequence_embeddings_5f/...`

---

## 8. Artifact reuse policy

### Late fusion
Late fusion should prefer reusing existing unimodal artifacts.

Examples:
- audio predictions from `predictions_ECAPA`
- audio embeddings from `embeddings_ECAPA`
- existing local video predictions
- existing local video sequence features

### Early fusion
Early fusion should be trained from raw or near-raw inputs.
Do not define early fusion as concatenating precomputed unimodal embeddings.

---

## 9. AV matching rule

Never pair audio and video with the full raw filename stem if `MM` differs.

Use:

```text
av_key = drop first segment MM from MM-VV-EE-II-SS-RR-AA
```

Examples:
- audio stem: `03-01-07-01-01-02-01`
- video stem: `02-01-07-01-01-02-01`
- shared `av_key`: `01-07-01-01-02-01`

This rule must be implemented once in a shared utility and reused everywhere.

---

## 10. The first milestone Codex should hit

Before any fusion model:
1. shared parser
2. actor split helper
3. canonical manifests
4. artifact inventory + validation
5. AV pairing

Only after that:
- late fusion
- early fusion
