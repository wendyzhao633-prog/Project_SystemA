# DATA_CONTRACTS.md

## Purpose

This document defines the canonical data contracts for the RAVDESS speech-only audio-video fusion repo.

It exists to prevent:
- path mismatches
- label-order mismatches
- audio-video pairing mistakes
- frame-level vs video-level confusion
- accidental use of non-speech data
- accidental use of noncanonical audio experiment folders

All code in `src/` must respect these contracts.

---

## 1. Fixed class order

Use this exact label order everywhere:

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

The following must all use this order:
- `label_id`
- classifier heads
- confusion matrices
- metrics summaries
- JSONL predictions
- one-hot encodings
- probability vectors
- fusion logits interpretation

---

## 2. Filename grammar

Canonical filename pattern:

`MM-VV-EE-II-SS-RR-AA`

Where:
- `MM` = modality
- `VV` = speech/song-related code segment used in the dataset naming convention
- `EE` = emotion code
- `II` = intensity code
- `SS` = statement code
- `RR` = repetition code
- `AA` = actor code

Example audio filename stem:

`03-01-07-01-01-02-01`

Example video filename stem:

`02-01-07-01-01-02-01`

---

## 3. Canonical join key

### 3.1 Rule
Audio and video must be paired using `av_key`, not the full filename stem.

### 3.2 Definition
`av_key` = filename stem with the first segment `MM` removed.

Examples:
- audio stem: `03-01-07-01-01-02-01`
- video stem: `02-01-07-01-01-02-01`
- shared `av_key`: `01-07-01-01-02-01`

### 3.3 Why this rule exists
Different modalities can have different `MM` values even when the utterance is the same sample.

### 3.4 Non-negotiable
Any audio-video manifest builder, pairing utility, late-fusion dataset, or evaluation script must use `av_key`.

---

## 4. Canonical split rule

Use actor-based splits only:

- train = actors 1-16
- valid = actors 17-20
- test = actors 21-24

This applies to:
- audio
- video
- AV paired manifests
- feature caches
- predictions
- metrics

Never random-split at the file level for the canonical baseline.

---

## 5. Dataset scope rules

### 5.1 Audio raw data
Canonical audio raw root:

`data/raw/ravdess/audio_speech/`

Expected layout:

`data/raw/ravdess/audio_speech/Actor_01/*.wav`
...
`data/raw/ravdess/audio_speech/Actor_24/*.wav`

Use only speech-only audio files for the project baseline.

### 5.2 Video raw data
Canonical video raw root:

`data/raw/ravdess/video_speech/`

Expected layout:

`data/raw/ravdess/video_speech/Actor_01/*.mp4`
...
`data/raw/ravdess/video_speech/Actor_24/*.mp4`

Canonical manifests must be rebuilt from raw video files.
Do not treat the partially uploaded video package as the canonical manifest source.

### 5.3 Song exclusion
Do not include song data in the canonical baseline.
Any manifest builder must either:
- explicitly filter to speech-only files
- or validate filename fields and exclude song-related files

---

## 6. Canonical audio profile

Default audio source for fusion is the colleague-confirmed best ECAPA run.

Canonical paths:
- hparams:
  `external/audio_ser_speechbrain/speechbrain-develop-main/recipes/RAVDESS_emotion_recognition/hparams/train_ravdess.yaml`
- predictions root:
  `external/audio_ser_speechbrain/speechbrain-develop-main/outputs/predictions_ECAPA`
- embeddings root:
  `external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA`

Canonical audio facts:
- model family: ECAPA-TDNN over 80-dim Fbank
- embedding dimension: 96
- num classes: 8
- split: actor-based as above

Unless explicitly overridden, fusion code should prefer these ECAPA artifacts over other audio experiment folders.

---

## 7. Canonical manifest schemas

## 7.1 Audio raw manifest

Suggested path:
`data/manifests/audio/audio_{train,valid,test}.csv`

Required columns:
- `sample_id`
- `stem`
- `av_key`
- `wav_path`
- `actor_id`
- `split`
- `emotion_code`
- `label_id`
- `label_name`
- `intensity_code`
- `statement_code`
- `repetition_code`

Optional columns:
- `duration_sec`
- `source_root`
- `notes`

Example:

```csv
sample_id,stem,av_key,wav_path,actor_id,split,emotion_code,label_id,label_name,intensity_code,statement_code,repetition_code
03-01-07-01-01-02-01,03-01-07-01-01-02-01,01-07-01-01-02-01,data/raw/ravdess/audio_speech/Actor_01/03-01-07-01-01-02-01.wav,1,train,07,6,disgust,01,01,02
```

---

## 7.2 Video raw manifest

Suggested path:
`data/manifests/video/video_{train,valid,test}.csv`

Required columns:
- `sample_id`
- `stem`
- `av_key`
- `video_path`
- `actor_id`
- `split`
- `emotion_code`
- `label_id`
- `label_name`
- `intensity_code`
- `statement_code`
- `repetition_code`

Optional columns:
- `num_frames`
- `fps`
- `duration_sec`
- `source_root`
- `notes`

Example:

```csv
sample_id,stem,av_key,video_path,actor_id,split,emotion_code,label_id,label_name,intensity_code,statement_code,repetition_code
02-01-07-01-01-02-01,02-01-07-01-01-02-01,01-07-01-01-02-01,data/raw/ravdess/video_speech/Actor_01/02-01-07-01-01-02-01.mp4,1,train,07,6,disgust,01,01,02
```

---

## 7.3 AV paired manifest

Suggested path:
`data/manifests/av/av_{train,valid,test}.csv`

Required columns:
- `av_key`
- `split`
- `actor_id`
- `label_id`
- `label_name`
- `emotion_code`
- `audio_sample_id`
- `audio_path`
- `video_sample_id`
- `video_path`

Recommended columns:
- `audio_prediction_path`
- `audio_embedding_path`
- `video_prediction_path`
- `video_sequence_path`

Example:

```csv
av_key,split,actor_id,label_id,label_name,emotion_code,audio_sample_id,audio_path,video_sample_id,video_path
01-07-01-01-02-01,train,1,6,disgust,07,03-01-07-01-01-02-01,data/raw/ravdess/audio_speech/Actor_01/03-01-07-01-01-02-01.wav,02-01-07-01-01-02-01,data/raw/ravdess/video_speech/Actor_01/02-01-07-01-01-02-01.mp4
```

---

## 7.4 Video frame manifest

Suggested path:
`data/manifests/video/video_frames_5f_{train,valid,test}.csv`

One row = one frame.

Required columns:
- `frame_sample_id`
- `video_sample_id`
- `av_key`
- `frame_idx`
- `image_path`
- `label_id`
- `label_name`
- `actor_id`
- `split`

Recommended columns:
- `face_found`
- `crop_mode`
- `source_video_path`

Example:

```csv
frame_sample_id,video_sample_id,av_key,frame_idx,image_path,label_id,label_name,actor_id,split,face_found,crop_mode
02-01-07-01-01-02-01_f000003,02-01-07-01-01-02-01,01-07-01-01-02-01,3,data/processed/video/frames_5f/train/02-01-07-01-01-02-01/f000003.png,6,disgust,1,train,1,haar_face
```

---

## 7.5 Video frame-embedding manifest

Suggested path:
`data/manifests/video/video_frame_embeddings_5f_{train,valid,test}.csv`

One row = one frame embedding.

Required columns:
- `frame_sample_id`
- `video_sample_id`
- `av_key`
- `frame_idx`
- `embedding_path`
- `embedding_dim`
- `label_id`
- `label_name`
- `actor_id`
- `split`

Embedding shape contract:
- each frame embedding must be shape `(2048,)`
- dtype should be `float32`

---

## 7.6 Video sequence manifest

Suggested path:
`data/manifests/video/video_sequences_5f_{train,valid,test}.csv`

One row = one video-level short sequence.

Required columns:
- `video_sample_id`
- `av_key`
- `sequence_path`
- `num_frames`
- `feature_dim`
- `label_id`
- `label_name`
- `actor_id`
- `split`

Sequence shape contract:
- canonical shape: `(5, 2048)`
- dtype should be `float32`

Example:

```csv
video_sample_id,av_key,sequence_path,num_frames,feature_dim,label_id,label_name,actor_id,split
02-01-07-01-01-02-01,01-07-01-01-02-01,data/processed/video/sequence_embeddings_5f/train/02-01-07-01-01-02-01.npy,5,2048,6,disgust,1,train
```

---

## 8. Canonical audio artifact contracts

## 8.1 Audio predictions JSONL

Canonical root:
`external/audio_ser_speechbrain/speechbrain-develop-main/outputs/predictions_ECAPA`

Expected files:
- `valid_predictions.jsonl`
- `test_predictions.jsonl`
- possibly `train_predictions.jsonl` if available or exported later

Required fields per JSONL row:
- `sample_id`
- `label_id`
- `pred_id`
- `pred_name`
- `probs`

Optional fields:
- `logits`
- `wav_path`

Probability vector contract:
- length = 8
- class order = fixed canonical label order

Example:

```json
{"sample_id":"03-01-07-01-01-02-21","label_id":6,"pred_id":6,"pred_name":"disgust","probs":[0.01,0.02,0.03,0.01,0.04,0.05,0.80,0.04]}
```

---

## 8.2 Audio embedding index JSONL

Canonical root:
`external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA`

Expected files:
- `train_embedding_index.jsonl`
- `valid_embedding_index.jsonl`
- `test_embedding_index.jsonl`

Required fields per row:
- `sample_id`
- `embedding_path`
- `embedding_dim`
- `label_id`
- `split`

Embedding contract:
- canonical embedding dim = 96
- each `.npy` should load to shape `(96,)`
- dtype should be `float32`

Example:

```json
{"sample_id":"03-01-07-01-01-02-21","embedding_path":"external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA/test/03-01-07-01-01-02-21.npy","embedding_dim":96,"label_id":6,"split":"test"}
```

---

## 9. Canonical video artifact contracts

Because the uploaded video package is partial, treat its artifacts as a reference format, not a canonical source of truth.

### 9.1 Frame embeddings
Contract:
- shape `(2048,)`
- dtype `float32`

### 9.2 Sequence embeddings
Contract:
- canonical sequence shape `(5, 2048)`
- dtype `float32`

### 9.3 Frame-level prediction JSONL
If existing video prediction JSONLs are frame-level, each row should be linked to:
- `frame_sample_id` or equivalent
- `video_sample_id`
- `label_id`
- `pred_id`
- `probs`

### 9.4 Video-level predictions
For canonical late fusion, video predictions must be at video level, not frame level.

If only frame-level predictions exist, they must be aggregated before late fusion.

Recommended aggregation methods:
- mean probabilities across frames
- mean logits across frames
- sequence-head prediction from `(5,2048)` input

Canonical late fusion should consume video-level outputs.

---

## 10. Late-fusion reuse mode contracts

Late fusion is allowed to reuse colleague artifacts instead of retraining unimodal branches.

### 10.1 Rule-based late fusion
If doing:
- probability average fusion
- weighted probability average fusion

Then minimum requirements are:
- audio valid/test sample-level probabilities
- video valid/test video-level probabilities or frame-level probabilities that can be aggregated to video-level
- valid/test AV pairing by `av_key`

### 10.2 Learned late fusion
If doing:
- embedding MLP
- learned fusion over probs/logits

Then recommended requirements are:
- train/valid/test sample-level artifact availability
- audio embeddings or predictions for train/valid/test
- video sequence features and/or video-level predictions for train/valid/test
- canonical AV paired manifests

If `train_predictions.jsonl` is missing on the audio side, it is acceptable to export predictions again without retraining the audio model.

---

## 11. Late-fusion input contracts

Late-fusion code may support three practical modes.

### 11.1 Probabilities mode
Required per paired sample:
- `audio_probs` shape `(8,)`
- `video_probs` shape `(8,)`

### 11.2 Logits mode
Required per paired sample:
- `audio_logits` shape `(8,)`
- `video_logits` shape `(8,)`

### 11.3 Embedding mode
Required per paired sample:
- `audio_embedding` shape `(96,)`
- `video_embedding` shape determined by your video head
  - recommended to pool `(5,2048)` into a compact vector before fusion MLP
  - or flatten to `(10240,)` if explicitly intended and memory permits

The late-fusion dataset class must align paired samples by `av_key`.

---

## 12. Early-fusion input contracts

True early fusion must start from raw or near-raw modality tensors.

Recommended canonical input:
- audio: log-mel spectrogram tensor derived from raw wav
- video: 5-frame image tensor from raw video or canonical extracted frames

Do not define true early fusion as:
- `concat(audio_ecapa_embedding, video_embedding)`

That is feature-level / intermediate fusion, not the requested true early-fusion baseline.

---

## 13. Metrics contracts

Minimum metrics for every canonical experiment:
- accuracy
- macro F1
- UAR
- confusion matrix

Recommended additional outputs:
- per-class recall
- JSONL predictions
- metrics JSON
- config snapshot used for the run

Suggested output layout:
- `outputs/metrics/...`
- `outputs/predictions/...`
- `checkpoints/...`

---

## 14. Path rules

All canonical paths in manifests and config files should be:
- repo-relative where practical
- or WSL/Linux-safe absolute paths if explicitly needed

Do not store Windows absolute paths in canonical manifests.

---

## 15. Safety against partial reference packages

The uploaded video reference zip had:
- nested folders
- omitted `.npy` files
- missing helper modules
- potentially mixed manifest content

Therefore:
- do not trust its CSV/JSONL as canonical manifests
- rebuild canonical manifests from raw video files
- use the reference package mainly to understand expected tensor formats and checkpoint usage
- if complete local video artifacts already exist on disk, validate them before reuse
