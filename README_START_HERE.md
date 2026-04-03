# README_START_HERE.md

This bundle is a self-contained Codex handoff for the RAVDESS speech-only audio-video fusion project.

It encodes the current project decisions:
- use RAVDESS speech-only data
- use late fusion as a baseline
- use true early fusion as a new trained multimodal model
- reuse the colleague-confirmed best ECAPA audio artifacts for late fusion by default
- keep colleague unimodal code in `external/`
- build canonical manifests and fusion code in the main repo

---

## 1. Create the main repo root in WSL

Suggested root:

```bash
mkdir -p ~/4thyear/ravdess_avfusion
cd ~/4thyear/ravdess_avfusion
```

Copy the bundle files into that root.

---

## 2. Place colleague code and data

### Audio code
Put the full audio repo here:

```text
external/audio_ser_speechbrain/speechbrain-develop-main/
```

### Audio raw data
Put RAVDESS speech audio here:

```text
data/raw/ravdess/audio_speech/Actor_01/*.wav
...
data/raw/ravdess/audio_speech/Actor_24/*.wav
```

### Video code
Simplify the nested uploaded video package and place it here:

```text
external/video_face_branch/
```

### Video raw data
Flatten raw video files into:

```text
data/raw/ravdess/video_speech/Actor_01/*.mp4
...
data/raw/ravdess/video_speech/Actor_24/*.mp4
```

---

## 3. What to tell Codex first

Open Codex in the repo root and send:

1. Prompt 0 from `CODEX_PROMPTS.md`
2. Prompt 1
3. Prompt 2
4. Prompt 3
5. Prompt 5
6. Prompt 6
7. Prompt 7

Only use Prompt 4 if you need Codex to rebuild canonical video artifacts from raw video files rather than just reuse existing local video artifacts.

Recommended order:
- inspect
- bootstrap parser/manifests
- audio integration
- video integration
- late fusion
- early fusion
- tests

---

## 4. Engineering policy

### Late fusion
Prefer artifact reuse:
- audio predictions/embeddings from the best ECAPA run
- existing local video predictions/sequence features if valid

### Early fusion
Train from raw/near-raw data:
- audio spectrograms
- video frames
- a real early-fusion model

---

## 5. Canonical files in this bundle

- `AGENT.md`
- `PROJECT_STRUCTURE.md`
- `DATA_CONTRACTS.md`
- `CODEX_PROMPTS.md`
- `configs/audio/audio_profile_ecapa_best.yaml`

Read them in that order.

---

## 6. Important warning

The uploaded video reference package used for discussion was partial:
- some `.npy` files were removed for upload size reasons
- some helper modules were missing
- nested folders were present

Your real local project can still use the complete local files.
Codex should inspect the actual local filesystem first, then adapt.
