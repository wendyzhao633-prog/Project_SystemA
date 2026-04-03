import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.preprocess.audio_ecapa import (
    build_ecapa_layout,
    build_fusion_ready_audio_tables,
    count_jsonl_rows,
    find_best_checkpoint,
    normalize_embedding_artifacts,
    normalize_prediction_records,
    sample_index_from_manifests,
    validate_embedding_file,
)


class AudioEcapaTests(unittest.TestCase):
    def test_find_best_checkpoint_uses_highest_validation_uar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_root = Path(tmpdir) / "save"
            ckpt_a = save_root / "CKPT+1"
            ckpt_b = save_root / "CKPT+2"
            ckpt_a.mkdir(parents=True)
            ckpt_b.mkdir(parents=True)
            (ckpt_a / "CKPT.yaml").write_text("uar: 0.51\nacc: 0.60\n", encoding="utf-8")
            (ckpt_b / "CKPT.yaml").write_text("uar: 0.56\nacc: 0.57\n", encoding="utf-8")

            best = find_best_checkpoint(save_root)

            self.assertEqual(best.checkpoint_dir, ckpt_b)
            self.assertAlmostEqual(best.uar, 0.56)

    def test_normalize_prediction_records_rewrites_windows_paths_and_checks_probs(self) -> None:
        sample_index = {
            "03-01-01-01-01-01-17": {
                "sample_id": "03-01-01-01-01-01-17",
                "stem": "03-01-01-01-01-01-17",
                "av_key": "01-01-01-01-01-17",
                "wav_path": "data/raw/ravdess/audio_speech/Audio_Speech_Actors_01-24/Actor_17/03-01-01-01-01-01-17.wav",
                "actor_id": 17,
                "split": "valid",
                "emotion_code": "01",
                "label_id": 0,
                "label_name": "neutral",
                "intensity_code": "01",
                "statement_code": "01",
                "repetition_code": "01",
            }
        }

        normalized = normalize_prediction_records(
            [
                {
                    "sample_id": "03-01-01-01-01-01-17",
                    "split": "valid",
                    "actor": 17,
                    "wav": "C:/Users/example.wav",
                    "emotion_code": "01",
                    "label_id": 0,
                    "label_name": "neutral",
                    "pred_id": 5,
                    "pred_name": "fearful",
                    "probs": [0.1] * 8,
                }
            ],
            split="valid",
            sample_index=sample_index,
        )

        self.assertEqual(normalized[0]["wav"], sample_index["03-01-01-01-01-01-17"]["wav_path"])
        self.assertEqual(normalized[0]["av_key"], "01-01-01-01-01-17")

        with self.assertRaisesRegex(ValueError, "length 8"):
            normalize_prediction_records(
                [
                    {
                        "sample_id": "03-01-01-01-01-01-17",
                        "pred_id": 1,
                        "pred_name": "calm",
                        "probs": [0.1] * 7,
                    }
                ],
                split="valid",
                sample_index=sample_index,
            )

    def test_normalize_embedding_artifacts_copies_and_rewrites_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "nested/test"
            destination_root = root / "canonical"
            source_dir.mkdir(parents=True)

            sample_id = "03-01-01-01-01-01-21"
            np.save(source_dir / f"{sample_id}.npy", np.zeros((96,), dtype=np.float32))
            source_index_path = root / "nested/test_embedding_index.jsonl"
            source_index_path.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "embedding_path": "C:/Users/NannanLi/Desktop/speechbrain-develop/outputs/embeddings/test/03-01-01-01-01-01-21.npy",
                        "embedding_dim": 96,
                        "label_id": 0,
                        "split": "test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            sample_index = {
                sample_id: {
                    "sample_id": sample_id,
                    "av_key": "01-01-01-01-01-21",
                    "label_id": 0,
                }
            }

            count = normalize_embedding_artifacts(
                split="test",
                source_dir=source_dir,
                destination_root=destination_root,
                sample_index=sample_index,
                source_index_path=source_index_path,
            )

            self.assertEqual(count, 1)
            validate_embedding_file(destination_root / "test" / f"{sample_id}.npy")
            self.assertEqual(count_jsonl_rows(destination_root / "test_embedding_index.jsonl"), 1)

    def test_build_fusion_ready_audio_tables_joins_by_sample_and_av_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/manifests/audio").mkdir(parents=True)
            (root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/predictions_ECAPA").mkdir(parents=True)
            (root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA/train").mkdir(parents=True)
            (root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA/valid").mkdir(parents=True)
            (root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA/test").mkdir(parents=True)
            (root / "external/audio_ser_speechbrain/speechbrain-develop-main/recipes/RAVDESS_emotion_recognition/results/ECAPA-TDNN/ECAPA-TDNN/1968/save/CKPT+x").mkdir(parents=True)
            (root / "external/audio_ser_speechbrain/speechbrain-develop-main/recipes/RAVDESS_emotion_recognition/results/ECAPA-TDNN/ECAPA-TDNN/1968/save/CKPT+x/CKPT.yaml").write_text("uar: 0.6\n", encoding="utf-8")

            split_specs = {
                "train": ("03-01-01-01-01-01-01", 1),
                "valid": ("03-01-01-01-01-01-17", 17),
                "test": ("03-01-01-01-01-01-21", 21),
            }

            for split, (sample_id, actor_id) in split_specs.items():
                manifest_path = root / "data/manifests/audio" / f"audio_{split}.csv"
                with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "sample_id",
                            "stem",
                            "av_key",
                            "wav_path",
                            "actor_id",
                            "split",
                            "emotion_code",
                            "label_id",
                            "label_name",
                            "intensity_code",
                            "statement_code",
                            "repetition_code",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "stem": sample_id,
                            "av_key": "01-01-01-01-01-" + f"{actor_id:02d}",
                            "wav_path": f"data/raw/{sample_id}.wav",
                            "actor_id": actor_id,
                            "split": split,
                            "emotion_code": "01",
                            "label_id": 0,
                            "label_name": "neutral",
                            "intensity_code": "01",
                            "statement_code": "01",
                            "repetition_code": "01",
                        }
                    )

                predictions_path = root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/predictions_ECAPA" / f"{split}_predictions.jsonl"
                predictions_path.write_text(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "actor": actor_id,
                            "actor_id": actor_id,
                            "av_key": "01-01-01-01-01-" + f"{actor_id:02d}",
                            "wav": f"data/raw/{sample_id}.wav",
                            "wav_path": f"data/raw/{sample_id}.wav",
                            "emotion_code": "01",
                            "label_id": 0,
                            "label_name": "neutral",
                            "pred_id": 0,
                            "pred_name": "neutral",
                            "probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                embedding_path = root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA" / split / f"{sample_id}.npy"
                np.save(embedding_path, np.ones((96,), dtype=np.float32))
                embedding_index_path = root / "external/audio_ser_speechbrain/speechbrain-develop-main/outputs/embeddings_ECAPA" / f"{split}_embedding_index.jsonl"
                embedding_index_path.write_text(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "av_key": "01-01-01-01-01-" + f"{actor_id:02d}",
                            "embedding_path": embedding_path.as_posix(),
                            "embedding_dim": 96,
                            "label_id": 0,
                            "split": split,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            layout = build_ecapa_layout(root)
            split_paths = build_fusion_ready_audio_tables(
                layout,
                expected_counts={"train": 1, "valid": 1, "test": 1},
            )

            self.assertTrue(split_paths["train"].is_file())
            with split_paths["train"].open(newline="", encoding="utf-8") as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(train_rows[0]["av_key"], "01-01-01-01-01-01")
            self.assertEqual(train_rows[0]["prob_neutral"], "1.0")


if __name__ == "__main__":
    unittest.main()
