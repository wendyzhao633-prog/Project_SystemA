import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.preprocess.video_actor15_backfill import (
    build_actor15_backfill_layout,
    EXPECTED_MISSING_ROWS,
    load_missing_actor15_rows,
    run_actor15_backfill,
    uniform_indices,
)


class VideoActor15BackfillTests(unittest.TestCase):
    def test_load_missing_actor15_rows_requires_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "video_missing_reusable.csv"
            fieldnames = [
                "split",
                "av_key",
                "sample_id",
                "actor_id",
                "label_id",
                "label_name",
                "missing_sequence",
                "missing_prediction",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(EXPECTED_MISSING_ROWS):
                    writer.writerow(
                        {
                            "split": "train",
                            "av_key": f"01-01-01-01-{index:02d}-15",
                            "sample_id": f"02-01-01-01-{index:02d}-15",
                            "actor_id": "15",
                            "label_id": "0",
                            "label_name": "neutral",
                            "missing_sequence": "1",
                            "missing_prediction": "1",
                        }
                    )

            rows = load_missing_actor15_rows(csv_path)
            self.assertEqual(len(rows), EXPECTED_MISSING_ROWS)
            self.assertEqual(rows[0]["actor_id"], "15")

    def test_uniform_indices_returns_five_unique_positions(self) -> None:
        indices = uniform_indices(109, 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(sorted(indices), indices)
        self.assertEqual(len(set(indices)), 5)

    def test_run_backfill_reuses_existing_artifacts_when_audit_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "data/processed/video/audit"
            audit_dir.mkdir(parents=True)
            (audit_dir / "video_missing_reusable.csv").write_text(
                "split,av_key,sample_id,actor_id,label_id,label_name,missing_sequence,missing_prediction\n",
                encoding="utf-8",
            )

            backfill_root = root / "data/processed/video/backfill_actor15"
            seq_dir = backfill_root / "sequence_embeddings_5f/train"
            seq_dir.mkdir(parents=True)
            pred_dir = backfill_root / "predictions_video_level"
            pred_dir.mkdir(parents=True)
            seq_index_dir = backfill_root / "sequence_index"
            seq_index_dir.mkdir(parents=True)

            prediction_rows = []
            sequence_rows = []
            for index in range(EXPECTED_MISSING_ROWS):
                sample_id = f"02-01-01-01-01-{index % 2 + 1:02d}-15-{index:02d}"
                av_key = f"01-01-01-01-{index % 2 + 1:02d}-15-{index:02d}"
                npy_path = seq_dir / f"{sample_id}.npy"
                npy_path.write_bytes(b"0")
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "av_key": av_key,
                        "split": "train",
                        "actor_id": 15,
                        "label_id": 0,
                        "label_name": "neutral",
                        "prediction_video_id": sample_id,
                        "prediction_video_id_source": "backfill_actor15",
                        "prediction_frame_count": 5,
                        "pred_id": 0,
                        "pred_name": "neutral",
                        "probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                )
                sequence_rows.append(
                    {
                        "sample_id": sample_id,
                        "av_key": av_key,
                        "split": "train",
                        "actor_id": 15,
                        "label_id": 0,
                        "label_name": "neutral",
                        "sequence_video_id": sample_id,
                        "sequence_video_id_source": "backfill_actor15",
                        "sequence_path": npy_path.as_posix(),
                        "sequence_num_frames": 5,
                        "sequence_feature_dim": 2048,
                    }
                )

            with (pred_dir / "video_backfill_train.jsonl").open("w", encoding="utf-8") as handle:
                for row in prediction_rows:
                    handle.write(f"{json.dumps(row)}\n")
            with (seq_index_dir / "video_backfill_train.jsonl").open("w", encoding="utf-8") as handle:
                for row in sequence_rows:
                    handle.write(f"{json.dumps(row)}\n")
            (backfill_root / "backfill_summary.json").write_text('{"device":"cuda:0"}', encoding="utf-8")

            layout = build_actor15_backfill_layout(root)
            summary = run_actor15_backfill(layout)

            self.assertEqual(summary["status"], "already_complete")
            self.assertEqual(summary["predictions_written"], EXPECTED_MISSING_ROWS)
            self.assertEqual(summary["sequences_written"], EXPECTED_MISSING_ROWS)


if __name__ == "__main__":
    unittest.main()
