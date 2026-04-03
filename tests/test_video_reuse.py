import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.preprocess.video_reuse import (
    aggregate_frame_predictions,
    build_fusion_ready_video_tables,
    build_video_reuse_layout,
    normalize_video_artifacts,
    select_preferred_record,
    validate_sequence_file,
)


VIDEO_MANIFEST_FIELDS = [
    "sample_id",
    "stem",
    "av_key",
    "video_path",
    "actor_id",
    "split",
    "emotion_code",
    "label_id",
    "label_name",
    "intensity_code",
    "statement_code",
    "repetition_code",
]


class VideoReuseTests(unittest.TestCase):
    def test_aggregate_frame_predictions_averages_probs(self) -> None:
        package_video_id = "02-01-03-01-01-01-17"
        rows = []
        for frame_idx in (10, 20, 30, 40, 50):
            rows.append(
                {
                    "sample_id": f"{package_video_id}_f{frame_idx:06d}",
                    "label_id": 2,
                    "label_name": "happy",
                    "probs": [0.05, 0.05, 0.70, 0.05, 0.05, 0.04, 0.03, 0.03],
                }
            )

        aggregated = aggregate_frame_predictions(rows, package_video_id=package_video_id)

        self.assertEqual(aggregated["prediction_video_id"], package_video_id)
        self.assertEqual(aggregated["prediction_frame_count"], 5)
        self.assertEqual(len(aggregated["probs"]), 8)
        self.assertEqual(aggregated["pred_id"], 2)
        self.assertEqual(aggregated["pred_name"], "happy")
        self.assertEqual(aggregated["av_key"], "01-03-01-01-01-17")

    def test_select_preferred_record_prefers_02_then_01(self) -> None:
        av_key = "01-03-01-01-01-17"
        preferred, source = select_preferred_record(
            av_key,
            {
                f"01-{av_key}": {"prediction_video_id": f"01-{av_key}"},
                f"02-{av_key}": {"prediction_video_id": f"02-{av_key}"},
            },
        )
        self.assertEqual(source, "canonical_02")
        self.assertEqual(preferred["prediction_video_id"], f"02-{av_key}")

        fallback, source = select_preferred_record(
            av_key,
            {f"01-{av_key}": {"prediction_video_id": f"01-{av_key}"}},
        )
        self.assertEqual(source, "fallback_01")
        self.assertEqual(fallback["prediction_video_id"], f"01-{av_key}")

    def test_validate_sequence_file_rejects_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.npy"
            np.save(path, np.zeros((4, 2048), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape"):
                validate_sequence_file(path)

    def test_normalize_and_build_tables_surface_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence_root = root / "external/video_face_branch/early_fusion_facial_v2/fusion_pack_5f/sequences"
            sequence_root.mkdir(parents=True)
            package_root = root / "external/video_face_branch/early_fusion_facial_v2"
            manifest_root = root / "data/manifests/video"
            manifest_root.mkdir(parents=True)

            expected_counts = {"train": 2, "valid": 1, "test": 1}

            split_rows = {
                "train": [
                    self._manifest_row("02-01-01-01-01-01-01", "train"),
                    self._manifest_row("02-01-01-01-01-01-15", "train"),
                ],
                "valid": [self._manifest_row("02-01-01-01-01-01-17", "valid")],
                "test": [self._manifest_row("02-01-01-01-01-01-21", "test")],
            }
            for split, rows in split_rows.items():
                with (manifest_root / f"video_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=VIDEO_MANIFEST_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

            np.save(sequence_root / "02-01-01-01-01-01-01.npy", np.ones((5, 2048), dtype=np.float32))
            np.save(sequence_root / "01-01-01-01-01-01-17.npy", np.ones((5, 2048), dtype=np.float32))
            np.save(sequence_root / "02-01-01-01-01-01-21.npy", np.ones((5, 2048), dtype=np.float32))

            self._write_prediction_file(
                package_root / "predictions_train.jsonl",
                "02-01-01-01-01-01-01",
                label_id=0,
                label_name="neutral",
            )
            self._write_prediction_file(
                package_root / "predictions_val.jsonl",
                "01-01-01-01-01-01-17",
                label_id=0,
                label_name="neutral",
            )
            self._write_prediction_file(
                package_root / "predictions_test.jsonl",
                "02-01-01-01-01-01-21",
                label_id=0,
                label_name="neutral",
            )

            layout = build_video_reuse_layout(root)
            result = normalize_video_artifacts(layout, expected_counts=expected_counts)
            split_paths = build_fusion_ready_video_tables(layout, expected_counts=expected_counts)

            self.assertFalse(result["audit"]["late_fusion_ready"])
            self.assertEqual(
                result["audit"]["selected_against_canonical_manifest"]["train"]["missing_sequence_count"],
                1,
            )
            self.assertEqual(
                result["audit"]["selected_against_canonical_manifest"]["valid"]["sequence_fallback_to_01"],
                1,
            )

            with split_paths["train"].open(newline="", encoding="utf-8") as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertEqual(len(train_rows), 2)
            ready_row = next(row for row in train_rows if row["sample_id"] == "02-01-01-01-01-01-01")
            missing_row = next(row for row in train_rows if row["sample_id"] == "02-01-01-01-01-01-15")
            self.assertEqual(ready_row["late_fusion_ready"], "1")
            self.assertEqual(ready_row["sequence_feature_dim"], "2048")
            self.assertEqual(ready_row["prediction_frame_count"], "5")
            self.assertEqual(missing_row["late_fusion_ready"], "0")
            self.assertEqual(missing_row["artifact_status"], "missing_both")

            with split_paths["valid"].open(newline="", encoding="utf-8") as handle:
                valid_rows = list(csv.DictReader(handle))
            self.assertEqual(valid_rows[0]["sequence_video_id_source"], "fallback_01")
            self.assertEqual(valid_rows[0]["prediction_video_id_source"], "fallback_01")

    @staticmethod
    def _manifest_row(sample_id: str, split: str) -> dict[str, str]:
        actor_id = int(sample_id.split("-")[-1])
        av_key = "-".join(sample_id.split("-")[1:])
        return {
            "sample_id": sample_id,
            "stem": sample_id,
            "av_key": av_key,
            "video_path": f"data/raw/ravdess/video_speech/Actor_{actor_id:02d}/{sample_id}.mp4",
            "actor_id": str(actor_id),
            "split": split,
            "emotion_code": "01",
            "label_id": "0",
            "label_name": "neutral",
            "intensity_code": "01",
            "statement_code": "01",
            "repetition_code": "01",
        }

    @staticmethod
    def _write_prediction_file(path: Path, package_video_id: str, *, label_id: int, label_name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for frame_idx in (11, 22, 33, 44, 55):
            records.append(
                {
                    "sample_id": f"{package_video_id}_f{frame_idx:06d}",
                    "label_id": label_id,
                    "label_name": label_name,
                    "pred_id": label_id,
                    "pred_name": label_name,
                    "probs": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            )
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    unittest.main()
