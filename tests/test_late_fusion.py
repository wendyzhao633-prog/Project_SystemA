import unittest

import numpy as np

from src.fusion.data import join_audio_video_rows
from src.fusion.metrics import compute_classification_metrics
from src.fusion.run_late_fusion import build_weight_grid


class LateFusionTests(unittest.TestCase):
    def test_join_audio_video_rows_uses_av_key_and_prefixes_probs(self) -> None:
        audio_rows = [
            {
                "av_key": "01-03-01-01-01-17",
                "sample_id": "03-01-03-01-01-01-17",
                "wav_path": "audio.wav",
                "actor_id": "17",
                "split": "valid",
                "emotion_code": "03",
                "label_id": "2",
                "label_name": "happy",
                "intensity_code": "01",
                "statement_code": "01",
                "repetition_code": "01",
                "pred_id": "2",
                "pred_name": "happy",
                "embedding_path": "audio.npy",
                "embedding_dim": "96",
                "prob_neutral": "0.0",
                "prob_calm": "0.0",
                "prob_happy": "1.0",
                "prob_sad": "0.0",
                "prob_angry": "0.0",
                "prob_fearful": "0.0",
                "prob_disgust": "0.0",
                "prob_surprise": "0.0",
            }
        ]
        video_rows = [
            {
                "av_key": "01-03-01-01-01-17",
                "sample_id": "02-01-03-01-01-01-17",
                "video_path": "video.mp4",
                "actor_id": "17",
                "split": "valid",
                "emotion_code": "03",
                "label_id": "2",
                "label_name": "happy",
                "intensity_code": "01",
                "statement_code": "01",
                "repetition_code": "01",
                "pred_id": "2",
                "pred_name": "happy",
                "sequence_path": "video.npy",
                "sequence_feature_dim": "2048",
                "sequence_num_frames": "5",
                "sequence_video_id_source": "canonical_02",
                "prediction_video_id_source": "canonical_02",
                "late_fusion_ready": "1",
                "has_sequence": "1",
                "has_prediction": "1",
                "prob_neutral": "0.0",
                "prob_calm": "0.0",
                "prob_happy": "1.0",
                "prob_sad": "0.0",
                "prob_angry": "0.0",
                "prob_fearful": "0.0",
                "prob_disgust": "0.0",
                "prob_surprise": "0.0",
            }
        ]

        joined = join_audio_video_rows(audio_rows, video_rows, split="valid")

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["av_key"], "01-03-01-01-01-17")
        self.assertEqual(joined[0]["audio_sample_id"], "03-01-03-01-01-01-17")
        self.assertEqual(joined[0]["video_sample_id"], "02-01-03-01-01-01-17")
        self.assertEqual(joined[0]["audio_prob_happy"], "1.0")
        self.assertEqual(joined[0]["video_prob_happy"], "1.0")

    def test_compute_classification_metrics_reports_perfect_scores(self) -> None:
        y_true = np.asarray([0, 2], dtype=np.int64)
        probs = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        metrics = compute_classification_metrics(y_true, probs)

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 0.25)
        self.assertAlmostEqual(metrics["uar"], 0.25)
        self.assertEqual(metrics["pred_ids"], [0, 2])

    def test_build_weight_grid_includes_zero_and_one(self) -> None:
        grid = build_weight_grid(0.25)
        self.assertEqual(grid, [0.0, 0.25, 0.5, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
