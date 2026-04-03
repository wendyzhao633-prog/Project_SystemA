import unittest

from src.eval.report_artifacts import (
    build_display_name,
    build_per_class_rows,
    build_report_model_id,
    diagnose_training_dynamics,
)


class ReportArtifactsTests(unittest.TestCase):
    def test_build_report_model_id_uses_run_name_for_early_fusion(self) -> None:
        self.assertEqual(
            build_report_model_id(
                family="early_fusion",
                model_name="early_fusion_3cnn",
                run_name="early_fusion_3cnn_gated",
            ),
            "early_fusion_3cnn_gated",
        )
        self.assertEqual(
            build_report_model_id(
                family="late_fusion",
                model_name="weighted_prob_avg",
                run_name="ecapa_video_reuse_baseline",
                mode="weighted_prob_avg",
            ),
            "late_weighted_prob_avg",
        )

    def test_build_display_name_marks_nondefault_early_run(self) -> None:
        self.assertEqual(
            build_display_name(
                family="early_fusion",
                model_name="early_fusion_3cnn",
                run_name="early_fusion_3cnn_gated",
            ),
            "early_fusion_3cnn (early_fusion_3cnn_gated)",
        )
        self.assertEqual(
            build_display_name(
                family="early_fusion",
                model_name="early_fusion_3cnn",
                run_name="early_fusion_3cnn",
            ),
            "early_fusion_3cnn",
        )

    def test_build_per_class_rows_uses_confusion_support(self) -> None:
        metrics = {
            "per_class_precision": {name: 1.0 for name in ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise")},
            "per_class_recall": {name: 0.5 for name in ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise")},
            "per_class_f1": {name: 0.66 for name in ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprise")},
            "confusion_matrix": [
                [2, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 0, 0, 0, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 6, 0, 0, 0],
                [0, 0, 0, 0, 0, 7, 0, 0],
                [0, 0, 0, 0, 0, 0, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 9],
            ],
        }

        rows = build_per_class_rows(metrics)

        self.assertEqual(rows[0]["label_name"], "neutral")
        self.assertEqual(rows[0]["support"], 2)
        self.assertEqual(rows[-1]["label_name"], "surprise")
        self.assertEqual(rows[-1]["support"], 9)

    def test_diagnose_training_dynamics_flags_overfitting(self) -> None:
        model_report = {
            "family": "early_fusion",
            "best_epoch": 6,
            "best_valid_uar": 0.52,
            "core_metrics": {"test_uar": 0.28},
        }
        history = [
            {"epoch": 1, "train_loss": 2.0, "valid_uar": 0.20},
            {"epoch": 2, "train_loss": 1.8, "valid_uar": 0.31},
            {"epoch": 3, "train_loss": 1.5, "valid_uar": 0.41},
            {"epoch": 4, "train_loss": 1.2, "valid_uar": 0.49},
            {"epoch": 5, "train_loss": 1.0, "valid_uar": 0.50},
            {"epoch": 6, "train_loss": 0.8, "valid_uar": 0.52},
            {"epoch": 7, "train_loss": 0.6, "valid_uar": 0.46},
            {"epoch": 8, "train_loss": 0.5, "valid_uar": 0.40},
        ]

        diagnostic = diagnose_training_dynamics(model_report, history)

        self.assertEqual(diagnostic["status"], "overfitting_after_peak")
        self.assertFalse(diagnostic["longer_training_recommended"])


if __name__ == "__main__":
    unittest.main()
