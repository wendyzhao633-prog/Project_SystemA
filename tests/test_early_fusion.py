import csv
from pathlib import Path
import tempfile
import unittest
import wave

import cv2
import numpy as np
import torch

from src.datasets.early_fusion_dataset import EarlyFusionDataConfig, EarlyFusionDataset, load_manifest_examples
from src.models.early_fusion_3cnn import EarlyFusion3CNN, EarlyFusion3CNNConfig, ModalityFusionGate
from src.models.early_fusion_bilinear import EarlyFusionBilinear
from src.models.early_fusion_xattn import EarlyFusionXAttn
from src.train.run_early_fusion import EarlyFusionTrainConfig, build_scheduler


class EarlyFusionTests(unittest.TestCase):
    def test_dataset_loads_logmel_and_sampled_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "tone.wav"
            video_path = root / "clip.avi"
            manifest_path = root / "av_train.csv"

            waveform = torch.sin(torch.linspace(0, 20 * np.pi, 16000, dtype=torch.float32))
            pcm = (waveform.clamp(-1.0, 1.0).numpy() * 32767.0).astype(np.int16)
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(pcm.tobytes())

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 64),
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(8):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[..., 0] = frame_index * 10
                frame[..., 1] = 255 - frame_index * 10
                frame[..., 2] = 50
                writer.write(frame)
            writer.release()

            row = {
                "av_key": "01-03-01-01-01-01",
                "split": "train",
                "actor_id": "1",
                "label_id": "2",
                "label_name": "happy",
                "emotion_code": "03",
                "audio_sample_id": "03-01-03-01-01-01-01",
                "audio_path": "tone.wav",
                "video_sample_id": "02-01-03-01-01-01-01",
                "video_path": "clip.avi",
            }
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer_csv = csv.DictWriter(handle, fieldnames=list(row))
                writer_csv.writeheader()
                writer_csv.writerow(row)

            examples = load_manifest_examples(manifest_path, repo_root=root)
            dataset = EarlyFusionDataset(
                repo_root=root,
                examples=examples,
                config=EarlyFusionDataConfig(
                    audio_num_frames=64,
                    image_size=32,
                    enable_cache=False,
                ),
            )

            item = dataset[0]
            self.assertEqual(item["audio_input"].shape, (1, 80, 64))
            self.assertEqual(item["video_input"].shape, (5, 3, 32, 32))
            self.assertEqual(item["av_key"], "01-03-01-01-01-01")
            self.assertEqual(item["label_id"], 2)

    def test_model_variants_output_8way_logits(self) -> None:
        audio_input = torch.randn(2, 1, 80, 128)
        video_input = torch.randn(2, 5, 3, 64, 64)

        for model in (EarlyFusion3CNN(), EarlyFusionBilinear(), EarlyFusionXAttn()):
            logits = model(audio_input, video_input)
            self.assertEqual(logits.shape, (2, 8))

    def test_3cnn_gating_module_scales_channels_and_preserves_shapes(self) -> None:
        gate = ModalityFusionGate(channels=128, hidden_dim=32, scale_floor=0.5)
        audio_map = torch.randn(2, 128, 8, 8)
        video_map = torch.randn(2, 128, 8, 8)

        audio_scale, video_scale = gate.compute_scales(audio_map, video_map)
        gated_audio, gated_video = gate(audio_map, video_map)

        self.assertEqual(audio_scale.shape, (2, 128, 1, 1))
        self.assertEqual(video_scale.shape, (2, 128, 1, 1))
        self.assertEqual(gated_audio.shape, audio_map.shape)
        self.assertEqual(gated_video.shape, video_map.shape)
        self.assertTrue(torch.all(audio_scale >= 0.5))
        self.assertTrue(torch.all(audio_scale <= 1.5))
        self.assertTrue(torch.all(video_scale >= 0.5))
        self.assertTrue(torch.all(video_scale <= 1.5))

    def test_3cnn_with_gating_outputs_8way_logits(self) -> None:
        model = EarlyFusion3CNN(EarlyFusion3CNNConfig(gating_enabled=True, gating_hidden_dim=32))
        logits = model(torch.randn(2, 1, 80, 128), torch.randn(2, 5, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 8))

    def test_train_augmentations_preserve_tensor_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "tone.wav"
            video_path = root / "clip.avi"
            manifest_path = root / "av_train.csv"

            waveform = torch.sin(torch.linspace(0, 20 * np.pi, 16000, dtype=torch.float32))
            pcm = (waveform.clamp(-1.0, 1.0).numpy() * 32767.0).astype(np.int16)
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(pcm.tobytes())

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 64),
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(8):
                frame = np.full((64, 64, 3), fill_value=frame_index * 12, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            row = {
                "av_key": "01-03-01-01-01-01",
                "split": "train",
                "actor_id": "1",
                "label_id": "2",
                "label_name": "happy",
                "emotion_code": "03",
                "audio_sample_id": "03-01-03-01-01-01-01",
                "audio_path": "tone.wav",
                "video_sample_id": "02-01-03-01-01-01-01",
                "video_path": "clip.avi",
            }
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer_csv = csv.DictWriter(handle, fieldnames=list(row))
                writer_csv.writeheader()
                writer_csv.writerow(row)

            examples = load_manifest_examples(manifest_path, repo_root=root)
            dataset = EarlyFusionDataset(
                repo_root=root,
                examples=examples,
                config=EarlyFusionDataConfig(
                    audio_num_frames=64,
                    image_size=32,
                    enable_cache=False,
                    train_audio_gain_jitter_db=2.0,
                    train_specaug_freq_mask_width=4,
                    train_specaug_time_mask_width=8,
                    train_video_random_crop_scale_min=0.9,
                    train_video_horizontal_flip_prob=0.5,
                    train_video_brightness_jitter=0.1,
                    train_video_contrast_jitter=0.1,
                ),
            )

            item = dataset[0]
            self.assertEqual(item["audio_input"].shape, (1, 80, 64))
            self.assertEqual(item["video_input"].shape, (5, 3, 32, 32))

    def test_build_scheduler_supports_reduce_on_plateau(self) -> None:
        model = EarlyFusion3CNN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = build_scheduler(
            optimizer,
            EarlyFusionTrainConfig(
                scheduler_name="reduce_on_plateau",
                scheduler_factor=0.5,
                scheduler_patience=2,
                scheduler_min_lr=1e-6,
            ),
        )

        self.assertIsNotNone(scheduler)
        self.assertEqual(type(scheduler).__name__, "ReduceLROnPlateau")


if __name__ == "__main__":
    unittest.main()
