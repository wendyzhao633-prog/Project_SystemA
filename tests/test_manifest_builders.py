import csv
from pathlib import Path
import tempfile
import unittest

from src.preprocess.manifests import (
    build_and_write_manifests,
    build_audio_manifest_rows,
    build_av_manifest_rows,
    build_video_manifest_rows,
)


class ManifestBuilderTests(unittest.TestCase):
    def test_audio_builder_supports_wrapped_audio_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_root = tmp_path / "data/raw/ravdess/audio_speech"
            actor_dir = audio_root / "Audio_Speech_Actors_01-24" / "Actor_01"
            actor_dir.mkdir(parents=True)
            sample = actor_dir / "03-01-05-01-01-01-01.wav"
            sample.touch()

            rows = build_audio_manifest_rows(audio_root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["wav_path"], sample.as_posix())
            self.assertEqual(rows[0]["label_name"], "angry")
            self.assertEqual(rows[0]["split"], "train")

    def test_video_builder_supports_wrapped_video_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_root = tmp_path / "data/raw/ravdess/video_speech"
            actor_dir = video_root / "Video_Speech_Actor_17" / "Actor_17"
            actor_dir.mkdir(parents=True)
            # The local dataset currently mixes full-AV (01) and video-only (02)
            # files. The builder should ignore 01 and keep canonical 02 rows.
            (actor_dir / "01-01-02-01-02-01-17.mp4").touch()
            sample = actor_dir / "02-01-02-01-02-01-17.mp4"
            sample.touch()

            rows = build_video_manifest_rows(video_root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_path"], sample.as_posix())
            self.assertEqual(rows[0]["label_name"], "calm")
            self.assertEqual(rows[0]["split"], "valid")

    def test_av_builder_pairs_using_av_key_instead_of_full_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_root = tmp_path / "data/raw/ravdess/audio_speech"
            video_root = tmp_path / "data/raw/ravdess/video_speech"

            audio_actor_dir = audio_root / "Audio_Speech_Actors_01-24" / "Actor_21"
            video_actor_dir = video_root / "Video_Speech_Actor_21" / "Actor_21"
            audio_actor_dir.mkdir(parents=True)
            video_actor_dir.mkdir(parents=True)

            (audio_actor_dir / "03-01-07-01-01-02-21.wav").touch()
            (video_actor_dir / "02-01-07-01-01-02-21.mp4").touch()

            audio_rows = build_audio_manifest_rows(audio_root)
            video_rows = build_video_manifest_rows(video_root)
            av_rows = build_av_manifest_rows(audio_rows, video_rows)

            self.assertEqual(len(av_rows), 1)
            self.assertEqual(av_rows[0]["av_key"], "01-07-01-01-02-21")
            self.assertEqual(av_rows[0]["audio_sample_id"], "03-01-07-01-01-02-21")
            self.assertEqual(av_rows[0]["video_sample_id"], "02-01-07-01-01-02-21")
            self.assertEqual(av_rows[0]["split"], "test")

    def test_build_and_write_manifests_writes_canonical_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_root = tmp_path / "data/raw/ravdess/audio_speech"
            video_root = tmp_path / "data/raw/ravdess/video_speech"
            output_root = tmp_path / "data/manifests"

            audio_actor_dir = audio_root / "Audio_Speech_Actors_01-24" / "Actor_01"
            video_actor_dir = video_root / "Video_Speech_Actor_01" / "Actor_01"
            audio_actor_dir.mkdir(parents=True)
            video_actor_dir.mkdir(parents=True)

            (audio_actor_dir / "03-01-03-02-02-01-01.wav").touch()
            (video_actor_dir / "02-01-03-02-02-01-01.mp4").touch()

            outputs = build_and_write_manifests(
                audio_root=audio_root,
                video_root=video_root,
                output_root=output_root,
            )

            self.assertEqual(output_root / "audio" / "audio_train.csv", outputs["audio"]["train"])
            self.assertEqual(output_root / "video" / "video_train.csv", outputs["video"]["train"])
            self.assertEqual(output_root / "av" / "av_train.csv", outputs["av"]["train"])

            with (output_root / "av" / "av_train.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["av_key"], "01-03-02-02-01-01")


if __name__ == "__main__":
    unittest.main()
