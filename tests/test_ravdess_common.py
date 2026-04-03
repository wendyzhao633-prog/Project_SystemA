import unittest

from src.common.ravdess import (
    EMOTION_CODE_TO_LABEL_ID,
    LABEL_ORDER,
    build_av_key,
    parse_ravdess_stem,
    split_for_actor,
)


class RavdessCommonTests(unittest.TestCase):
    def test_parse_ravdess_stem_exposes_contract_fields(self) -> None:
        parsed = parse_ravdess_stem("03-01-07-01-01-02-21")

        self.assertEqual(parsed.modality, "03")
        self.assertEqual(parsed.vocal_channel, "01")
        self.assertEqual(parsed.emotion_code, "07")
        self.assertEqual(parsed.label_id, 6)
        self.assertEqual(parsed.label_name, "disgust")
        self.assertEqual(parsed.actor_id, 21)
        self.assertEqual(parsed.split, "test")
        self.assertEqual(parsed.av_key, "01-07-01-01-02-21")

    def test_build_av_key_drops_modality_only(self) -> None:
        self.assertEqual(build_av_key("03-01-03-02-02-01-04"), "01-03-02-02-01-04")
        self.assertEqual(build_av_key("02-01-03-02-02-01-04"), "01-03-02-02-01-04")

    def test_fixed_label_mapping_and_actor_split(self) -> None:
        self.assertEqual(
            LABEL_ORDER,
            (
                "neutral",
                "calm",
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgust",
                "surprise",
            ),
        )
        self.assertEqual(EMOTION_CODE_TO_LABEL_ID["01"], 0)
        self.assertEqual(EMOTION_CODE_TO_LABEL_ID["08"], 7)
        self.assertEqual(split_for_actor(1), "train")
        self.assertEqual(split_for_actor(18), "valid")
        self.assertEqual(split_for_actor(24), "test")

    def test_parse_ravdess_stem_rejects_bad_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "MM-VV-EE-II-SS-RR-AA"):
            parse_ravdess_stem("03-01-07-01-01-02")


if __name__ == "__main__":
    unittest.main()
